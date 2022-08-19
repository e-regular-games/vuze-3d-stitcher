#!/usr/bin/python

import sys
import getopt
import fisheye
import splice
import transform
import refine_seams
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure

def usage():
    print('VuzeMerge 3D 360 Image Generator')
    print('  By S. Ryan Edgar')
    print('  August 1st, 2022')
    print('')
    print('usage: vuze_merge.py -v -h -c config.dat')
    print('')
    print('-c,--config\t\t(required) Specify the config file.')
    print('-v,--verbose\t\tMore detailed output to the console.')
    print('-d,--display\t\tShow intermediate progress images. Enums separated by "|" or "all". {regression|exposure|fisheye|seams|matches}')
    print('-w,--write-equation\t\tWrite the alignment equation constants to the provided file.')
    print('-r,--read-equation\t\tRead the alignment equation constants to the provided file.')
    print('-h,--help\t\tDisplay this message.')
    print('\n')
    print('--Config File Format--    A comma separated value format file with various options.')
    print('')
    print('input,<image_prefix>\t\t\tThe prefix used by the Vuze camera.')
    print('output,<image>\t\t\t\tA file name without the extension.')
    print('radius,<pixels>\t\t\t\tNumber of pixels for all fisheye lenses.')
    print('aperture,<degrees>\t\t\tAperture angle of all fisheye lenses. (degrees)')
    print('resolution,<pixels>\t\t\tOutput verticle resolution.')
    print('exposure_match,<1-8>\t\t\tImage to use as reference for exposure histogram matching.')
    print('exposure_fuse,<image_prefix>\t\t\tFile name to include in the exposure fusion stack.')
    print('lens,<1-8>,<x_pixels>,<y_pixels>\tThe center of the fisheye for each lens.')
    print('\n')

class ProgramOptions:
    def __init__(self):
        self.verbose = False
        self.config = ""
        self.display = {}
        self.read_equation = ''
        self.write_equation = ''

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'd:hw:r:vc:',
            ['help', 'write-equation=', 'read-equation=', 'config=', 'verbose', 'display='])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            elif o in ("-v", "--verbose"):
                self.verbose = True
            elif o in ("-r", "--read-equation"):
                self.read_equation = a
            elif o in ("-w", "--write-equation"):
                self.write_equation = a
            elif o in ("-d", "--display"):
                for s in a.split('|'):
                    self.display[s] = True
            elif o in ("-h", "--help"):
                usage()
                exit(0)

    def valid(self):
        return len(self.config) > 0

class AdjustmentCoeffs:
    def __init__(self, fname, debug):
        self._fname = fname
        self._debug = debug

    def write(self, stitches, transforms):
        with open(self._fname, 'w') as f:
            for i, s in enumerate(stitches):
                f.write(','.join(['seam', str(i), 'phi'] + np.char.mod('%f', s[:,0]).tolist()) + '\n')
                f.write(','.join(['seam', str(i), 'theta'] + np.char.mod('%f', s[:,1]).tolist()) + '\n')

            for i, t in enumerate(transforms):
                f.write(','.join(['phi', str(i), str(t.phi_coeffs_order)] + np.char.mod('%f', t.phi_coeffs).tolist()) + '\n')
                f.write(','.join(['theta', str(i), str(t.theta_coeffs_order)] + np.char.mod('%f', t.theta_coeffs).tolist()) + '\n')

    def read(self):
        with open(self._fname, 'r') as f:
            stitches_phi = [None]*8
            stitches_theta = [None]*8
            transforms = [transform.Transform(self._debug) for t in range(8)]
            for l in f.readlines():
                cmd = l.strip().split(',')
                if cmd[0] == 'seam' and len(cmd) >= 3:
                    data = np.array([float(s) for s in cmd[3:]])
                    if cmd[2] == 'phi':
                        stitches_phi[int(cmd[1])] = data
                    elif cmd[2] == 'theta':
                        stitches_theta[int(cmd[1])] = data
                if cmd[0] == 'phi':
                    t = int(cmd[1])
                    transforms[t].phi_coeffs_order = int(cmd[2])
                    transforms[t].phi_coeffs = np.array([float(s) for s in cmd[3:]])
                if cmd[0] == 'theta':
                    t = int(cmd[1])
                    transforms[t].theta_coeffs_order = int(cmd[2])
                    transforms[t].theta_coeffs = np.array([float(s) for s in cmd[3:]])

            stitches = []
            for i in range(8):
                st = np.zeros((stitches_phi[i].shape[0], 2))
                st[:,0] = stitches_phi[i]
                st[:,1] = stitches_theta[i]
                stitches.append(st)

            return stitches, transforms

class Debug:
    def __init__(self, options):
        self.display = options.display
        self.verbose = options.verbose

    def enable(self, opt):
        return opt in self.display and self.display[opt]

class Config:
    def __init__(self, file_path):
        self.input = ""
        self.output = ""
        self.radius = 0
        self.aperture = 180
        self.resolution = 1080
        self.lens_centers = [(0,0)]*8
        self.exposure_match = 0
        self.exposure_fuse = []

        f = open(file_path, 'r')
        for l in f.readlines():
            cmd = l.strip().split(',')
            if cmd[0] == 'input' and len(cmd) == 2:
                self.input = cmd[1]
            if cmd[0] == 'output' and len(cmd) == 2:
                self.output = cmd[1]
            if cmd[0] == 'lens' and len(cmd) == 4:
                l = int(cmd[1]) - 1
                self.lens_centers[l] = (int(cmd[2]), int(cmd[3]))
            if cmd[0] == 'radius' and len(cmd) == 2:
                self.radius = int(cmd[1])
            if cmd[0] == 'resolution' and len(cmd) == 2:
                self.resolution = int(cmd[1])
            if cmd[0] == 'exposure_match' and len(cmd) == 2:
                self.exposure_match = int(cmd[1])
            if cmd[0] == 'aperture' and len(cmd) == 2:
                self.aperture = float(cmd[1])
            if cmd[0] == 'exposure_fuse' and len(cmd) == 2:
                self.exposure_fuse.append(cmd[1])

    def valid(self):
        return self.input != '' and self.radius != 0

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

def set_middle(img, value):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    img[:,middle] = value

def plot_lenses(images, title):
    f, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    f.canvas.manager.set_window_title(title)
    for i, img in enumerate(images):
        axs[int(i/3), i%3].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axs[int(i/3), i%3].axes.xaxis.set_ticklabels([])
        axs[int(i/3), i%3].axes.yaxis.set_ticklabels([])

def main():
    options = ProgramOptions()
    if not options.valid():
        usage()
        exit(1)

    config = Config(options.config)
    if not config.valid():
        usage()
        exit(1)

    debug = Debug(options)

    np.set_printoptions(suppress=True)
    print('loading images')
    images = []
    fish = fisheye.FisheyeImage(debug)
    fish.set_output_resolution(config.resolution)
    fish.restrict_to_image_angle(False)
    for l in range(1, 9):
        img = cv.imread(config.input + '_' + str(l) + '.JPG')
        img = np.rot90(img)
        fish.set_image(img, config.lens_centers[l-1], config.radius, config.aperture)
        images.append(fish.to_equirect())

    if debug.enable('fisheye'): plot_lenses(images, '1) Equirectangular')

    if len(config.exposure_fuse) > 0:
        for l in range(1, 9):
            print('fusing exposures lens ' + str(l))
            images_exp = [images[l-1]]
            for exp in config.exposure_fuse:
                img = cv.imread(exp + '_' + str(l) + '.JPG')
                img = np.rot90(img)
                fish.set_image(img, config.lens_centers[l-1], config.radius, config.aperture)
                images_exp.append(fish.to_equirect())

            alignMTB = cv.createAlignMTB()
            alignMTB.process(images_exp, images_exp)
            mergeMertens = cv.createMergeMertens()
            merged = np.clip(mergeMertens.process(images_exp) * 255, 0, 255)
            images[l-1] = merged.astype(np.uint8)
        if debug.enable('exposure'): plot_lenses(images, '2) Exposures Fused')

    if config.exposure_match != 0:
        print('matching exposure')
        # match histograms using the reference image.
        ref = get_middle(images[config.exposure_match])
        for i in range(len(images)):
            if i != (config.exposure_match - 1):
                mid = get_middle(images[i])
                set_middle(images[i], exposure.match_histograms(mid, ref, channel_axis=2))
        if debug.enable('exposure'): plot_lenses(images, '3) Exposures Matched')

    stitches = []
    ts = []
    if options.read_equation != '':
        print('loading seams')
        stitches, ts = AdjustmentCoeffs(options.read_equation, debug).read()
    else:
        print('computing seams')
        seam = refine_seams.RefineSeams(images, debug)
        stitches = seam.align()
        ts = seam._transforms;

    splice_left = splice.SpliceImages(images[0:8:2], debug)
    splice_right = splice.SpliceImages(images[1:8:2], debug)

    for s in range(4):
        st_l = stitches[2*s].copy()
        st_l[:,1] += s * math.pi / 2
        splice_left.set_stitch(s, st_l)
        splice_left.set_transform(s, ts[2*s])

        st_r = stitches[2*s+1].copy()
        st_r[:,1] += s * math.pi / 2
        splice_right.set_stitch(s, st_r)
        splice_right.set_transform(s, ts[2*s+1])

    print('generate left eye')
    left = splice_left.generate(config.resolution)

    print('generate right eye')
    right = splice_right.generate(config.resolution)

    combined = np.concatenate([left, right])
    cv.imwrite(config.output + '.JPG', combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    if options.write_equation != '':
        AdjustmentCoeffs(options.write_equation, debug).write(stitches, ts)

    plt.show()


if __name__ == '__main__':
    main()
