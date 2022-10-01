#!/usr/bin/python

import color_correction
import sys
import getopt
import fisheye
from format_vr import FormatVR
import splice
import transform
import refine_seams
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure
import threading

def usage():
    print('VuzeMerge 3D 360 Image Generator')
    print('  By S. Ryan Edgar')
    print('  August 1st, 2022')
    print('')
    print('usage: vuze_merge.py -v -h -c config.dat')
    print('')
    print('-c,--config\t\tSpecify the config file.')
    print('-i,--image\t\tOverride the input and output config options.')
    print('-I,--in\t\tOverride the input config option.')
    print('-O,--out\t\tOverride the output config option.')
    print('-f,--format\t\tA "," separated list of output formats: {gpano, stereo, over-under}')
    print('-w,--write-equation\t\tWrite the alignment equation constants to the provided file.')
    print('-r,--read-equation\t\tRead the alignment equation constants to the provided file.')
    print('-v,--verbose\t\tMore detailed output to the console.')
    print('-d,--display\t\tShow intermediate progress images. Enums separated by ",". {regression, exposure, fisheye, seams, matches}')
    print('-h,--help\t\tDisplay this message.')
    print('\n')
    print('--Config File Format--    A comma separated value format file with various options.')
    print('')
    print('input,<image_prefix>\t\t\tThe prefix used by the Vuze camera.')
    print('output,<image>\t\t\t\tA file name without the extension.')
    print('format,<fmt>\t\t\t\tEnable output format: {gpano, stereo, over-under}')
    print('radius,<pixels>\t\t\t\tNumber of pixels for all fisheye lenses.')
    print('aperture,<degrees>\t\t\tAperture angle of all fisheye lenses. (degrees)')
    print('resolution,<pixels>\t\t\tOutput verticle resolution.')
    print('exposure_match,<1-8>\t\t\tImage to use as reference for exposure histogram matching.')
    print('exposure_fuse,<image_prefix>\t\tFile name to include in the exposure fusion stack.')
    print('color_correction,mean\t\t\tAdjust colors of all lenses using the mean between lenses.')
    print('color_correction,seams,<dist_deg>\tUse the mean between lenses, but fade the effect from the seam.')
    print('color_correction,kmeans\tCompute the adjustment based on the kmeans colors.')
    print('contrast_equ,<clip>,<gridx>,<gridy>\tEnable adaptive hsv-value histogram equalization.')
    print('seam,blend,<margin>\t\t\tBlend by taking a linear weighted average across the margin about the seam.')
    print('seam,pyramid,<depth>\t\t\tBlend using Laplacian Pyramids to the specified depth. Experimental: causes color and image distortion.')
    print('lens,<1-8>,<x_pixels>,<y_pixels>\tThe center of the fisheye for each lens.')
    print('\n')

class ProgramOptions:
    def __init__(self):
        self.verbose = False
        self.config = ""
        self.display = {}
        self.read_equation = ''
        self.write_equation = ''
        self.image_override = ''
        self.in_override = ''
        self.out_override = ''
        self.format = []

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'd:hw:r:vc:i:f:I:O:',
            [
                'help',
                'image=',
                'in=',
                'out=',
                'format=',
                'write-equation=',
                'read-equation=',
                'config=',
                'verbose',
                'display='
            ])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            elif o in ("-v", "--verbose"):
                self.verbose = True
            elif o in ("-r", "--read-equation"):
                self.read_equation = a
            elif o in ("-w", "--write-equation"):
                self.write_equation = a
            elif o in ("-i", "--image"):
                self.image_override = a
            elif o in ("-I", "--in"):
                self.in_override = a
            elif o in ("-O", "--out"):
                self.out_override = a
            elif o in ("-f", "--format"):
                self.format = a.split(",")
            elif o in ("-d", "--display"):
                for s in a.split(','):
                    self.display[s] = True
            elif o in ("-h", "--help"):
                usage()
                exit(0)

    def valid(self):
        return len(self.config) > 0 or len(self.image_override) > 0 \
            or (len(self.in_override) > 0 and len(self.out_override) > 0)


class AdjustmentCoeffs:
    def __init__(self, fname, debug):
        self._fname = fname
        self._debug = debug

    def write(self, stitches, transforms, matches):
        with open(self._fname, 'w') as f:
            for i, s in enumerate(stitches):
                f.write(','.join(['seam', str(i), 'phi'] + np.char.mod('%f', s[:,0]).tolist()) + '\n')
                f.write(','.join(['seam', str(i), 'theta'] + np.char.mod('%f', s[:,1]).tolist()) + '\n')

            for i, t in enumerate(transforms):
                f.write(','.join(['phi', str(i), str(t.phi_coeffs_order)] + np.char.mod('%f', t.phi_coeffs).tolist()) + '\n')
                f.write(','.join(['theta', str(i), str(t.theta_coeffs_order)] + np.char.mod('%f', t.theta_coeffs).tolist()) + '\n')

            for i, m in enumerate(matches):
                l = m.shape[0] * m.shape[1]
                f.write(','.join(['matches', str(i)] + np.char.mod('%f', m.reshape((l))).tolist()) + '\n')

    def read(self):
        with open(self._fname, 'r') as f:
            stitches_phi = [None]*8
            stitches_theta = [None]*8
            matches = [None]*4
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
                if cmd[0] == 'matches':
                    i = int(cmd[1])
                    matches[i] = np.array([float(s) for s in cmd[2:]]).reshape((int((len(cmd)-2) / 8), 8))

            stitches = []
            for i in range(8):
                st = np.zeros((stitches_phi[i].shape[0], 2))
                st[:,0] = stitches_phi[i]
                st[:,1] = stitches_theta[i]
                stitches.append(st)

            return stitches, transforms, matches

class Debug:
    def __init__(self, options=None):
        self.display = options.display if options is not None else {}
        self.verbose = options.verbose if options is not None else False
        self._window_title = ''
        self._figures = {}
        self._subplot = (1, 1, 1)

    def enable(self, opt):
        return opt in self.display and self.display[opt]

    def figure(self, id, reset=False):
        if id not in self._figures or reset:
            self._figures[id] = plt.figure()
            self._figures[id].canvas.set_window_title(self._window_title + ' - ' + id)
        return self._figures[id]

    def subplot(self, id, projection=None):
        return self.figure(id) \
                   .add_subplot(self._subplot[0], self._subplot[1], \
                                self._subplot[2], projection=projection)

    # create a new figure window for each figure
    def window(self, prefix):
        # avoid the full clone because we want to reset the figures and subplot
        d = Debug(self)
        d._window_title = self._window_title + ' - ' + prefix
        return d

    def set_subplot(self, h, w, idx):
        d = self.clone()

        # parent row and column
        rp = int((self._subplot[2] - 1) / self._subplot[1])
        cp = (self._subplot[2] - 1) % self._subplot[1]

        # child row and column
        rc = int((idx - 1) / w)
        cc = (idx - 1) % w

        rowcells = h * w * self._subplot[1]
        i = rp * rowcells + rc * w * self._subplot[1] + cp * w + cc + 1

        d._subplot = (self._subplot[0] * h, self._subplot[1] * w, i)
        return d

    def clone(self):
        d = Debug(self)
        d._figures = self._figures
        d._subplot = self._subplot
        d._window_title = self._window_title
        return d

    def none(self):
        return Debug()

class Config:
    def __init__(self, file_path):
        self.input = ""
        self.output = ""
        self.radius = 734
        self.format = {}
        self.aperture = 179
        self.resolution = 2160
        self.lens_centers = [
            (544,778),
            (560,778),
            (500,800),
            (570,770),
            (490,830),
            (580,800),
            (544,778),
            (580,800)
        ]
        self.exposure_match = 0
        self.exposure_fuse = []
        self.color_correct = 'kmeans'
        self.color_seams = 15 * math.pi / 180
        self.contrast_equ = None
        self.seam_blend_margin = 8 * math.pi / 180
        self.seam_pyramid_depth = 0
        self.denoise = ()

        if len(file_path) > 0:
            f = open(file_path, 'r')
            for l in f.readlines():
                cmd = l.strip().split(',')
                self.process_line(cmd)

        if len(self.format) == 0:
            self.format['over-under'] = True

    def process_line(self, cmd):
        if cmd[0] == 'input' and len(cmd) == 2:
            self.input = cmd[1]
        elif cmd[0] == 'output' and len(cmd) == 2:
            self.output = cmd[1]
        elif cmd[0] == 'lens' and len(cmd) == 4:
            l = int(cmd[1]) - 1
            self.lens_centers[l] = (int(cmd[2]), int(cmd[3]))
        elif cmd[0] == 'radius' and len(cmd) == 2:
            self.radius = int(cmd[1])
        elif cmd[0] == 'resolution' and len(cmd) == 2:
            self.resolution = int(cmd[1])
        elif cmd[0] == 'exposure_match' and len(cmd) == 2:
            self.exposure_match = int(cmd[1])
        elif cmd[0] == 'aperture' and len(cmd) == 2:
            self.aperture = float(cmd[1])
        elif cmd[0] == 'denoise' and len(cmd) == 4:
            self.denoise = (int(cmd[1]), int(cmd[2]), int(cmd[3]))
        elif cmd[0] == 'exposure_fuse' and len(cmd) == 2:
            self.exposure_fuse.append(cmd[1])
        elif cmd[0] == 'color_correction' and len(cmd) >= 2:
            self.color_correct = cmd[1]
            if cmd[1] == 'seams' and len(cmd) == 3:
                self.color_seams = float(cmd[2]) * math.pi / 180
        elif cmd[0] == 'contrast_equ' and len(cmd) == 4:
            self.contrast_equ = (float(cmd[1]), int(cmd[2]), int(cmd[3]))
        elif cmd[0] == 'seam' and len(cmd) == 3:
            if cmd[1] == 'blend':
                self.seam_blend_margin = float(cmd[2]) * math.pi / 180
                self.seam_pyramid_depth = 0
            elif cmd[1] == 'pyramid':
                self.seam_pyramid_depth = int(cmd[2])
                self.seam_blend_margin = 0
        elif cmd[0] == 'format' and len(cmd) == 2:
                self.format[cmd[1]] = True

    def valid(self):
        return self.input != '' and self.output != ''

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

class ComputeSplice(threading.Thread):
    def __init__(self, splice, resolution, margin):
        threading.Thread.__init__(self)
        self._splice = splice
        self._resolution = resolution
        self._margin = margin

        self.result = None

    def run(self):
        self.result = self._splice.generate_fade(self._resolution, self._margin)

def main():
    options = ProgramOptions()
    if not options.valid():
        usage()
        print('\nERROR: The image file or config must be specified.')
        exit(1)

    # generate the config, if the file is empty a default config is used.
    # override some of the config options which can be provide by the command line
    config = Config(options.config)
    if len(options.image_override) > 0:
        config.input = options.image_override
        config.output = options.image_override
    if len(options.in_override) > 0:
        config.input = options.in_override
    if len(options.out_override) > 0:
        config.output = options.out_override

    if len(options.format) > 0:
        config.format = {}
        for c in options.format:
            config.format[c] = True

    if not config.valid():
        usage()
        print('The config file is invalid, missing input or output.')
        exit(1)

    debug = Debug(options)

    np.set_printoptions(suppress=True, threshold=sys.maxsize)
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

    if debug.enable('fisheye'): plot_lenses(images, 'Equirectangular')

    if len(config.exposure_fuse) > 0:
        for l in range(1, 9):
            print('fusing exposures lens ' + str(l))
            images_exp = [get_middle(images[l-1])]
            for exp in config.exposure_fuse:
                img = cv.imread(exp + '_' + str(l) + '.JPG')
                img = np.rot90(img)
                fish.set_image(img, config.lens_centers[l-1], config.radius, config.aperture)
                images_exp.append(get_middle(fish.to_equirect()))

            mergeMertens = cv.createMergeMertens()
            merged = np.clip(mergeMertens.process(images_exp) * 255, 0, 255)
            set_middle(images[l-1], merged.astype(np.uint8))
        if debug.enable('exposure'): plot_lenses(images, 'Exposures Fused')

    if config.exposure_match != 0:
        print('matching exposure')
        # match histograms using the reference image.
        ref = get_middle(images[config.exposure_match])
        for i in range(len(images)):
            if i != (config.exposure_match - 1):
                mid = get_middle(images[i])
                set_middle(images[i], exposure.match_histograms(mid, ref, channel_axis=2))
        if debug.enable('exposure'): plot_lenses(images, 'Exposures Matched')

    if len(config.denoise) == 3:
        print('denoising images')
        for i, img in enumerate(images):
            mid = get_middle(img)
            midd = cv.fastNlMeansDenoisingColored(mid, None, \
                                                  config.denoise[0], \
                                                  config.denoise[0], \
                                                  config.denoise[1], \
                                                  config.denoise[2])
            set_middle(images[i], midd)

        if debug.enable('denoise'): plot_lenses(images, 'Denoise')

    stitches = []
    ts = []
    matches = []
    cc = None
    if options.read_equation != '':
        print('loading seams')
        stitches, ts, matches = AdjustmentCoeffs(options.read_equation, debug).read()
    else:
        print('computing seams')
        seam = refine_seams.RefineSeams(images, debug)
        stitches, matches = seam.align()
        ts = seam._transforms

    color = color_correction.ColorCorrectionRegion(images, ts, config, debug)
    if config.color_correct == 'mean':
        print('computing color mean')
        cc = color.match_colors(matches, stitches)
    elif config.color_correct == 'seams':
        print('computing color seam fade')
        cc = color.fade_colors(matches, stitches, config.color_seams)
    elif config.color_correct == 'kmeans':
        print('computing color mean - kmeans')
        cc = color.match_colors_kmeans(matches, stitches)

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

        if cc is not None:
            splice_left.set_color_transform(s, cc[2*s])
            splice_right.set_color_transform(s, cc[2*s+1])


    if config.seam_pyramid_depth > 0:
        print('generate left eye - pyramid blending')
        left = splice_left.generate_pyramid(config.resolution, config.seam_pyramid_depth)
        print('generate right eye - pyramid blending')
        right = splice_right.generate_pyramid(config.resolution, config.seam_pyramid_depth)
    elif config.seam_blend_margin >= 0:
        print('generate eyes - seam blending')
        t_left = ComputeSplice(splice_left, config.resolution, config.seam_blend_margin)
        t_right = ComputeSplice(splice_right, config.resolution, config.seam_blend_margin)

        # must be done one at a time else, will run the computer out of memory.
        t_left.start()
        t_left.join()

        t_right.start()
        t_right.join()

        left = t_left.result
        right = t_right.result


    if config.contrast_equ is not None:
        print('equilizing value histogram')
        clahe = cv.createCLAHE(clipLimit=config.contrast_equ[0], \
                               tileGridSize=config.contrast_equ[1:3])
        combined = np.concatenate([left, right])
        combined_hsv = cv.cvtColor(combined, cv.COLOR_BGR2HSV)
        combined_hsv[:,:,2] = clahe.apply(combined_hsv[:,:,2:3])
        combined = cv.cvtColor(combined_hsv, cv.COLOR_HSV2BGR)
        left = combined[:left.shape[0]]
        right = combined[left.shape[0]:]

    print('writing output files')
    fvr = FormatVR(left, right)
    if 'stereo' in config.format:
        fvr.write_stereo(config.output + '_left.JPG', config.output + '_right.JPG')
    if 'over-under' in config.format:
        fvr.write_over_under(config.output + '.JPG')
    if 'gpano' in config.format:
        fvr.write_cardboard(config.output + '_gpano.JPG')

    if options.write_equation != '':
        AdjustmentCoeffs(options.write_equation, debug).write(stitches, ts, matches)

    plt.show()


if __name__ == '__main__':
    main()
