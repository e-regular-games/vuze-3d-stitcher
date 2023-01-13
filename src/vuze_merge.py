#!/usr/bin/python

import color_correction
import sys
import os
import getopt
from format_vr import FormatVR
import splice
import transform
import refine_seams
import json
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure
import threading
from exiftool import ExifToolHelper
from image_loader import ImageLoader
from image_loader import CalibrationParams
from depth_mesh import DepthMesh
from depth_mesh import DepthCalibration

def usage():
    print('VuzeMerge 3D 360 Image Generator')
    print('  By S. Ryan Edgar')
    print('  August 1st, 2022')
    print('')
    print('usage: vuze_merge.py -v -h -c config.dat')
    print('')

    print('-a,--alignment <file>\t\tThe alignment coefficients for image processing. This file will be updated by operations such as yaml-config, ellipse-coeffs, write-coeffs, and depth.')
    print('--yaml-config <file>\t\tSpecify the yml config file generated by the camera.')
    print('--ellipse-coeffs\t\tDetermine the coefficients for fisheye/sensor offset and skew from the provided images. Default is to read from the equation coefficients file.')
    print('--depth <action>\t\t\tRead the color/distance information from <file_prefix>.json (same as input image) and add the coordinate data to the provided coefficients file provided by alignment. All other options will be ignored, except for the input image which is required. Actions: {add-only, regression}')
    print('--write-coeffs\t\tWrite the alignment equation constants to the provided alingment file.')
    print('--ignore-alignment-seams\t\t\tDo not read previous seam data from the alignment file. Fresh seam data will be written to the file, if write-coeffs is enabled.')
    print('')

    print('-c,--config <file>\t\tSpecify the config file.')
    print('-i,--image <file_prefix>\t\tOverride the input and output config options.')
    print('-I,--in <file_prefix>\t\tOverride the input config option.')
    print('-O,--out <file_prefix>\t\tOverride the output config option.')
    print('-C,--color <type>\t\tOverride the color_correction config option.')
    print('-q,--quality <verticle_pixels>\t\tVertical resolution of the full output image.')
    print('-F,--fast <level>\t\t\tSkip recomputing seams and color correction. Levels {1: skip seams, 2: skip color, 3: skip seams and color}.')
    print('-f,--format <spec>\t\tA "," separated list of output formats: {gpano, stereo, over-under}. See the config file description below for more information.')
    print('')

    print('-l,--load-processed <file>\t\tA previously generated over-under 3D 360 image, with extension "image.JPG". Only the format output option is enabled. (Will not overwrite.)')
    print('')

    print('-v,--verbose\t\tMore detailed output to the console.')
    print('-d,--display <enums>\t\tShow intermediate progress images. Enums separated by ",". {regression, exposure, fisheye, seams, matches}')
    print('-h,--help\t\tDisplay this message.')
    print('\n')
    print('--Config File Format--    A comma separated value format file with various options.')
    print('')
    print('input,<image_prefix>\t\t\tThe prefix used by the Vuze camera.')
    print('output,<image>\t\t\t\tA file name without the extension.')
    print('format,<fmt>\t\t\t\tEnable output format: {gpano[:<spec>], stereo, over-under[:<spec>], anaglyph:<spec>}. This option can be repeated.\n\t\t\t\t\t\t anaglyph:<spec> --> "anaglyph:<fov>:<phi>:<theta>:<x-res>:<y-res>" with angles in degrees\n\t\t\t\t\t\t gpano:<spec> --> "gpano:<latitude>:<longitude>:<vertical-fov>:<horizontal-fov>" with angles in degrees\n\t\t\t\t\t\t over-under:<spec> --> "over-under:<fov>:<phi>:<theta>:<xres>:<yres>" with angles in degrees')
    print('radius,<pixels>\t\t\t\tNumber of pixels for all fisheye lenses.')
    print('aperture,<degrees>\t\t\tAperture angle of all fisheye lenses. (degrees)')
    print('view_direction,<degrees>\t\tThe initial view direction which will be at the center of the generated image. (degrees)')
    print('resolution,<pixels>\t\t\tOutput vertical resolution.')
    print('exposure_match,<1-8>\t\t\tImage to use as reference for exposure histogram matching.')
    print('exposure_fuse,<image_prefix>\t\tFile name to include in the exposure fusion stack.')
    print('color_correction,mean-seams\t\t\tAdjust colors of all lenses using the mean between lenses along the entire seam.')
    print('color_correction,mean-matches\t\t\tAdjust colors of all lenses using the mean between lenses for matching points between images.')
    print('accel_align,<enabled>\t\t\tUse the accelerometer to rotate and translate the lens. default: false')
    print('contrast_equ,<clip>,<gridx>,<gridy>\tEnable adaptive hsv-value histogram equalization.')
    print('seam,blend,<margin>\t\t\tBlend by taking a linear weighted average across the margin about the seam.')
    print('seam,pyramid,<depth>\t\t\tBlend using Laplacian Pyramids to the specified depth. Experimental: causes color and image distortion.')
    print('super_resolution,<image>\t\tSpecified at least 2 times. Each image will be merged into the result.')
    print('super_resolution,<bucket>,<image>\t\tSpecified at least 4 times, 2 separate buckets. Buckets can be arbitrary, images within a bucket will be merged, then buckets will be exposure merged using Mertens.')
    print('super_resolution_config,<variable>,<value>\t\tSupport variables {outlier_limit,sharpen}. Outlier limit is a number of standard deviations, and sharpen is "true" or "false". By default sharpen is "false" and outlier limit is not applied.')
    print('lens,<1-8>,<x_pixels>,<y_pixels>\tThe center of the fisheye for each lens.')
    print('\n')

class ProgramOptions:
    def __init__(self):
        self.verbose = False
        self.config = ""

        self.alignment_file = None
        self.yaml_config = None
        self.depth = None
        self.recalc_ellipse = False
        self.write_coeffs = False
        self.reuse_seams = True

        self.display = {}
        self.image_override = None
        self.in_override = ''
        self.out_override = ''
        self.format = []
        self.resolution = 0
        self.color_correction = None
        self.fast = 0
        self.load_processed = None

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'd:hF:w:r:vc:i:f:I:O:q:C:a:l:',
            [
                'help',
                'image=',
                'in=',
                'out=',
                'format=',
                'write-coeffs',
                'write-equation=',
                'read-equation=',
                'config=',
                'verbose',
                'display=',
                'quality=',
                'color=',
                'ellipse-coeffs',
                'fast=',
                'yaml-config=',
                'depth=',
                'alignment=',
                'ignore-alignment-seams',
                'write-coeffs',
                'load-processed='
            ])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            elif o == "--yaml-config":
                self.yaml_config = a
            elif o == "--depth":
                self.depth = a
            elif o in ("-a", "--alignment"):
                self.alignment_file = a
            elif o == "--ignore-alignment-seams":
                self.reuse_seams = False
            elif o == "--write-coeffs":
                self.write_coeffs = True
            elif o == "--ellipse-coeffs":
                self.recalc_ellipse = True
            elif o in ("-r", "--read-equation"):
                self.alignment_file = a
            elif o in ("-w", "--write-equation"):
                self.alignment_file = a
                self.write_coeffs = True
            elif o in ("-l", "--load-processed"):
                self.load_processed = a
            elif o in ("-v", "--verbose"):
                self.verbose = True
            elif o in ("-F", "--fast"):
                self.fast = int(a)
            elif o in ("-i", "--image"):
                self.image_override = a
            elif o in ("-I", "--in"):
                self.in_override = a
            elif o in ("-O", "--out"):
                self.out_override = a
            elif o in ("-C", "--color"):
                self.color_correction = a
            elif o in ("-q", "--quality"):
                self.resolution = int(a)
            elif o in ("-f", "--format"):
                for f in a.split(","):
                    self.format.append(f)
            elif o in ("-d", "--display"):
                for s in a.split(','):
                    self.display[s] = True
            elif o in ("-h", "--help"):
                usage()
                exit(0)

    def valid(self):
        return len(self.config) > 0 or self.image_override is not None \
            or (len(self.in_override) > 0 and len(self.out_override) > 0) \
            or self.load_processed is not None


class Debug:
    def __init__(self, options=None):
        self.display = options.display if options is not None else {}
        self.verbose = options.verbose if options is not None else False
        self.enable_threads = len(self.display) == 0
        self._window_title = ''
        self._figures = {}
        self._subplot = (1, 1, 1)

    def enable(self, opt):
        return opt in self.display and self.display[opt]

    def figure(self, id, reset=False):
        if id not in self._figures or reset:
            self._figures[id] = plt.figure()
            self._figures[id].canvas.manager.set_window_title(self._window_title + ' - ' + id)
        return self._figures[id]

    def subplot(self, id, projection=None):
        return self.figure(id) \
                   .add_subplot(self._subplot[0], self._subplot[1], \
                                self._subplot[2], projection=projection)

    # create a new figure window for each figure
    def window(self, prefix):
        # avoid the full clone because we want to reset the figures and subplot
        d = self.clone()
        d._figures = {}
        d._subplot = (1, 1, 1)
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
        d.enable_threads = self.enable_threads
        return d

    def none(self):
        return Debug()

class Config:
    def __init__(self, file_path):
        self.input = ""
        self.output = ""
        self.format = {}
        self.aperture = 180
        self.resolution = 2160
        self.view_direction = 180
        self.exposure_match = 0
        self.exposure_fuse = []
        self.super_res = []
        self.super_res_config = {}
        self.super_res_buckets = {}
        self.color_correct = 'mean-seams'
        self.accel_align = True
        self.contrast_equ = None
        self.seam_blend_margin = 5 * math.pi / 180
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
        elif cmd[0] == 'resolution' and len(cmd) == 2:
            self.resolution = int(cmd[1])
        elif cmd[0] == 'view_direction' and len(cmd) == 2:
            self.view_direction = float(cmd[1])
        elif cmd[0] == 'exposure_match' and len(cmd) == 2:
            self.exposure_match = int(cmd[1])
        elif cmd[0] == 'aperture' and len(cmd) == 2:
            self.aperture = float(cmd[1])
        elif cmd[0] == 'denoise' and len(cmd) == 4:
            self.denoise = (int(cmd[1]), int(cmd[2]), int(cmd[3]))
        elif cmd[0] == 'exposure_fuse' and len(cmd) == 2:
            self.exposure_fuse.append(cmd[1])
        elif cmd[0] == 'super_resolution' and len(cmd) == 2:
            self.super_res.append(cmd[1])
        elif cmd[0] == 'super_resolution' and len(cmd) == 3:
            bucket = cmd[1]
            if bucket not in self.super_res_buckets:
                self.super_res_buckets[bucket] = []
            self.super_res_buckets[bucket].append(cmd[2])
        elif cmd[0] == 'super_resolution_config' and len(cmd) == 3:
            self.super_res_config[cmd[1]] = cmd[2]
        elif cmd[0] == 'accel_align' and len(cmd) == 2:
            self.accel_align = (cmd[1] == 'true')
        elif cmd[0] == 'color_correction' and len(cmd) >= 2:
            self.color_correct = cmd[1]
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

def read_image_exif(fname):
    data_map = {}
    exif = ExifToolHelper()
    meta = exif.get_metadata([fname])[0]

    if 'EXIF:GPSLatitude' in meta:
        data_map['last_position_latitude'] = meta['EXIF:GPSLatitude']
    elif 'Composite:GPSLatitude' in meta:
        data_map['last_position_latitude'] = meta['Composite:GPSLatitude']

    if 'EXIF:GPSLongitude' in meta:
        data_map['last_position_longitude'] = meta['EXIF:GPSLongitude']
    elif 'Composite:GPSLongitude' in meta:
        data_map['last_position_longitude'] = meta['Composite:GPSLongitude']

    if 'EXIF:DateTimeOriginal' in meta:
        data_map['date'] = meta['EXIF:DateTimeOriginal']
    elif 'XMP:FirstPhotoDate' in meta:
        data_map['date'] = meta['XMP:FirstPhotoDate']
    return data_map


def read_image_metadata(fname):
    data_map = {}
    with open(fname + ".CSV", 'r') as f:
        for l in f:
            pairs = l.split(',')
            for p in pairs:
                var = p.split('=')
                if var[0] == 'serial_number' or var[0] == 'fw':
                    data_map[var[0]] = var[1]
                elif var[0] == 'last_position':
                    coords = var[1].split('|')
                    data_map[var[0] + '_latitude'] = float(coords[0])
                    data_map[var[0] + '_longitude'] = float(coords[1])
                    data_map[var[0] + '_elevation'] = float(coords[2])
                else:
                    data_map[var[0]] = float(var[1])

    exif = ExifToolHelper()
    meta = exif.get_metadata([fname + '_1.JPG'])[0]
    data_map['date'] = meta['EXIF:DateTimeOriginal']
    return data_map

class ComputeSplice(threading.Thread):
    def __init__(self, splice, resolution, margin):
        threading.Thread.__init__(self)
        self._splice = splice
        self._resolution = resolution
        self._margin = margin

        self.result = None

    def run(self):
        self.result = self._splice.generate_fade(self._resolution, self._margin)


def write_output(meta_map, left, right, config):
    print('writing output files')
    fvr = FormatVR(left, right)
    fvr.set_date(meta_map['date'])
    fvr.set_gps({'latitude': meta_map['last_position_latitude'], \
                 'longitude': meta_map['last_position_longitude']})

    if 'stereo' in config.format:
        fvr.write_stereo(config.output + '_left.JPG', config.output + '_right.JPG')
    if 'over-under' in config.format:
        fvr.write_over_under(config.output + '.JPG')
    if 'gpano' in config.format:
        fvr.write_cardboard(config.output + '_gpano.JPG')

    for f in config.format:
        if not f.startswith('anaglyph'):
            continue

        anaglyph = f.split(':')
        if len(anaglyph) != 6:
            print('anaglyph format requires 5 parameters (degrees) in the format, anaglyph:fov:phi:theta:xres:yres')
            continue

        fvr.write_anaglyph(config.output + '_' + '_'.join(anaglyph) + '.JPG',\
                           float(anaglyph[1]), \
                           float(anaglyph[2]), \
                           float(anaglyph[3]), \
                           float(anaglyph[4]), \
                           float(anaglyph[5]))

    for f in config.format:
        if not f.startswith('over-under:'):
            continue

        over_under = f.split(':')
        if len(over_under) != 6:
            print('over-under format requires 5 parameters (degrees) in the format, over-under:fov:phi:theta:xres:yres')
            continue

        fvr.write_over_under_cropped(config.output + '_' + '_'.join(over_under) + '.JPG',\
                                     float(over_under[1]), \
                                     float(over_under[2]), \
                                     float(over_under[3]), \
                                     float(over_under[4]), \
                                     float(over_under[5]))

    for f in config.format:
        if not f.startswith('gpano:'):
            continue

        cardboard = f.split(':')
        if len(cardboard) != 5:
            print('carboard format requires 4 parameters (degrees) in the format, cardboard:lat:long:vfov:hfov')
            continue

        fvr.write_cardboard_cropped(config.output + '_' + '_'.join(cardboard) + '.JPG', \
                                    float(cardboard[1]), \
                                    float(cardboard[2]), \
                                    float(cardboard[3]), \
                                    float(cardboard[4]))

    for f in config.format:
        if not f.startswith('stereo:'):
            continue

        stereo = f.split(':')
        if len(stereo) != 6:
            print('stereo format requires 5 parameters (degrees) in the format, stereo:fov:phi:theta:xres:yres')
            continue

        fvr.write_stereo_cropped(config.output + '_' + '_'.join(stereo) + '_left.JPG', \
                                 config.output + '_' + '_'.join(stereo) + '_right.JPG', \
                                 float(stereo[1]), \
                                 float(stereo[2]), \
                                 float(stereo[3]), \
                                 float(stereo[4]), \
                                 float(stereo[5]))

def depth_analysis(action, config, coeffs, debug):
    if not os.path.exists(coeffs):
        print('The yaml or ellipse calibration must be performed before depth analysis, please provide a valid alignment file.')
        exit(1)

    with open(coeffs, 'r') as f:
        saved_data = json.load(f)

    if 'calib' not in saved_data:
        print('Calibration must be performed before depth analysis, please provide a valid alignment file.')
        exit(1)
    calibration = [CalibrationParams().from_dict(c) for c in saved_data['calib']]

    loader = ImageLoader(config, debug)
    images = loader.load(calibration)
    calibration = loader.get_calibration()

    with open(config.input + '.json', 'r') as f:
        patches = json.load(f)['patches']

    depths = []
    for ii in range(0, 8, 2):
        depth_cal = DepthCalibration(images[ii], images[ii+1], \
                                     calibration[ii].t, calibration[ii+1].t, debug)
        depth_cal.determine_coordinates(patches)
        depths.append(depth_cal)

    if 'depths' not in saved_data:
        saved_data['depths'] = {}

    saved_data['depths'][config.input] = [d.to_dict() for d in depths]

    # merge after saving the determined coordinates.
    if action == 'regression':
        for f, sd in saved_data['depths'].items():
            if f != config.input:
                for i, sdd in enumerate(sd):
                    depths[i].merge_dict(sdd)

        saved_data['depthCalibration'] = [d.finalize(patches).to_dict() for d in depths]

    with open(coeffs, 'w') as f:
        json.dump(saved_data, f, indent=4)

def main():
    options = ProgramOptions()
    if not options.valid():
        usage()
        print('\nERROR: The image file or config must be specified.')
        exit(1)

    # generate the config, if the file is empty a default config is used.
    # override some of the config options which can be provide by the command line
    config = Config(options.config)
    if options.image_override is not None:
        config.input = options.image_override
        config.output = options.image_override
    if len(options.in_override) > 0:
        config.input = options.in_override
    if len(options.out_override) > 0:
        config.output = options.out_override
    if options.resolution > 0:
        config.resolution = options.resolution
    if options.color_correction is not None:
        config.color_correct = options.color_correction

    if len(options.format) > 0:
        config.format = {}
        for c in options.format:
            config.format[c] = True

    if not config.valid() and not options.load_processed:
        usage()
        print('The config file is invalid, missing input or output.')
        exit(1)

    if options.alignment_file is None and \
       (options.depth is not None or options.yaml_config is not None or options.recalc_ellipse):
        print('--alignment must be provided.')
        exit(1)

    debug = Debug(options)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)

    if options.load_processed:
        img = cv.imread(options.load_processed)
        w = img.shape[1]
        h = int(img.shape[0]/2)
        left = img[:h]
        right = img[h:]
        config.format['over-under'] = False
        meta_map = read_image_exif(options.load_processed)
        config.input = config.output = os.path.splitext(options.load_processed)[0]
        write_output(meta_map, left, right, config)
        exit(1)

    if options.depth is not None:
        depth_analysis(options.depth, config, options.alignment_file, debug)
        print('depth calibration added to:', options.alignment_file)
        plt.show()
        exit(0)

    saved_data = None
    if options.alignment_file is not None and os.path.exists(options.alignment_file):
        print('loading saved coefficients')
        with open(options.alignment_file, 'r') as f:
            saved_data = json.load(f)

    if options.yaml_config is not None:
        fs = cv.FileStorage(options.yaml_config, cv.FILE_STORAGE_READ)
        model = fs.getNode("CamModel_V2_Set")
        yaml_coeffs = [model.getNode('CAM_' + str(i)) for i in range(8)]

        if saved_data is not None and 'calib' in saved_data:
            calibration = [CalibrationParams().from_dict(c).from_yaml(yaml_coeffs[i], i) \
                           for i, c in enumerate(saved_data['calib'])]
        else:
            calibration = [CalibrationParams().from_yaml(yaml_coeffs[i], i) for i in range(8)]

        with open(options.alignment_file, 'w') as f:
            content = saved_data if saved_data is not None else {}
            content['calib'] = [c.to_dict() for c in calibration]
            json.dump(content, f, indent=4)

        print('yaml config added to:', options.alignment_file)
        exit(0)

    loader = ImageLoader(config, debug)
    if saved_data is not None and 'calib' in saved_data:
        calibration = [CalibrationParams().from_dict(c) for c in saved_data['calib']]
    else:
        calibration = [CalibrationParams() for c in range(8)]

    if options.recalc_ellipse:
        for c in calibration:
            c.recalc_ellipse = True
        images = loader.load(calibration)
        calibration = loader.get_calibration()

        with open(options.alignment_file, 'w') as f:
            content = saved_data if saved_data is not None else {}
            content['calib'] = [c.to_dict() for c in calibration]
            json.dump(content, f, indent=4)

        print('ellipse calculations added to:', options.alignment_file)
        exit(0)


    if 'depthCalibration' in saved_data:
        print('depth calibration loaded')
        for i, sd in enumerate(saved_data['depthCalibration']):
            calibration[2*i + 1].depth = DepthCalibration(debug=debug).from_dict(sd)

    # contains gps, date, and camera orientation information.
    meta_map = read_image_metadata(config.input)

    images = loader.load(calibration)
    calibration = loader.get_calibration()

    stitches = []
    ts = []
    matches = None
    cc = None
    seam = refine_seams.RefineSeams(images, debug)

    if options.reuse_seams and saved_data is not None and 'seam' in saved_data:
        print('reusing seam data from alignment')
        seam.from_dict(saved_data['seam'])

    if not options.reuse_seams or (options.fast != 1 and options.fast != 3):
        print('computing seams')
        seam.align()

    if options.write_coeffs:
        with open(options.alignment_file, 'w') as f:
            content = saved_data if saved_data is not None else {}
            content['seam'] = seam.to_dict()
            json.dump(content, f, indent=4)
        print('alignment coeffs added to file:', options.alignment_file)

    stitches = seam._seams
    matches = seam._matches
    ts = seam._transforms

    if config.color_correct == 'mean-seams' and options.fast != 2 and options.fast != 3:
        color = color_correction.ColorCorrectionSeams(images, ts, stitches, debug)
        print('computing color mean-seams')
        cc = color.match_colors()
    elif config.color_correct == 'mean-matches' and options.fast != 2 and options.fast != 3:
        color = color_correction.ColorCorrectionMatches(images, matches, debug)
        print('computing color mean-matches')
        cc = color.match_colors()

    print('')

    splice_left = splice.SpliceImages(images[0:8:2], debug)
    splice_left.set_initial_view(config.view_direction)
    splice_right = splice.SpliceImages(images[1:8:2], debug)
    splice_right.set_initial_view(config.view_direction)

    if config.accel_align:
        rotate_x = math.atan2(meta_map['xAccel'], abs(meta_map['zAccel']))
        rotate_y = math.atan2(meta_map['yAccel'], abs(meta_map['zAccel']))
        splice_left.set_camera_rotations(rotate_x, rotate_y)
        splice_right.set_camera_rotations(rotate_x, rotate_y)

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
        if debug.enable_threads:
            t_left.start()
            t_right.start()
            t_left.join()
            t_right.join()
        else:
            t_left.run()
            t_right.run()

        left = t_left.result
        right = t_right.result
        print('')


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

    write_output(meta_map, left, right, config)

    plt.show()


if __name__ == '__main__':
    main()
