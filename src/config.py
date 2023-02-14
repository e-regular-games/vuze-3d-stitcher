import getopt
import math
import sys

def print_usage():
    print('VuzeMerge 3D 360 Image Generator')
    print('  By S. Ryan Edgar')
    print('  August 1st, 2022')
    print('')
    print('usage: vuze_merge.py -v -h -c config.dat')
    print('')

    print('-a,--alignment <file>\t\tThe alignment coefficients for image processing. This file will be updated by operations such as yaml-config, ellipse-coeffs, write-coeffs, and depth.')
    print('--yaml-config <file>\t\tSpecify the yml config file generated by the camera.')
    print('--ellipse-coeffs\t\tDetermine the coefficients for fisheye/sensor offset and skew from the provided images. Default is to read from the equation coefficients file.')
    print('--depth <method>:<file1>,<file2>,...\t\t\tRead the color/distance information from <file_prefix>.json (same as input image) and add the coordinate data to the provided coefficients file provided by alignment. All other options will be ignored, except for the input image which is required. Methods: {linreg, kabsch}')
    print('--write-coeffs\t\tWrite the alignment equation constants to the provided alingment file.')
    print('--ignore-alignment-seams\t\t\tDo not read previous seam data from the alignment file. Fresh seam data will be written to the file, if write-coeffs is enabled.')
    print('--setup\t\t\tA provided setup file including settings for ellipse-coeffs, yaml-config, and depth calibration as an object with "method" and a list of "files".')
    print('')

    print('-c,--config <file>\t\tSpecify the config file.')
    print('-o,--config-option <option>\t\tComma separate config option line.')
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
        self.config_options = []

        self.alignment_file = None
        self.yaml_config = None
        self.depth_method = None
        self.depth_files = None
        self.recalc_ellipse = False
        self.write_coeffs = False
        self.reuse_seams = True
        self.setup = None

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
            'd:hF:w:r:vc:i:f:I:O:q:C:a:l:o:',
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
                'config-option=',
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
                'load-processed=',
                'setup='
            ])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            if o in ("-o", "--config-option"):
                self.config_options.append(a)
            elif o == "--yaml-config":
                self.yaml_config = a
            elif o == "--depth":
                self.depth_method = a.split(':')[0]
                self.depth_files = a.split(':')[1].split(',')
            elif o == '--setup':
                self.setup = a
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
                print_usage()
                exit(0)

    def valid(self):
        v = True
        if len(self.config) == 0 and self.image_override is None \
           and (len(self.in_override) == 0 or len(self.out_override) == 0) \
           and self.load_processed is None and self.setup is None:
            print('Please provide image file names or --load-processed or --setup.')
            v = False

        if self.alignment_file is None:
            print('An alignment file must be provided using --alignment.')
            v = False
        return v

class Config:
    def __init__(self, file_path='', addl_options=[]):
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

        for l in addl_options:
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