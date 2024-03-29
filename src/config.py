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

    print('-a,--alignment <file>\t\tThe alignment coefficients for image processing. This file will be updated by operations such as setup and write-coeffs.')
    print('--write-coeffs\t\tWrite the alignment equation constants to the provided alingment file.')
    print('--ignore-alignment-seams\t\t\tDo not read previous seam data from the alignment file. Fresh seam data will be written to the file, if write-coeffs is enabled.')
    print('--setup\t\t\tA provided setup file including settings for ellipse-coeffs, yaml-config, and depth calibration as an object with "method" and a list of "files".')
    print('--depth\t\t\tDisplay the depth map for each lens. No ouput images are rendered.')
    print('')

    print('-c,--config <file>\t\tSpecify the config file.')
    print('-o,--config-option <option>\t\tComma separated config option line.')
    print('-i,--image <file_prefix>\t\tOverride the input and output config options.')
    print('-I,--in <file_prefix>\t\tOverride the input config option.')
    print('-O,--out <file_prefix>\t\tOverride the output config option.')
    print('-q,--quality <verticle_pixels>\t\tVertical resolution of the full output image.')
    print('-F,--fast <level>\t\t\tSkip recomputing seams and color correction. Levels {1: skip seams, 2: skip color, 3: skip seams and color}.')
    print('-f,--format <spec>\t\tA "," separated list of output formats: {gpano, stereo, over-under}. See the config file description below for more information.')
    print('')

    print('-l,--load-processed <file>\t\tA previously generated over-under 3D 360 image, with extension "image.JPG". Only the format output option is enabled. (Will not overwrite.)')
    print('')

    print('-v,--verbose\t\tMore detailed output to the console.')
    print('-d,--display <enums>\t\tShow intermediate progress images. Enums separated by ",". Use "--help-display" to see a complete list.')
    print('-h,--help\t\tDisplay this message.')
    print('--help-display\t\tList the possible enums for the "display" arguement')
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
    print('denoise,<h>,<template-size>,<search-size>\tDenoise the images using fastNlMeansDenoisingColor algorithm. Recommend: 3,7,21')
    print('world,<r-x>,<r-y>,<r-z>,<height>\tDefine the radius of the world and set the height from the ground. Inside: 5,5,3,1.8 or Default: 40,40,20,1.8')
    print('exposure_match,<1-8>\t\t\tImage to use as reference for exposure histogram matching.')
    print('exposure_fuse,<image_prefix>\t\tFile name to include in the exposure fusion stack.')
    print('accel_align,<enabled>\t\t\tUse the accelerometer to rotate and translate the lens. default: false')
    print('contrast_equ,<clip>,<gridx>,<gridy>\tEnable adaptive hsv-value histogram equalization.')
    print('seam,blend,<margin>\t\t\tBlend by taking a linear weighted average across the margin about the seam. (degrees)')
    print('seam,color,<margin>\t\t\tWidth of the area around the seam that is used for color matching. Wider margins require more seam points located away from the image edge. (degrees)')
    print('seam,pyramid,<depth>\t\t\tBlend using Laplacian Pyramids to the specified depth. Experimental: causes color and image distortion.')
    print('super_resolution,<image>\t\tSpecified at least 2 times. Each image will be merged into the result.')
    print('super_resolution,<bucket>,<image>\t\tSpecified at least 4 times, 2 separate buckets. Buckets can be arbitrary, images within a bucket will be merged, then buckets will be exposure merged using Mertens.')
    print('super_resolution_config,<variable>,<value>\t\tSupport variables {outlier_limit,sharpen}. Outlier limit is a number of standard deviations, and sharpen is "true" or "false". By default sharpen is "false" and outlier limit is not applied.')
    print('lens,<1-8>,<x_pixels>,<y_pixels>\tThe center of the fisheye for each lens.')
    print('\n')

def print_usage_display():
    print('VuzeMerge 3D 360 Image Generator')
    print('  By S. Ryan Edgar')
    print('  August 1st, 2022')
    print('')
    print('Display Options')
    print('')
    print('fisheye', '\t\t\t', 'Graph the equirectangular conversion of the original fisheye images.')
    print('fisheye-fio', '\t\t\t', 'Write to files "equirect_#" the equirectangular conversion of the original fisheye images.')
    print('calib', '\t\t\t\t', 'Graph the elliptical calibration result.')
    print('calib-fio', '\t\t\t', 'Write to files "lens_alignment_in_lens#", the elliptical calibration result.')
    print('exposure-match', '\t\t\t', 'Graph the result of exposure matching between lenses.')
    print('exposure-fuse', '\t\t\t', 'Graph the result of fusing exposure stacks across multiple images within the same lens.')
    print('denoise', '\t\t\t', 'Graph the result of denoising the images.')
    print('depth-samples', '\t\t\t', 'Plot in 3d the samples used during depth calibration and where they exist in the polar coordinate system relative to the lens.')
    print('depth-error', '\t\t\t', 'Plot in 3d the error between the measured distance at each polar coordinate sample and the expected distance.')
    print('depth-cal-finalize', '\t\t', 'Graph each lens after applying corrections to ensure the depth of each sample is as close to the expected value as possible.')
    print('depth-map', '\t\t\t', 'Graph the distance in equirectangular space to each pixel.')
    print('color-regression', '\t\t', 'Histogram the error for each color when calculating the linear regression for color corrections.')
    print('color-delta', '\t\t\t', 'Graph the difference in each color channel within an image due to color correction.')
    print('color-regression-kmeans', '\t', 'Graph the histogram of separating colors into kmeans.')
    print('color-correction', '\t\t', 'Histogram of the number of pixels falling into each of the kmeans used to correct colors.')
    print('color', '\t\t\t\t', 'An image of the original colors, the desired color, and the corrected color per pixel for the image.')
    print('features-keypoints', '\t\t', 'Show an image of all the detected keypoints (marked in red) within the image.')
    print('features-matches', '\t\t', 'Show an image of all the matching keypoints (marked in red) between two or more images.')
    print('seam-path', '\t\t\t', 'Plot in 3d the points along the seam paths between the images. Each plot will have points for each of the 4 images along the seam.')
    print('align-depth', '\t\t\t', 'Graph the distance in equirectangular space to each pixel along the seam for 2 out of 4 of the images.')
    print('align-regression', '\t\t', 'Graphs of the linear regression results of merging the seams in the world coordinate space.')
    print('seam-path-cost', '\t\t\t', 'Graph the cost matrix and its component costs used to find the lowest cost path between the top and bottom of the image, ie the seam.')

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

        self.depth  = False
        self.display = {}
        self.image_override = None
        self.in_override = ''
        self.out_override = ''
        self.format = []
        self.resolution = 0
        self.fast = 0
        self.load_processed = None

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'd:hF:w:r:vc:i:f:I:O:q:a:l:o:',
            [
                'help',
                'help-display',
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
                'fast=',
                'alignment=',
                'ignore-alignment-seams',
                'write-coeffs',
                'depth',
                'load-processed=',
                'setup='
            ])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            if o in ("-o", "--config-option"):
                self.config_options.append(a)
            elif o == '--setup':
                self.setup = a
            elif o in ("-a", "--alignment"):
                self.alignment_file = a
            elif o == "--ignore-alignment-seams":
                self.reuse_seams = False
            elif o == "--write-coeffs":
                self.write_coeffs = True
            elif o == "--depth":
                self.depth = True
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
            elif o == "--help-display":
                print_usage_display();
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
        self.accel_align = True
        self.contrast_equ = None
        self.seam_blend_margin = 1.0 * math.pi / 180
        self.seam_color_margin = 6.0 * math.pi / 180
        self.seam_pyramid_depth = 0
        self.denoise = ()
        self.world_radius = (40, 40, 20, 1.8)

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
        elif cmd[0] == 'world' and len(cmd) == 5:
            self.world_radius = (float(cmd[1]), float(cmd[2]), float(cmd[3]), float(cmd[4]))
        elif cmd[0] == 'contrast_equ' and len(cmd) == 4:
            self.contrast_equ = (float(cmd[1]), int(cmd[2]), int(cmd[3]))
        elif cmd[0] == 'seam' and len(cmd) == 3:
            if cmd[1] == 'blend':
                self.seam_blend_margin = float(cmd[2]) * math.pi / 180
                self.seam_pyramid_depth = 0
            elif cmd[1] == 'color':
                self.seam_color_margin = float(cmd[2]) * math.pi / 180
            elif cmd[1] == 'pyramid':
                self.seam_pyramid_depth = int(cmd[2])
                self.seam_blend_margin = 0
        elif cmd[0] == 'format' and len(cmd) == 2:
                self.format[cmd[1]] = True

    def valid(self):
        return self.input != '' and self.output != ''
