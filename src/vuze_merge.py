#!/usr/bin/python

import color_correction
import sys
import os
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
from depth_mesh import DepthCalibration
from debug_utils import Debug
from config import ProgramOptions
from config import Config
from camera_setup import CameraSetup

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


def main():
    options = ProgramOptions()
    if not options.valid():
        usage()
        print('\nERROR: The image file or config must be specified.')
        exit(1)

    # generate the config, if the file is empty a default config is used.
    # override some of the config options which can be provide by the command line
    config = Config(options.config, options.config_options)
    if options.image_override is not None:
        config.input = options.image_override
        config.output = options.image_override
    if len(options.in_override) > 0:
        config.input = options.in_override
    if len(options.out_override) > 0:
        config.output = options.out_override
    if options.resolution > 0:
        config.resolution = options.resolution

    if len(options.format) > 0:
        config.format = {}
        for c in options.format:
            config.format[c] = True

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
        exit(0)

    if options.setup is not None:
        camera_setup = CameraSetup(options.alignment_file, options.setup, debug)
        if not camera_setup.valid():
            exit(1)
        camera_setup.run()
        plt.show()
        exit(0)

    saved_data = None
    if options.alignment_file is not None and os.path.exists(options.alignment_file):
        print('loading saved coefficients')
        with open(options.alignment_file, 'r') as f:
            saved_data = json.load(f)
    else:
        print('An alignment file must be provided using --alignment.')
        exit(1)

    loader = ImageLoader(config, debug)
    if saved_data is None or not 'calib' in saved_data:
        print('Use --setup to determine the calibration coefficients before processing images.')
        exit(1)
    calibration = [CalibrationParams(debug).from_dict(c) for c in saved_data['calib']]

    for i, c in enumerate(calibration):
        debug.log('lens(' + str(i) + ') calibration', c.aperture)
        debug.log(np.transpose(c.t))
        debug.log(c.ellipse)
        debug.log(c.camera_matrix)
        debug.log(c.R)

    # contains gps, date, and camera orientation information.
    meta_map = read_image_metadata(config.input)

    images = loader.load(calibration)
    calibration = loader.get_calibration()

    stitches = []
    ts = []
    matches = None
    cc = None
    seam = refine_seams.RefineSeams(images, debug)
    seam.calibration = calibration
    seam.border = 2 * config.seam_blend_margin

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

    seams, transforms = seam.data()

    # remove the alpha channel from images.

    for i, img in enumerate(images):
        images[i] = img[...,0:3]

    if options.fast != 2 and options.fast != 3:
        color = color_correction.ColorCorrectionSeams(images, transforms, seams, debug)
        color.border = 1.75 * config.seam_blend_margin
        print('computing color mean-seams')
        cc = color.match_colors()

    print('')

    splice_left = splice.SpliceImages(images[0:8:2], debug)
    splice_left.set_initial_view(config.view_direction)
    splice_left.set_calibration(calibration[0:8:2])
    splice_right = splice.SpliceImages(images[1:8:2], debug)
    splice_right.set_initial_view(config.view_direction)
    splice_right.set_calibration(calibration[1:8:2])

    if config.accel_align:
        rotate_x = math.atan2(meta_map['xAccel'], abs(meta_map['zAccel']))
        rotate_y = math.atan2(meta_map['yAccel'], abs(meta_map['zAccel']))
        splice_left.set_camera_rotations(rotate_x, rotate_y)
        splice_right.set_camera_rotations(rotate_x, rotate_y)

    for s in range(4):
        st_l = seams[2*s].copy()
        st_l[:,1] += s * math.pi / 2
        splice_left.set_stitch(s, st_l)
        splice_left.set_transform(s, transforms[2*s])

        st_r = seams[2*s+1].copy()
        st_r[:,1] += s * math.pi / 2
        splice_right.set_stitch(s, st_r)
        splice_right.set_transform(s, transforms[2*s+1])

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
