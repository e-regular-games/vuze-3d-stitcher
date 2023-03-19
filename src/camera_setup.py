import sys
import os
import json
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from image_loader import ImageLoader
from image_loader import CalibrationParams
from depth_mesh import DepthCalibration
from config import Config

def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]


class CameraSetup():
    def __init__(self, alignment_file, setup_file, debug):
        self._alignment_file = alignment_file
        self._setup_file = setup_file
        self._setup = None
        self._alignment = None
        self._calibration = None
        self._debug = debug

    def valid(self):
        v = True
        if not os.path.exists(self._setup_file):
            print('The setup file is invalid: ' + self._setup_file)
            v = False
        return v

    def run(self):
        with open(self._setup_file, 'r') as f:
            self._setup = json.load(f)

        if os.path.exists(self._alignment_file):
            with open(self._alignment_file, 'r') as f:
                self._alignment = json.load(f)
        else:
            self._alignment = {}

        if 'calib' in self._alignment:
            self._calibration = \
                [CalibrationParams(self._debug).from_dict(c) for c in self._alignment['calib']]
        else:
            self._calibration = [CalibrationParams(self._debug) for c in range(8)]

        self._yaml_config()
        self._ellipse()

        for i, c in enumerate(self._calibration):
            self._debug.log('lens(' + str(i) + ') calibration', c.aperture)
            self._debug.log(np.transpose(c.t))
            self._debug.log(c.ellipse)
            self._debug.log(c.camera_matrix)
            self._debug.log(c.R)

        self._depth()

        self._alignment['calib'] = [c.to_dict() for c in self._calibration]

        with open(self._alignment_file, 'w') as f:
            json.dump(self._alignment, f, indent=4)


    def _ellipse(self):
        if not 'ellipse' in self._setup:
            return
        if not 'image' in self._setup['ellipse']:
            print('An "image" base name is required for ellipse calibration.')
            return

        config = Config()
        config.input = self._setup['ellipse']['image']
        loader = ImageLoader(config, self._debug)

        for c in self._calibration:
            c.recalc_ellipse = True
        loader.load(self._calibration)
        self._calibration = loader.get_calibration()

        print('ellipse calculations complete.')

    def _yaml_config(self):
        if not 'yamlConfig' in self._setup:
            return

        fs = cv.FileStorage(self._setup['yamlConfig'], cv.FILE_STORAGE_READ)
        model = fs.getNode("CamModel_V2_Set")
        yaml_coeffs = [model.getNode('CAM_' + str(i)) for i in range(8)]
        for i, c in enumerate(self._calibration):
            c.from_yaml(yaml_coeffs[i], i)
        print('yaml config read.')

    def _seams(self):
        if not 'depth' in self._setup:
            return

        if not 'patches' in self._setup['depth']:
            print('A "patches" list must be provided with color and distance information.')
            return
        if not 'method' in self._setup['depth']:
            print('The "method" must be provided: {linreg, kabsch}.')
            return
        if not 'seams' in self._setup['depth']:
            print('The "seams" must be provided as an array of objects with "name" and "seam".')
            return

        for c in self._calibration:
            if c.empty():
                print('The yaml or ellipse calibration must be performed before depth analysis.')
                return

        patches = self._setup['depth']['patches']
        mode = self._setup['depth']['method']

        seams = [SeamCalibration(self._debug) \
                 .set_mode(mode) \
                 .set_patches(patches) \
                 for i in range(2)]

        locations = [c.t for c in (self._calibration + self._calibration[-2:])]
        for s in self._setup['depth']['seams']:
            if not 'name' in s or not 'seam' in s:
                print('Each object in seams must contain "seam" and "name".')
                return
            config = Config()
            config.input = s['name']
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            images = images + images[-2:]
            i = s['seam']
            seam = seams[int(i/2)] \
                .set_side(i%2) \
                .set_images(images[i*2:i*2+4]) \
                .set_locations(locations[i*2:i*2+4]) \
                .determine_coordinates()


        self._calibration[2*i+1].depth = d

    def _depth(self):
        if not 'depth' in self._setup:
            return

        if not 'patches' in self._setup['depth']:
            print('A "patches" list must be provided with color and distance information.')
            return
        if not 'images' in self._setup['depth']:
            print('A list of "files" with image base names must be provided.')
            return
        if not 'method' in self._setup['depth']:
            print('The "method" must be provided: {linreg, kabsch}.')
            return
        if not 'seams' in self._setup['depth']:
            print('The "seams" must be provided as an array of objects with "name" and "seam".')
            return

        for c in self._calibration:
            if c.empty():
                print('The yaml or ellipse calibration must be performed before depth analysis.')
                return

        patches = self._setup['depth']['patches']
        mode = self._setup['depth']['method']

        depths = [DepthCalibration(self._debug.set_subplot(2, 4, i+1)) \
                  .set_mode(mode) \
                  .set_patches(patches) \
                  for i in range(8)]

        class ImageSet:
            def __init__(self, images, name, seam_only=None):
                self.images = images
                self.filename = name
                self.seam_only = seam_only

        locations = [c.t for c in self._calibration]
        image_sets = []
        for img in self._setup['depth']['images']:
            config = Config()
            config.input = img
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            image_sets.append(ImageSet(images, img))

        for s in image_sets:
            for i, d in enumerate(depths):
                o = 1 - 2*(i%2)
                d.add_coordinates(s.images[i], s.images[i+o], locations[i], locations[i+o])

        seam_sets = []
        for s in self._setup['depth']['seams']:
            if not 'name' in s or not 'seam' in s:
                print('Each object in seams must contain "seam" and "name".')
                return
            config = Config()
            config.input = s['name']
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            seam_sets.append(ImageSet(images, s['name'], s['seam']))

        def rotate_seam_image(i, direction):
            tmp = np.zeros(i.shape, np.uint8)
            w = i.shape[1]
            if direction == 1:
                tmp[:,int(w/2):] = i[:,int(w/4):int(3*w/4)]
            elif direction == -1:
                tmp[:,:int(w/2)] = i[:,int(w/4):int(3*w/4)]
            return tmp

        def rotate(p, direction):
            if direction == 1:
                return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float32) @ p
            elif direction == -1:
                return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32) @ p

        for s in seam_sets:
            a = 2*s.seam_only
            b = 2*s.seam_only + 1
            c = (2*s.seam_only + 2) % 8
            d = (2*s.seam_only + 3) % 8
            #combos = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            combos = [(a, c), (b, d)]
            for m in combos:
                i0 = s.images[m[0]]
                i1 = s.images[m[1]]
                depths[m[0]].add_coordinates(i0, rotate_seam_image(i1, 1), \
                                             locations[m[0]], rotate(locations[m[1]], 1))

                depths[m[1]].add_coordinates(i1, rotate_seam_image(i0, -1), \
                                             locations[m[1]], rotate(locations[m[0]], -1))

        for i, d in enumerate(depths):
            d.finalize()
            self._calibration[i].depth = d

        fit_info = np.zeros((len(image_sets), 8, 4))
        for si, s in enumerate(image_sets):
            print(s.filename)

            if self._debug.enable('depth-cal-finalize'):
                self._debug._window_title = s.filename
                self._debug.figure('depth-cal-adjusted', True)
                self._debug.figure('depth-cal-original', True)

            for i, d in enumerate(depths):
                o = 1 - 2*(i%2)
                i1 = create_from_middle(depths[i+o].apply(get_middle(s.images[i+o])))
                fit_info[si, i] = d.result_info(s.images[i], s.images[i+o], i1, \
                                                locations[i], locations[i+o])

        print(fit_info)
