import sys
import os
import json
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from image_loader import ImageLoader
from image_loader import CalibrationParams
from depth_mesh import DepthMesh
from depth_mesh import DepthCalibration
from config import Config

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

        self._alignment = {}
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

        for c in self._calibration:
            if c.empty():
                print('The yaml or ellipse calibration must be performed before depth analysis.')
                return

        patches = self._setup['depth']['patches']
        mode = self._setup['depth']['method']

        depths = [DepthCalibration(self._debug.set_subplot(1, 4, i+1)) \
                  .set_mode(mode) \
                  .set_patches(patches) \
                  for i in range(4)]

        class ImageSet:
            def __init__(self, images, name):
                self.images = images
                self.filename = name

        image_sets = []
        for img in self._setup['depth']['images']:
            config = Config()
            config.input = img
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            image_sets.append(ImageSet(images, img))

        for s in image_sets:
            for i, d in enumerate(depths):
                ii = 2*i
                d.set_image_pair(s.images[ii], s.images[ii+1], \
                                 self._calibration[ii].t, self._calibration[ii+1].t)
                d.determine_coordinates()

        for i, d in enumerate(depths):
            d.finalize()
            self._calibration[2*i+1].depth = d

        fit_info = np.zeros((len(image_sets), 4, 4))
        for si, s in enumerate(image_sets):
            print(s.filename)

            if self._debug.enable('depth-cal-finalize'):
                self._debug._window_title = s.filename
                self._debug.figure('depth-cal-left', True)
                self._debug.figure('depth-cal-original', True)
                self._debug.figure('depth-cal-right', True)

            for i, d in enumerate(depths):
                ii = 2*i
                d.set_image_pair(s.images[ii], s.images[ii+1], \
                                 self._calibration[ii].t, self._calibration[ii+1].t)
                fit_info[si, i] = d.result_info()

        print(fit_info)
