import sys
import os
import json
import math
import numpy as np
import cv2 as cv
import coordinates
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

        if 'center' in self._setup['depth']:
            center = np.array(self._setup['depth']['center'], np.float32)
        else:
            center = np.array([0, 0, 0], np.float32)
        center = center.reshape((3, 1))

        for c in self._calibration:
            if c.empty():
                print('The yaml or ellipse calibration must be performed before depth analysis.')
                return

        patches = self._setup['depth']['patches']
        mode = self._setup['depth']['method']

        depths = [DepthCalibration(self._debug.set_subplot(2, 4, i+1)) \
                  .set_mode(mode) \
                  .set_patches(patches, center) \
                  for i in range(8)]

        # clear out any existing depth calibration
        for c in self._calibration:
            c.depth = None

        class ImageSet:
            def __init__(self, images, name, rotations=None, seam_only=None):
                self.images = [get_middle(img) for img in images]
                self.filename = name
                self.rotation = self.R(rotations) if rotations is not None else self.R([])
                self.seam_only = seam_only

            def R(self, rotations):
                def x(rad):
                    return np.array([[1, 0, 0], [0, math.cos(rad), -math.sin(rad)], [0, math.sin(rad), math.cos(rad)]], np.float32)
                def y(rad):
                    return np.array([[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-math.sin(rad), 0, math.cos(rad)]], np.float32)
                def z(rad):
                    return np.array([[math.cos(rad), -math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]], np.float32)

                R = np.identity(3, np.float32)
                for r in rotations:
                    rad = r["degree"] * math.pi / 180
                    if r["axis"] == "x":
                        R = R @ x(rad)
                    elif r["axis"] == "y":
                        R = R @ y(rad)
                    elif r["axis"] == "z":
                        R = R @ z(rad)
                return R


        locations = [coordinates.switch_axis(c.t) for c in self._calibration]
        image_sets = []
        for i in self._setup['depth']['images']:
            config = Config()
            config.input = i["name"]
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            rotations = i["rotations"] if "rotations" in i else None
            image_sets.append(ImageSet(images, i["name"], rotations))

        seam_sets = []
        for s in self._setup['depth']['seams']:
            if not 'name' in s or not 'seam' in s:
                print('Each object in seams must contain "seam" and "name".')
                return
            config = Config()
            config.input = s['name']
            loader = ImageLoader(config, self._debug)
            images = loader.load(self._calibration)
            rotations = s["rotations"] if "rotations" in s else None
            seam_sets.append(ImageSet(images, s['name'], rotations, s['seam']))

        all_sets = image_sets + seam_sets
        fit_info = np.zeros((len(all_sets), 8, 4))

        for si, s in enumerate(image_sets):
            R = s.rotation
            for i in range(4):
                imgs = [s.images[2*i], s.images[2*i+1]]
                l = [R @ locations[2*i], R @ locations[2*i+1]]
                offset = np.zeros((2,), np.float32)
                fit_info[si,2*i,0:2] = depths[2*i].add_coordinates(0, imgs, l, offset)
                fit_info[si,2*i+1,0:2] = depths[2*i+1].add_coordinates(1, imgs, l, offset)

        def rotate(p, direction):
            if direction == 1:
                return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32) @ p
            elif direction == -1:
                return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], np.float32) @ p
            return p

        for si, s in enumerate(seam_sets):
            idx = [(2*s.seam_only + i) % 8 for i in range(4)]
            R = s.rotation
            l = [R @ rotate(locations[ii], -int(i/2)) for i, ii in enumerate(idx)]
            imgs = [s.images[ii] for ii in idx]
            offset = np.array([0, 0, math.pi/2, math.pi/2], np.float32)

            for i, ii in enumerate(idx):
                fit_info[len(image_sets) + si,ii,0:2] = \
                    depths[ii].add_coordinates(i, imgs, l, offset)

        for i, d in enumerate(depths):
            d.finalize()
            self._calibration[i].depth = d

        # evaluate how well the calibration performed by adjusting all the images
        # and then recalculating the error between the expected and actual depth
        for si, s in enumerate(all_sets):
            print(s.filename)
            adjusted = [depths[i].apply(img) for i, img in enumerate(s.images)]
            R = s.rotation

            if self._debug.enable('depth-cal-finalize'):
                f0 = plt.figure()
                f0.canvas.manager.set_window_title('depth-cal-original -- ' + s.filename)
                f1 = plt.figure()
                f1.canvas.manager.set_window_title('depth-cal-adjusted -- ' + s.filename)
                for i in range(8):
                    f0.add_subplot(2, 4, i+1) \
                      .imshow(cv.cvtColor(adjusted[i], cv.COLOR_BGR2RGB))
                    f1.add_subplot(2, 4, i+1) \
                      .imshow(cv.cvtColor(s.images[i], cv.COLOR_BGR2RGB))

            if s.seam_only is None:
                for i in range(4):
                    imgs = [adjusted[2*i], adjusted[2*i+1]]
                    l = [R @ locations[2*i], R @ locations[2*i+1]]
                    offset = np.zeros((2,), np.float32)
                    fit_info[si,2*i,2:4] = depths[2*i].add_coordinates(0, imgs, l, offset)
                    fit_info[si,2*i+1,2:4] = depths[2*i+1].add_coordinates(1, imgs, l, offset)
            else:
                idx = [(2*s.seam_only + i) % 8 for i in range(4)]
                l = [R @ rotate(locations[ii], -int(i/2)) for i, ii in enumerate(idx)]
                imgs = [adjusted[ii] for ii in idx]
                offset = np.array([0, 0, math.pi/2, math.pi/2], np.float32)

                for i, ii in enumerate(idx):
                    fit_info[si,ii,2:4] = \
                        depths[ii].add_coordinates(i, imgs, l, offset)

        print(fit_info)
        plt.show()
