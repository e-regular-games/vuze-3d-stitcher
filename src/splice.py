#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular

class SpliceImages():
    def __init__(self, images, debug):
        self._images = images

        # tansform(i) of image (i)
        self._transforms = [None]*len(images)
        self._color_transforms = [None]*len(images)

        # seam(i) between images (i) and (i+1)
        self._stitches = [None]*len(images)
        self._st_slopes = [None]*len(images)
        self._st_offset = [None]*len(images)

        self._debug = debug

    def set_transform(self, idx, t):
        self._transforms[idx] = t

    def set_color_transform(self, idx, cc):
        self._color_transforms[idx] = cc

    # line is a numpy array of (phi, theta) rows
    # the first row should be phi = 0, and the last should be phi = pi
    def set_stitch(self, idx, line):
        self._stitches[idx] = line

        phi = line[:, 0]
        theta = line[:, 1]

        slope = (theta[1:] - theta[:-1]) / (phi[1:] - phi[:-1])
        offset = theta[:-1] - slope * phi[:-1]

        self._st_slopes[idx] = slope
        self._st_offset[idx] = offset

    def _compute_left_side(self, polar, idx):
        slope = self._st_slopes[idx]
        offset = self._st_offset[idx]

        seam_theta = np.zeros(polar.shape[0])

        COMP_SIZE = 1000000
        for i in range(0, polar.shape[0], COMP_SIZE):
            p = polar[i:i+COMP_SIZE,:]
            f_mat = np.ones((p.shape[0], slope.shape[0] + 1)) * self._stitches[idx][:, 0]
            in_range = np.logical_and(p[:,0:1] <= f_mat[:,1:], p[:,0:1] >= f_mat[:,:-1])

            f_slope = (np.ones((p.shape[0], slope.shape[0])) * slope)[in_range]
            f_offset = (np.ones((p.shape[0], offset.shape[0])) * offset)[in_range]

            seam_theta[i:i+COMP_SIZE] = p[:,0] * f_slope + f_offset

        return polar[:, 1] < seam_theta


    def _filter_to_stitch(self, polar, l):
        f = None
        if l == 0:
            right = self._compute_left_side(polar, l)
            left = np.logical_not(self._compute_left_side(polar, len(self._stitches) - 1))
            f = np.logical_or(left, right)
        else:
            left = self._compute_left_side(polar, l)
            right = np.logical_not(self._compute_left_side(polar, l - 1))
            f = np.logical_and(left, right)

        indices = np.arange(0, polar.shape[0]).reshape(polar.shape[0], 1)
        return np.concatenate([polar.copy()[f], indices[f]], axis=-1)

    def _interp(self, eqr, img):
        l = eqr.shape[0]
        s = math.floor(math.sqrt(l) + 1)
        padding = np.zeros(s*s - l, dtype=np.float32)

        x = np.concatenate([eqr[:, 0], padding]).reshape(s, s).astype(np.float32)
        y = np.concatenate([eqr[:, 1], padding]).reshape(s, s).astype(np.float32)

        pixels = cv.remap(img, x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        return pixels.reshape(s * s, 3)[:l]

    def generate(self, v_res):
        shape = (v_res, 2 * v_res, 3)
        px_count = shape[1] * shape[0]
        eq = np.zeros((px_count, 2), dtype=np.float)

        x_range = range(shape[1])
        y_range = range(shape[0])
        for y in y_range:
            for x in x_range:
                i = y * len(x_range) + x
                eq[i] = [x, y]

        polar = coordinates.eqr_to_polar(eq, shape)
        result = np.zeros((px_count, 3), dtype=np.uint8)
        for s in range(len(self._stitches)):
            print('filter to segment ' + str(s))
            pts = self._filter_to_stitch(polar, s)
            pts[:,1] -= s * math.pi / 2
            pts[:,1] += math.pi
            pts[:,1] = pts[:,1] % (2 * math.pi)
            local_pts_polar = self._transforms[s].reverse(pts)
            local_pts_eqr = coordinates.polar_to_eqr(local_pts_polar, self._images[s].shape)
            print('determine pixel colors ' + str(s))
            pixels = coordinates.eqr_interp(local_pts_eqr, self._images[s])
            if self._color_transforms[s] is not None:
                pixels = self._color_transforms[s] \
                             .correct_bgr(pixels, pts[:,0:2] - [0, math.pi])
            result[local_pts_eqr[:,2].astype(np.int)] = pixels

        return result.reshape(shape)
