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

    # polar is assumed to be an N x M x 2 matrix with each row
    # having the same value of phi.
    def _compute_left_side(self, polar, idx, margin):
        phi = polar[:,0,0:1]
        phi_n = phi.shape[0]
        seam_theta = coordinates.seam_intersect(self._stitches[idx], phi)

        delta = seam_theta.reshape((phi_n,1)) - polar[:,:,1]
        if margin == 0:
            delta[delta >= 0] = 1
            delta[delta < 0] = 0
            return delta

        delta = delta / np.absolute(margin / 2)
        if margin > 0:
            border = np.logical_and(delta < 1, delta >= -1)
            delta[delta >= 1] = 1
            delta[border] = (delta[border] + 1) / 2
            delta[delta < -1] = 0
        return delta

    def _filter_to_stitch(self, polar, l, margin):
        f = None
        if l == 0:
            right = self._compute_left_side(polar, l, margin)
            left = 1.0 - self._compute_left_side(polar, len(self._stitches) - 1, margin)
            f = left + right
        else:
            left = self._compute_left_side(polar, l, margin)
            right = 1.0 - self._compute_left_side(polar, l - 1, margin)
            f = left * right

        flt = f > 0
        return f, flt

    def _generate(self, v_res, images, margin):
        shape = (v_res, 2 * v_res, 3)
        mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        eq = np.concatenate([x.reshape((x.shape[0], x.shape[1], 1)) for x in mesh], axis=2)

        polar = coordinates.eqr_to_polar_3d(eq)
        result = np.zeros(shape, dtype=np.uint8)
        for s in range(len(self._stitches)):
            print('compute segment ' + str(s))
            global_pts_polar = polar.copy()
            local_pts_polar = np.zeros(polar.shape)
            local_pts_eqr = np.zeros(polar.shape)
            pixels = np.zeros(polar.shape[:-1] + (3,))
            weight, flt = self._filter_to_stitch(polar, s, margin)

            global_pts_polar[flt,1] -= s * math.pi / 2
            global_pts_polar[flt,1] += math.pi
            global_pts_polar[flt,1] = global_pts_polar[flt,1] % (2 * math.pi)
            local_pts_polar[flt] = self._transforms[s].reverse(global_pts_polar[flt])
            local_pts_eqr[flt] = coordinates.polar_to_eqr(local_pts_polar[flt], shape)
            pixels[flt] = coordinates.eqr_interp(local_pts_eqr[flt], images[s])
            if self._color_transforms[s] is not None:
                pixels = self._color_transforms[s] \
                             .correct_bgr(pixels, \
                                          global_pts_polar[...,0:2] - [0, math.pi], \
                                          local_pts_polar[...,0:2],
                                          flt)


            n = np.count_nonzero(flt)
            result[flt] += (pixels[flt] * weight[flt].reshape((n, 1))).astype(np.uint8)

        return result

    def generate(self, v_res):
        return self._generate(v_res, self._images, 0)

    def generate_fade(self, v_res, dist):
        return self._generate(v_res, self._images, dist)

    # This looks to merge the image correctly, but introduce a discrepancy in color
    # and it reduces the overall image quality significantly.
    # Hypothesis, the laplacian pyramids lose quality when they are converted from
    # equirect -> polar -> equirect with interpolation at the final step.
    # the lower resolution images may not be able to survive the conversions.
    def generate_pyramid(self, v_res, layers):
        vr = v_res
        for i in range(layers-1):
            if vr % 2 != 0:
                print('Verticle resolution must be divisible by 2 a total number of ' + str(layers) + ' times.')
            vr = vr / 2

        # generate Gaussian pyramid for each images
        # a list of lists of np.array(n, 2n, 3)
        gp = []
        for img in self._images:
            G = img.copy()
            gpi = [G]
            for i in range(layers):
                G = cv.pyrDown(G)
                gpi.append(G)
            gp.append(gpi)

        # generate Laplacian Pyramid
        lp = []
        for gpi in gp:
            lpi = [gpi[layers-1]]
            for i in range(layers-1, 0, -1):
                GE = cv.pyrUp(gpi[i])
                L = cv.subtract(gpi[i-1],GE)
                lpi.append(L)
            lp.append(lpi)

        # re-arrange the dimensions, initially: lp[img][layer]
        # final: lp[layer][img]
        # this allows the lp[layer] array to be passed to generate.
        lpl = [([None]*len(self._images)) for i in range(layers)]
        for i, lpi in enumerate(lp):
            for l, img in enumerate(lp[i]):
                lpl[l][i] = img

        # stitched images for the single eye
        ls = []
        cts = self._color_transforms
        for i, lpi in enumerate(lpl):
            ls.append(self._generate(int(v_res / pow(2, layers - i - 1)), lpi, 0)),
            self._color_transforms = [None]*len(self._images)

        # now reconstruct
        final = ls[0]
        for i in range(1,layers):
            final = cv.pyrUp(final)
            final = cv.add(final, ls[i])

        return final
