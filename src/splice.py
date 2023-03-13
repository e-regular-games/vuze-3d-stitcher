#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
import threading
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular

# polar is assumed to be an N x M x 2 matrix with each row
# having the same value of phi.
def compute_left_side(polar, stitch, margin):
    phi = polar[:,0,0:1]
    phi_n = phi.shape[0]
    seam_theta = coordinates.seam_intersect(stitch, phi)

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

def filter_to_stitch(polar, l, margin, stitches):
    f = None
    if l == 0:
        right = compute_left_side(polar, stitches[l], margin)
        left = 1.0 - compute_left_side(polar, stitches[-1], margin)
        f = left + right
    else:
        left = compute_left_side(polar, stitches[l], margin)
        right = 1.0 - compute_left_side(polar, stitches[l - 1], margin)
        f = left * right

    flt = f > 0
    return f, flt

class ComputeSegment(threading.Thread):
    def __init__(self, image, polar, stitches, idx, margin, pt_transform, clr_transform, debug):
        threading.Thread.__init__(self)
        self._image = image
        self._polar = polar
        self._stitches = stitches
        self._idx = idx
        self._margin = margin
        self._pt_transform = pt_transform
        self._clr_transform = clr_transform
        self._debug = debug

        self.result = np.zeros(polar.shape[:-1] + (3,), dtype=np.uint8)

    def run(self):
        self._debug.perf('splice.filter')
        global_pts_polar = self._polar.copy()
        weight, flt = filter_to_stitch(self._polar, self._idx, self._margin, self._stitches)
        self._debug.perf('splice.filter')

        global_pts_polar[flt,1] -= self._idx * math.pi / 2
        global_pts_polar[flt,1] += math.pi
        global_pts_polar[flt,1] = global_pts_polar[flt,1] % (2 * math.pi)

        self._debug.perf('splice.transform')
        local_pts_polar = np.zeros(self._polar.shape)
        local_pts_polar[flt] = self._pt_transform.reverse(global_pts_polar[flt])
        self._debug.perf('splice.transform')

        self._debug.perf('splice.interp')
        local_pts_eqr = np.zeros(self._polar.shape)
        local_pts_eqr[flt] = coordinates.polar_to_eqr(local_pts_polar[flt], self._polar.shape)

        pixels = np.zeros(self._polar.shape[:-1] + (3,), np.uint8)
        pixels[flt] = coordinates.eqr_interp(local_pts_eqr[flt], self._image)
        local_pts_eqr = None
        self._debug.perf('splice.interp')

        self._debug.perf('splice.color')
        if self._clr_transform is not None:
            global_pts_polar -= [0, math.pi]
            pixels = self._clr_transform \
                         .correct_bgr(pixels, global_pts_polar, local_pts_polar, flt)
        self._debug.perf('splice.color')

        n = np.count_nonzero(flt)
        self.result[flt] = (pixels[flt] * weight[flt].reshape((n, 1))).round().astype(np.uint8)
        print('.', end='', flush=True)

class SpliceImages():
    def __init__(self, images, debug):
        self._images = images

        # tansform(i) of image (i)
        self._transforms = [None]*len(images)
        self._color_transforms = [None]*len(images)
        self._calibration = [None]*len(images)

        # seam(i) between images (i) and (i+1)
        self._stitches = [None]*len(images)

        self._debug = debug
        self._view0 = 0
        self._rotate_x = 0
        self._rotate_y = 0

    def set_transform(self, idx, t):
        self._transforms[idx] = t

    def set_color_transform(self, idx, cc):
        self._color_transforms[idx] = cc

    def set_initial_view(self, deg):
        self._view0 = deg / 180 * math.pi

    def set_camera_rotations(self, x, y):
        self._rotate_x = x
        self._rotate_y = y

    def set_calibration(self, calib):
        self._calibration = [c.t for c in calib]

    # line is a numpy array of (phi, theta) rows
    # the first row should be phi = 0, and the last should be phi = pi
    def set_stitch(self, idx, line):
        self._stitches[idx] = line

    def _apply_rotations(self, image, eq):
        if self._rotate_x == 0 and self._rotate_y == 0:
            return image

        plr = coordinates.eqr_to_polar(coordinates.to_1d(eq), image.shape)
        cart = coordinates.polar_to_cart(plr, 2)
        X = np.array([[math.cos(self._rotate_x), 0, math.sin(self._rotate_x)],
                      [0, 1, 0],
                      [-math.sin(self._rotate_x), 0, math.cos(self._rotate_x)]], dtype=np.float32)
        Y = np.array([[1, 0, 0],
                      [0, math.cos(self._rotate_y), math.sin(self._rotate_y)],
                      [0, -math.sin(self._rotate_y), math.cos(self._rotate_y)]], dtype=np.float32)
        cart = np.matmul(Y, np.matmul(X, cart.transpose())).transpose()

        plr = coordinates.cart_to_polar(cart)
        eq = coordinates.polar_to_eqr(plr, image.shape)

        return coordinates.eqr_interp(eq, image).reshape(image.shape)

    def _generate(self, v_res, images, margin):
        shape = (v_res, 2 * v_res, 3)
        mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        eq = np.concatenate([x.reshape((x.shape[0], x.shape[1], 1)) for x in mesh], axis=2)

        polar = coordinates.eqr_to_polar_3d(eq)
        threads = []
        for s in range(len(self._stitches)):
            t = ComputeSegment(images[s], polar, self._stitches, s, margin, \
                               self._transforms[s], self._color_transforms[s], self._debug)

            if self._debug.enable_threads:
                t.start()
            else:
                t.run()
            threads.append(t)

        result = np.zeros(shape, dtype=np.uint8)
        for t in threads:
            if self._debug.enable_threads:
                t.join()
            result += t.result

        result = self._apply_rotations(result, eq)

        # adjust for view0 being a non-zero value
        if self._view0 != 0:
            view0_l = int(self._view0 / math.pi * v_res)
            view0_r = 2 * v_res - view0_l
            shift = result[:,:view0_l].copy()
            result[:,:view0_r] = result[:,view0_l:]
            result[:,view0_r:] = shift
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
