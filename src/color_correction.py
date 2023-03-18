#!/usr/bin/python

import coordinates
import cv2 as cv
import linear_regression
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from skimage.exposure import match_histograms
from coordinates import to_1d

class ColorTransform():
    def __init__(self, image, debug):
        self._image = image
        self._debug = debug

    def compute_bgr(self, i, f, c):
        pass

    # operates on a matrix of bgr(float), global coords, local coords, and
    # a boolean filter of which elements to calculate.
    def delta_bgr(self, bgr, c_global, c_local, flt):
        delta = np.zeros(bgr.shape, np.float32)
        delta[flt] = self._delta_bgr(bgr[flt], c_local[flt])
        return delta

    # requires Nx3 matrices for bgr(float32) and c.
    def _delta_bgr(self, bgr, c):
        return np.zeros(bgr.shape, np.float32)

    # bgr, c_global, c_local can be 3d matrices
    # bgr is expected to be uint8 in the range [0,255]
    # returns bgr(uint8) in the range [0, 255]
    def correct_bgr(self, bgr, c_global, c_local, flt):
        delta = self.delta_bgr(bgr / 255.0, c_global, c_local, flt)

        K = 15
        kernel = np.ones((K,K), np.float32)/(K**2)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug.enable('color-delta'):
            norm = (delta + 1) / 2
            self._debug.figure('color-delta', True)
            for i in range(3):
                self._debug.set_subplot(2, 2, i+1) \
                    .subplot('color-delta') \
                    .imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))
        return np.round(255.0 * np.clip(bgr / 255.0 + delta, 0, 1)).astype(np.uint8)

class ColorTransformTable(ColorTransform):
    def __init__(self, image, debug):
        super().__init__(image, debug)
        self._tree = None # will translate from uint8 to float
        self._delta = None # will translate from uint8 to float

    # expects i and f to be Nx3
    # expects c to be Nx2
    def compute_bgr(self, i, f, c):
        unq_i, idx, inv, cnts = np.unique(i, axis=0, return_index=True, return_inverse=True, return_counts=True)
        # these conversions to float before doing the subtraction is very important
        collapsed = np.zeros(unq_i.shape, np.float32)
        collapsed[:,0] = np.bincount(inv, weights=f[:,0].astype(np.float32)) / cnts
        collapsed[:,1] = np.bincount(inv, weights=f[:,1].astype(np.float32)) / cnts
        collapsed[:,2] = np.bincount(inv, weights=f[:,2].astype(np.float32)) / cnts

        delta = (collapsed - unq_i.astype(np.float32)) / 255.0
        dist = np.sqrt(np.sum(delta**2, axis=-1))
        keep = linear_regression.trim_outliers(delta, 0.75)
        self._delta = delta[keep]
        self._tree = KDTree(unq_i[keep] / 255.0)
        return np.zeros((i.shape[0],), np.float32)

    # expects bgr to be in the range of [0.0, 1.0]
    def _delta_bgr(self, bgr, c):
        dist, idx = self._tree.query(bgr, k=4, workers=12)

        result = np.zeros(bgr.shape, np.float32)
        exact = np.abs(dist[:,0]) < 0.00001
        if np.count_nonzero(exact) > 0:
            result[exact] = self._delta[idx[exact,0]]

        MAX_DIST = 0.1
        out_of_range = (dist > MAX_DIST).any(axis=-1)

        approx = np.logical_and(np.logical_not(exact), np.logical_not(out_of_range))
        if np.count_nonzero(approx) > 0:
            dist = dist[approx]
            idx = idx[approx]
            scale = (MAX_DIST - np.min(dist, axis=-1).reshape(dist.shape[:-1] + (1,))) / MAX_DIST
            limit = (np.max(dist, axis=-1) + np.min(dist, axis=-1)).reshape(dist.shape[:-1] + (1,))
            dist = (limit - dist) ** 2
            result[approx] = scale * (np.sum(self._delta[idx].reshape(dist.shape + (3,)) * dist.reshape(dist.shape + (1,)), axis=-2) \
                              / np.sum(dist, axis=-1).reshape(dist.shape[:-1] + (1,)))
        return result

class ColorTransformSided(ColorTransform):
    def __init__(self, image, debug):
        super().__init__(image, debug)

        self._left = ColorTransformTable(image, debug.set_subplot(1, 2, 1))
        self._right = ColorTransformTable(image, debug.set_subplot(1, 2, 2))

    def compute_bgr(self, i, f, c):
        left = c[...,1] <= math.pi
        right = c[...,1] > math.pi

        errors = np.zeros(i.shape[:-1])
        errors[left] = self._left.compute_bgr(i[left], f[left], c[left])
        errors[right] = self._right.compute_bgr(i[right], f[right], c[right])
        return errors

    def cubic_fade(self, x, r):
        x = np.clip(x, 0, r)
        a = 6 / (r * r * r)
        return  a * r * np.power(x, 2) / 2 - a * np.power(x, 3) / 3

    def delta_bgr(self, bgr, c_global, c_local, flt):
        left_delta = self._left.delta_bgr(bgr, c_global, c_local, flt)
        right_delta = self._right.delta_bgr(bgr, c_global, c_local, flt)

        right_weight = np.zeros(flt.shape + (1,))
        right_weight[flt,0] = self.cubic_fade(c_local[flt,1] - 3*math.pi/4, math.pi/2)

        left_weight = np.zeros(flt.shape + (1,))
        left_weight[flt,0] = 1.0 - right_weight[flt,0]

        return left_weight * left_delta + right_weight * right_delta

class ColorCorrection():
    def __init__(self, images, debug):
        self._images = images
        self._debug = debug

    def _generate_regions(self):
        # needs to be implemented by derived classes.
        pass

    def match_colors(self):
        regions, coords = self._generate_regions()

        targets = []
        for i in range(4):
            region_mean = np.median(regions[4*i:4*i+4], axis=0)
            bgr = region_mean.astype(np.uint8)
            targets.append(bgr)

        return self._compute(regions, coords, targets)

    def _compute(self, regions, coords, targets):
        transforms = []
        dbg_color = []

        height = np.max([r.shape[0] for r in regions])
        black = np.zeros((height, 1, 3))
        def debug_resize(blocks):
            result = []
            for b in blocks:
                r = np.zeros((height,) + b.shape[1:])
                r[:b.shape[0]] = b
                result.append(r)
            return result

        self._debug.log_pause();

        for i in range(8):
            left_bgr = regions[(2*i+1) % 16].astype(np.uint8)
            left_coord = coords[(2*i+1) % 16]
            left_target = targets[int(i/2)]

            right_bgr = regions[(2*i+4) % 16].astype(np.uint8)
            right_coord = coords[(2*i+4) % 16]
            right_target = targets[int(i/2+1)%4]

            bgr = np.concatenate([left_bgr, right_bgr])
            target = np.concatenate([left_target, right_target])
            coord = np.concatenate([left_coord, right_coord])

            t = ColorTransformSided(self._images[i], self._debug.window('image ' + str(i)))
            t.compute_bgr(to_1d(bgr), to_1d(target), to_1d(coord))
            transforms.append(t)

            print('.', end='', flush=True)

            if self._debug.enable('color'):
                left_corrected = t.correct_bgr(left_bgr, left_coord, left_coord, np.full(left_bgr.shape[:-1], True))
                right_corrected = t.correct_bgr(right_bgr, right_coord, right_coord, np.full(right_bgr.shape[:-1], True))
                dbg_color = dbg_color + debug_resize([
                    left_bgr, black,
                    left_target, black,
                    left_corrected, black, black,
                    right_bgr, black,
                    right_target, black,
                    right_corrected, black, black, black, black
                ])

        print('')
        self._debug.log_resume();

        if self._debug.enable('color'):
            f = plt.figure()
            f.canvas.manager.set_window_title('color_with_target')
            img = np.concatenate(dbg_color, axis=1).astype(np.uint8)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        return transforms

class ColorCorrectionSeams(ColorCorrection):
    def __init__(self, images, transforms, seams, debug):
        super().__init__(images, debug)
        self._transforms = transforms
        self._seams = seams
        self.border = 5 / 180 * math.pi

    def _generate_regions(self):
        h = self._images[0].shape[0]
        w = self._images[0].shape[1]
        y = np.arange(h/20, 19*h/20)
        phi = y / (h-1) * math.pi
        x_range = self.border / (2 * math.pi) * w
        x_delta = np.arange(math.floor(-1 * x_range / 2), math.ceil(x_range / 2))
        theta_delta = x_delta / (w-1) * 2 * math.pi

        regions = []
        coords = []
        for i, seam in enumerate(self._seams):
            ll = (i-2) % 8
            lr = i

            intersect = coordinates.seam_intersect(seam + [0, math.pi], phi)
            theta = intersect.reshape(intersect.shape + (1,)) + theta_delta
            plr = np.zeros(theta.shape + (2,))
            plr[...,0] = phi.reshape(phi.shape + (1,)) * np.ones(theta.shape)
            plr[...,1] = theta

            for j, idx in enumerate([ll, lr]):
                shift = [0, math.pi / 2] if j % 2 == 1 else [0, 0]
                coord = self._transforms[idx].reverse(to_1d(plr - shift)) \
                                             .reshape(plr.shape)
                coords.append(coord)
                eqr = coordinates.polar_to_eqr(coord, self._images[idx].shape)
                bgr = coordinates.eqr_interp_3d(eqr.astype(np.float32), self._images[idx]) \
                    .astype(np.uint8)
                regions.append(bgr)

        if self._debug.enable('color'):
            f = plt.figure()
            f.canvas.manager.set_window_title('color_slices')
            img = np.concatenate(regions, axis=1).astype(np.uint8)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        return regions, coords
