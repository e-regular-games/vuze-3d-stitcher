#!/usr/bin/python

import coordinates
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

# opencv uses HSV ranges: (0, 179), (0, 255), (0, 255)
def normal_hsv(hsv):
    return hsv / [179, 255, 255]

def normal_bgr(bgr):
    return bgr / [255, 255, 255]

    # given a 3 dimensional cube
    # where the 1st dim identifies a row of values (N rows)
    # the 2nd dim identifies a value
    # and the 3rd dim identifies the color component (HSV)
    # @returns an Nx4 matrix, the 4th column is the closeness of the hues
def average_hsv(hsv):
    hue_a = hsv[:,:,0] * math.pi / 180.0 * 2.0
    hue_sin = np.mean(np.sin(hue_a), axis=1)
    hue_cos = np.mean(np.cos(hue_a), axis=1)
    hue_a_avg = np.arctan2(hue_sin, hue_cos) % (2 * math.pi)
    hue_a_mag = np.sqrt(hue_sin*hue_sin + hue_cos*hue_cos)

    avg = np.zeros((hsv.shape[0], 4))
    avg[:,0] = hue_a_avg / math.pi * 180.0 / 2
    avg[:,1] = np.mean(hsv[:,:,1], axis=1)
    avg[:,2] = np.mean(hsv[:,:,2], axis=1)
    avg[:,3] = hue_a_mag
    return avg

class ColorTransform():
    def __init__(self, image, debug):
        self._image = image
        self._debug = debug
        self._c = np.zeros((6, 6))

        if debug.enable('color_regression'):
            self._fig = plt.figure()

    def get_uncorrected_hsv(self, polar, size):
        rpp = math.pi / self._image.shape[0]

        all_polar = np.zeros((0, 2))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp
                all_polar = np.concatenate([all_polar, polar + [dphi, dtheta]])

        all_eqr = coordinates.polar_to_eqr(all_polar, self._image.shape)
        all_bgr = coordinates.eqr_interp(all_eqr, self._image)
        all_hsv = cv.cvtColor(all_bgr.reshape(all_bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                    .reshape(all_bgr.shape[0], 3)
        return all_hsv

    def get_uncorrected_hsv_mean(self, polar, size):
        rpp = math.pi / self._image.shape[0]

        all_polar = np.zeros((polar.shape[0], size * size, 2))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp
                all_polar[:, i*size + j] = polar + [dphi, dtheta]

        all_hsv = np.zeros((polar.shape[0], size * size, 3))
        for i in range(size * size):
            all_eqr = coordinates.polar_to_eqr(all_polar[:, i], self._image.shape)
            all_bgr = coordinates.eqr_interp(all_eqr, self._image)
            all_hsv[:, i] = cv.cvtColor(all_bgr.reshape(all_bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                           .reshape(all_bgr.shape[0], 3)
        return average_hsv(all_hsv)[:,:3]

    # i, f are HSV color arrays, Nx3
    def compute_hsv_coeffs(self, i, f, c):
        hue_i = (i[:,0] * math.pi / 180.0 * 2.0).reshape((i.shape[0], 1))
        hue_f = (f[:,0] * math.pi / 180.0 * 2.0).reshape((i.shape[0], 1))
        x = np.concatenate([np.sin(hue_i), np.cos(hue_i), i[:,1:3] / [255, 255], np.full((i.shape[0], 1), 1)], axis=1)
        y = np.concatenate([np.sin(hue_f), np.cos(hue_f), f[:,1:3] / [255, 255], np.full((i.shape[0], 1), 1)], axis=1)

        if self._debug.enable('color_regression'):
            ax = self._fig.add_subplot(2, 2, 1, projection='3d')
            ax.plot3D(x[:,0], x[:,1], y[:,0] - x[:,0], 'ro', markersize=1)
            self._fig.add_subplot(2, 2, 3).hist(y[:,0] - x[:,0])

        Q, R = np.linalg.qr(x)
        self._c = np.linalg.inv(R).dot(np.transpose(Q)) @ y
        print(self._c)

        err = y - np.clip(x @ self._c, -1, 1)
        if self._debug.verbose:
            print('err', np.mean(err, axis=0), np.std(err, axis=0))

        if self._debug.enable('color_regression'):
            self._fig.add_subplot(2, 2, 4).hist(err[:,0])
            self._fig.add_subplot(2, 2, 2, projection='3d') \
                     .plot3D(x[:,0], x[:,1], err[:,0], 'ro', markersize=1)

    def correct_hsv(self, hsv, c):
        hue = (hsv[:,0] * math.pi / 180.0 * 2.0).reshape((hsv.shape[0], 1))
        x = np.concatenate([np.sin(hue), np.cos(hue), hsv[:,1:3] / [255, 255], np.full((hsv.shape[0], 1), 1)], axis=1)

        cor = np.clip(x @ self._c, -1, 1)

        r = np.zeros(hsv.shape)
        r[:,0] = 180 / (2 * math.pi) * (np.arctan2(cor[:,0], cor[:,1]) % (2 * math.pi))
        r[:,1:3] = np.clip(cor[:,2:4], 0, 1) * 255
        return r

    def correct_bgr(self, bgr, c):
        n = bgr.shape[0]
        bgr_f = bgr.reshape((n, 1, 3)).astype(np.uint8)
        hsv = cv.cvtColor(bgr_f, cv.COLOR_BGR2HSV)
        hsv_c = self.correct_hsv(hsv.reshape((n, 3)), c)
        hsv_f = hsv_c.reshape((n, 1, 3)).astype(np.uint8)
        return cv.cvtColor(hsv_f, cv.COLOR_HSV2BGR).reshape((n, 3))

class ColorTransition():
    def __init__(self, image, seam_left, seam_right, debug):
        self._image = image
        self._debug = debug
        self._c = np.zeros((4, 4))
        self._fade_dist = 10 * math.pi / 180

        if debug.enable('color_regression'):
            self._fig = plt.figure()

        self._seam_left = seam_left
        self._seam_right = seam_right


    def get_uncorrected_hsv(self, polar, size):
        rpp = math.pi / self._image.shape[0]

        all_polar = np.zeros((0, 2))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp
                all_polar = np.concatenate([all_polar, polar + [dphi, dtheta]])

        all_eqr = coordinates.polar_to_eqr(all_polar, self._image.shape)
        all_bgr = coordinates.eqr_interp(all_eqr, self._image)
        all_hsv = cv.cvtColor(all_bgr.reshape(all_bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                    .reshape(all_bgr.shape[0], 3)
        return all_hsv

    def _distance_from_seam(self, seam, polar):
        phi = seam[:, 0]
        theta = seam[:, 1]
        slope = (theta[1:] - theta[:-1]) / (phi[1:] - phi[:-1])
        offset = theta[:-1] - slope * phi[:-1]
        seam_theta = np.zeros(polar.shape[0])
        COMP_SIZE = 1000000
        for i in range(0, polar.shape[0], COMP_SIZE):
            p = polar[i:i+COMP_SIZE,:] % [math.pi, 2 * math.pi]
            f_mat = np.ones((p.shape[0], slope.shape[0] + 1)) * seam[:, 0]
            in_range = np.logical_and(p[:,0:1] <= f_mat[:,1:], p[:,0:1] >= f_mat[:,:-1])

            f_slope = (np.ones((p.shape[0], slope.shape[0])) * slope)[in_range]
            f_offset = (np.ones((p.shape[0], offset.shape[0])) * offset)[in_range]

            seam_theta[i:i+COMP_SIZE] = p[:,0] * f_slope + f_offset

        return seam_theta - polar[:,1]


    def get_uncorrected_hsv_mean(self, polar, size):
        rpp = math.pi / self._image.shape[0]

        all_polar = np.zeros((polar.shape[0], size * size, 2))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp
                all_polar[:, i*size + j] = polar + [dphi, dtheta]

        all_hsv = np.zeros((polar.shape[0], size * size, 3))
        for i in range(size * size):
            all_eqr = coordinates.polar_to_eqr(all_polar[:, i], self._image.shape)
            all_bgr = coordinates.eqr_interp(all_eqr, self._image)
            all_hsv[:, i] = cv.cvtColor(all_bgr.reshape(all_bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                           .reshape(all_bgr.shape[0], 3)
        return average_hsv(all_hsv)[:,:3]

    # i, f are HSV color arrays, Nx3
    def compute_hsv_coeffs(self, i, f, c):
        hue_i = (i[:,0] * math.pi / 180.0 * 2.0).reshape((i.shape[0], 1))
        hue_f = (f[:,0] * math.pi / 180.0 * 2.0).reshape((i.shape[0], 1))
        x = np.concatenate([np.sin(hue_i), np.cos(hue_i), i[:,1:3] / [255, 255], np.full((i.shape[0], 1), 1)], axis=1)
        y = np.concatenate([np.sin(hue_f), np.cos(hue_f), f[:,1:3] / [255, 255], np.full((i.shape[0], 1), 1)], axis=1)

        if self._debug.enable('color_regression'):
            ax = self._fig.add_subplot(1, 4, 1, projection='3d')
            ax.plot3D(x[:,0], x[:,1], y[:,0] - x[:,0], 'ro', markersize=1)
            self._fig.add_subplot(1, 4, 2).hist(y[:,0] - x[:,0])

        Q, R = np.linalg.qr(x)
        self._c = np.linalg.inv(R).dot(np.transpose(Q)) @ (y-x)

        err = (y-x) - np.clip(x @ self._c, -1, 1)
        if self._debug.verbose:
            print('err', np.mean(err, axis=0), np.std(err, axis=0))

        if self._debug.enable('color_regression'):
            self._fig.add_subplot(1, 4, 3).hist(err[:,0])
            self._fig.add_subplot(1, 4, 4, projection='3d') \
                     .plot3D(x[:,0], x[:,1], err[:,0], 'ro', markersize=1)

    def correct_hsv(self, hsv, c):
        df = np.zeros((hsv.shape[0]))
        df += 1.0 - np.clip(np.absolute(self._distance_from_seam(self._seam_left - [0, math.pi/2], c) / self._fade_dist), 0, 1)
        df += 1.0 - np.clip(np.absolute(self._distance_from_seam(self._seam_right, c) / self._fade_dist), 0, 1)
        hue = (hsv[:,0] * math.pi / 180.0 * 2.0).reshape((hsv.shape[0], 1))
        x = np.concatenate([np.sin(hue), np.cos(hue), hsv[:,1:3] / [255, 255], np.full((hsv.shape[0], 1), 1)], axis=1)

        cor = x + df.reshape((df.shape[0], 1)) * np.clip(x @ self._c, -1, 1)

        r = np.zeros(hsv.shape)
        r[:,0] = 180 / (2 * math.pi) * (np.arctan2(cor[:,0], cor[:,1]) % (2 * math.pi))
        r[:,1:3] = np.clip(cor[:,2:4], 0, 1) * 255
        return r

    def correct_bgr(self, bgr, c):
        n = bgr.shape[0]
        bgr_f = bgr.reshape((n, 1, 3)).astype(np.uint8)
        hsv = cv.cvtColor(bgr_f, cv.COLOR_BGR2HSV)
        hsv_c = self.correct_hsv(hsv.reshape((n, 3)), c)
        hsv_f = hsv_c.reshape((n, 1, 3)).astype(np.uint8)
        return cv.cvtColor(hsv_f, cv.COLOR_HSV2BGR).reshape((n, 3))

class ColorCorrection():
    def __init__(self, images, debug):
        self._images = images
        self._debug = debug
        self._size = 7
        self._align_thres = 0.90

    def match_colors(self, matches):
        transforms = [ColorTransform(i, self._debug) for i in self._images]
        return self._regression(matches, transforms)

    def fade_colors(self, matches, seams, fade):
        transforms = [ColorTransition(im, seams[(i-2)%8], seams[i], self._debug) \
                      for i, im in enumerate(self._images)]
        for t in transforms:
            t._fade_dist = fade
        return self._regression(matches, transforms)

    def _regression(self, matches, transforms):
        colors = []
        targets = []
        left = [0, math.pi / 2]
        fig = plt.figure() if self._debug.enable('color') else None
        for i, m in enumerate(matches):
            ll = (2*i-2) % 8
            lr = 2*i % 8
            rl = (2*i-1) % 8
            rr = (2*i+1) % 8

            color = np.zeros((m.shape[0], 4, 3))
            color[:,0,:] = transforms[ll].get_uncorrected_hsv_mean(m[:,0:2], self._size)
            color[:,1,:] = transforms[lr].get_uncorrected_hsv_mean(m[:,2:4] - left, self._size)
            color[:,2,:] = transforms[rl].get_uncorrected_hsv_mean(m[:,4:6], self._size)
            color[:,3,:] = transforms[rr].get_uncorrected_hsv_mean(m[:,6:8] - left, self._size)
            colors.append(color)

            target = average_hsv(color)
            targets.append(target)

            if self._debug.enable('color'):
                t = target[:,0:3].reshape(target.shape[0], 1, 3)
                img = np.concatenate([color, t], axis=1).astype(np.uint8)
                fig.add_subplot(1, 4, i+1).imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))

        for i in range(8):
            l = int(i/2)
            r = (l+1) % 4
            cl = 2 * (i % 2) + 1 # column within the matched set of 4 images.
            cr = 2 * (i % 2) # column within the matched set of 4 images.
            clc = 4 * (i % 2) + 2
            clr = 4 * (i % 2)
            color = np.concatenate([colors[l][:,cl], colors[r][:,cr]])
            target = np.concatenate([targets[l], targets[r]])
            coords = np.concatenate([matches[l][:,clc:clc+2] - [0, math.pi/2], matches[r][:,clr:clr+2]])
            inc = target[:,3] > self._align_thres
            transforms[i].compute_hsv_coeffs(color[inc], target[inc,:3], coords[inc])

        return transforms
