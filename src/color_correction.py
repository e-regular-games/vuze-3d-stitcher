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

def hsv_to_trig_norm(hsv):
    hue = hsv[...,0:1] * math.pi / 180.0 * 2.0
    return np.concatenate([
        np.sin(hue) / 2 + 0.5,
        np.cos(hue) / 2 + 0.5,
        hsv[...,1:2] / 255,
        hsv[...,2:3] / 255
    ], axis=-1)

def trig_norm_to_hsv(t):
    hue = 180 / (2 * math.pi) * (np.arctan2(t[...,0:1] * 2 - 1, t[...,1:2] * 2 - 1) % (2 * math.pi))
    return np.concatenate([
        hue,
        t[...,2:3] * 255,
        t[...,3:4] * 255
    ], axis=-1)

def hsv_stats(hsv):
    hue_a = hsv[:,0] * math.pi / 180.0 * 2.0
    hue_sin = np.mean(np.sin(hue_a))
    hue_cos = np.mean(np.cos(hue_a))
    hue_mean = np.arctan2(hue_sin, hue_cos) % (2 * math.pi)
    hue_align = np.sqrt(hue_sin*hue_sin + hue_cos*hue_cos)

    return [hue_mean * 180 / math.pi / 2, np.mean(hsv[:,1]), np.mean(hsv[:,2])], \
        [hue_align, np.std(hsv[:,1]), np.std(hsv[:,2])]

def hsv_error(i, f):
    hues = np.concatenate([i[:,0:1], f[:,0:1]], axis=1)
    hues = hues * math.pi / 180.0 * 2.0
    hues_sin = np.sin(hues)
    hues_cos = np.cos(hues)

    hues_sin = (hues_sin[:,1] + hues_sin[:,0]) / 2
    hues_cos = (hues_cos[:,1] + hues_cos[:,0]) / 2
    hues_align = np.sqrt(hues_sin*hues_sin + hues_cos*hues_cos)
    sat_diff = f[:,1:2] - i[:,1:2]
    val_diff = f[:,2:3] - i[:,2:3]

    return np.array([1 - np.median(hues_align), np.mean(sat_diff), np.mean(val_diff),
                     np.std(hues_align), np.std(val_diff), np.std(sat_diff)])

# @returns a 2x3 matrix with [[h_mean, s_min, v_min], [h_min_mag, s_max, v_max]]
def hsv_range(hsv, hue_align_thres, sat_thres, val_thres):
    hue_a = hsv[:,0] * math.pi / 180.0 * 2.0
    hue_sin = np.sin(hue_a)
    hue_cos = np.cos(hue_a)
    hue_median = np.arctan2(np.median(hue_sin), np.median(hue_cos)) % (2 * math.pi)
    hue_px_align_sin = (hue_sin + np.sin(hue_median)) / 2
    hue_px_align_cos = (hue_cos + np.cos(hue_median)) / 2
    hue_align = np.log(np.sqrt(hue_px_align_sin*hue_px_align_sin + hue_px_align_cos*hue_px_align_cos))

    rg = np.zeros((2, 3))
    rg[0,0] = hue_median * 180.0 / 2.0 / math.pi
    rg[1,0] = np.exp(np.median(hue_align) - hue_align_thres * np.std(hue_align))
    rg[0,1:3] = np.mean(hsv[:,1:3], axis=0) - [sat_thres, val_thres] * np.std(hsv[:,1:3], axis=0)
    rg[1,1:3] = np.mean(hsv[:,1:3], axis=0) + [sat_thres, val_thres] * np.std(hsv[:,1:3], axis=0)
    return rg

# @returns an aligned vector of boolean flags. True for elements in the range.
def hsv_in_range(hsv, rg):
    hue_a = hsv[...,0] * math.pi / 180.0 * 2.0
    hue_sin = np.sin(hue_a)
    hue_cos = np.cos(hue_a)
    hue_median = rg[0,0] * math.pi / 180.0 * 2.0
    hue_px_align_sin = (hue_sin + np.sin(hue_median)) / 2
    hue_px_align_cos = (hue_cos + np.cos(hue_median)) / 2
    hue_align = np.sqrt(hue_px_align_sin*hue_px_align_sin + hue_px_align_cos*hue_px_align_cos)

    in_range = np.zeros(hsv.shape[:-1] + (3,))
    in_range[...,0] = hue_align > rg[1,0]
    in_range[...,1:3] = np.logical_and(hsv[...,1:3] >= rg[0,1:3], hsv[...,1:3] <= rg[1,1:3])
    return np.all(in_range, axis=-1)

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

def mean_hsv(hsv):
    hue_a = hsv[...,0] * math.pi / 180.0 * 2.0
    hue_sin = np.mean(np.sin(hue_a))
    hue_cos = np.mean(np.cos(hue_a))
    hue_a_avg = np.arctan2(hue_sin, hue_cos) % (2 * math.pi)
    return [hue_a_avg / math.pi * 180.0 / 2, np.mean(hsv[...,1]), np.mean(hsv[...,2])]

def trim_outliers(hsv, hue_align_thres, sat_thres, val_thres):
    hue = hsv[:,0] * math.pi / 180.0 * 2.0
    hue_sin = np.sin(hue)
    hue_cos = np.cos(hue)
    hue_mean = np.arctan2(np.mean(hue_sin), np.mean(hue_cos)) % (2 * math.pi)
    hue_px_align_sin = (hue_sin + np.sin(hue_mean)) / 2
    hue_px_align_cos = (hue_cos + np.cos(hue_mean)) / 2

    hue_align = np.sqrt(hue_px_align_sin*hue_px_align_sin + hue_px_align_cos*hue_px_align_cos)
    hue_inc = np.absolute(hue_align - np.median(hue_align)) < hue_align_thres * np.std(hue_align)

    sat_mean = np.mean(hsv[:,1])
    sat_std = np.std(hsv[:,1])
    sat_inc = np.absolute(hsv[:,1] - sat_mean) < sat_thres * sat_std

    val_mean = np.mean(hsv[:,2])
    val_std = np.std(hsv[:,2])
    val_inc = np.absolute(hsv[:,2] - val_mean) < val_thres * val_std

    return np.logical_and(hue_inc, np.logical_and(sat_inc, val_inc))

def trim_diff_outliers(i, f, hue_align_thres, sat_thres, val_thres):
    hues = np.concatenate([i[:,0:1], f[:,0:1]], axis=1)
    hues = hues * math.pi / 180.0 * 2.0
    hues_sin = np.sin(hues)
    hues_cos = np.cos(hues)

    hues_sin_diff = (hues_sin[:,1] + hues_sin[:,0]) / 2
    hues_cos_diff = (hues_cos[:,1] + hues_cos[:,0]) / 2

    hues_align = np.log(np.sqrt(hues_sin_diff*hues_sin_diff + hues_cos_diff*hues_cos_diff))
    hues_inc = np.absolute(hues_align - np.median(hues_align)) < hue_align_thres * np.std(hues_align)

    diff = f[:,1:3] - i[:,1:3]
    m = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)
    sat_inc = np.absolute(diff[:,0] - m[0]) < sat_thres * std[0]
    val_inc = np.absolute(diff[:,1] - m[1]) < val_thres * std[1]

    return np.logical_and(hues_inc, np.logical_and(sat_inc, val_inc))

def hsv_to_rgb_hex(hsv):
    n = hsv.shape[0]
    rgb = cv.cvtColor(hsv.reshape((n, 1, 3)).astype(np.uint8), cv.COLOR_HSV2RGB).reshape((n, 3))
    rgb = rgb[:,0] * 0x10000 + rgb[:,1] * 0x100 + rgb[:,2]
    strings = np.vectorize(lambda x: np.base_repr(x, 16))(rgb)
    return np.char.add('#', np.char.zfill(strings, 6))

def plot_color_errors(ax, i, f):
    rgb_i = hsv_to_rgb_hex(i)
    rgb_f = hsv_to_rgb_hex(f)

    hues = np.concatenate([i[:,0:1], f[:,0:1]], axis=1)
    hues = hues * math.pi / 180.0 * 2.0
    hues_sin = np.sin(hues)
    hues_cos = np.cos(hues)

    hues_sin_diff = (hues_sin[:,1] + hues_sin[:,0]) / 2
    hues_cos_diff = (hues_cos[:,1] + hues_cos[:,0]) / 2

    hue_align = np.sqrt(hues_sin_diff*hues_sin_diff + hues_cos_diff*hues_cos_diff)
    sat_diff = (f[:,1] - i[:,1]) / 255
    val_diff = (f[:,2] - i[:,2]) / 255

    dist = np.sqrt(sat_diff*sat_diff + val_diff*val_diff)

    ax.scatter(sat_diff, val_diff, np.log(hue_align), c=rgb_i, marker='.')


class ColorTransformArea():
    def __init__(self, image, debug):
        self._image = image
        self._debug = debug
        self._size = 5
        self._x_size = 11
        self._c = np.zeros((self._x_size, self._x_size))

    def coord_sizing(self, n):
        return n * self._size * self._size

    def get_uncorrected_hsv(self, polar):
        rpp = math.pi / self._image.shape[0]
        size = self._size

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
        return all_hsv, all_polar

    def get_target_hsv(self, i, c):
        return i

    def compute_hsv_coeffs(self, i, f, c):
        inc = trim_diff_outliers(i, f, 3, 3, 3)
        i = i[inc]
        f = f[inc]
        c = c[inc]

        x = self._arrange_hsv(i, c)
        y = self._arrange_hsv(f, c)

        if self._debug[0].enable('color_regression'):
            ax = self._debug[0] \
                     .figure('color_regression_0') \
                     .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1, \
                                  projection='3d')
            plot_color_errors(ax, i, f)

        Q, R = np.linalg.qr(x)
        self._c = np.linalg.inv(R).dot(np.transpose(Q)) @ y

        y_a = self._correct_hsv(i, c)

        if self._debug[0].enable('color_regression'):
            ax = self._debug[0] \
                     .figure('color_regression_1') \
                     .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1, \
                                  projection='3d')
            plot_color_errors(ax, y_a, f)

        err_i = hsv_error(i, f)
        err_f = hsv_error(y_a, f)
        return np.absolute((err_f - err_i) / err_i)

    def correct_hsv(self, hsv, c_global, c_local):
        return self._correct_hsv(hsv, c_local)

    def _correct_hsv(self, hsv, c_local):
        x = self._arrange_hsv(hsv, c_local)
        cor = np.clip(x @ self._c, 0, 1)
        return trig_norm_to_hsv(cor)

    def _arrange_hsv(self, hsv, c):
        hue = (hsv[...,0:1] * math.pi / 180.0 * 2.0)
        return np.concatenate([
            hsv_to_trig_norm(hsv),
            np.full(hsv.shape[:-1] + (1,), 1),
            (np.sin(hue) / 2 + 0.5) * (np.cos(hue) / 2 + 0.5),
            (hsv[...,1:2] / 255) * (hsv[...,2:3] / 255),
            (hsv[...,1:2] / 255) * (hsv[...,1:2] / 255),
            (hsv[...,2:3] / 255) * (hsv[...,2:3] / 255),
            c[...,0:1] / math.pi,
            c[...,1:2] / (2 * math.pi)
        ], axis=-1)

    def correct_bgr(self, bgr, c_global, c_local):
        hsv = cv.cvtColor(bgr.astype(np.uint8), cv.COLOR_BGR2HSV)
        hsv_c = self.correct_hsv(hsv, c_global, c_local).astype(np.uint8)
        return cv.cvtColor(hsv_c, cv.COLOR_HSV2BGR)

class ColorTransformMeaned(ColorTransformArea):
    def __init__(self, image, debug):
        super().__init__(image, debug)
        self._size = 5

    def coord_sizing(self, n):
        return n

    def get_uncorrected_hsv(self, polar):
        rpp = math.pi / self._image.shape[0]
        size = self._size

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
        return average_hsv(all_hsv)[:,:3], polar


class ColorTransformKMeansRegression(ColorTransformArea):
    def __init__(self, image, means, debug):
        super().__init__(image, debug);

        self._num_means = means
        self._debug_w = math.ceil(math.sqrt(3 * self._num_means / 2))
        self._debug_h = math.ceil(self._debug_w * 2 / 3)
        self._k = np.zeros((self._num_means, self._x_size, self._x_size))
        self._k_range = np.zeros((self._num_means, 2, 3))
        self._k_valid = np.full((self._num_means), False)

        self._kmeans_image = None
        self._kmeans_labels = None


    def _kmeans(self):
        if self._kmeans_image is not None:
            return

        I = np.float32(self._image.reshape((-1,3)))

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv.kmeans(I, self._num_means, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        img_meaned = center[label.flatten()].reshape((self._image.shape))

        if self._debug[0].enable('kmeans'):
            self._debug[0].figure('kmeans') \
                .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1) \
                .imshow(cv.cvtColor(img_meaned, cv.COLOR_BGR2RGB))

        self._kmeans_image = img_meaned
        self._kmeans_label = label.flatten().reshape((self._image.shape[:-1])).astype(np.float32)
        self._kmeans_center = cv.cvtColor(center.reshape((center.shape[0], 1, 3)), cv.COLOR_BGR2HSV) \
                                .reshape((center.shape[0], 3))

    def get_target_hsv(self, i, c):
        return super().get_target_hsv(i, c)

    def compute_hsv_coeffs(self, i, f, c):
        self._kmeans()

        c_eqr = coordinates.polar_to_eqr(c, self._image.shape).astype(np.float32)
        labels = coordinates.eqr_interp(c_eqr, self._kmeans_label, cv.INTER_NEAREST)

        if self._debug[0].enable('color_regression_kmeans'):
            self._debug[0].figure('color_regression_kmeans') \
                .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1) \
                .hist(labels, self._num_means, [0, self._num_means])

        if self._debug[0].enable('color_regression'):
            self._debug[0].figure('color_regression_0', True)
            self._debug[0].figure('color_regression_1', True)

        dbg = self._debug
        err = np.zeros((self._num_means, 6))
        for l in range(0, self._num_means):
            inc = labels == l
            inc_c = np.count_nonzero(inc)
            if inc_c == 0: continue

            in_range = np.logical_and(trim_outliers(i[inc], 2.5, 3, 3),
                                      trim_outliers(f[inc], 2.5, 3, 3))
            in_range_c = np.count_nonzero(in_range)

            if dbg[0].verbose and (inc_c - in_range_c) / inc_c > 0.15:
                print('trim exceeds 0.15: ', inc_c - in_range_c, inc_c)

            if in_range_c < 200:
                print('skip kmean:', l, 'too few inputs:', in_range_c)
                continue

            self._debug = (dbg[0], l, self._debug_h, self._debug_w)

            err[l] = super().compute_hsv_coeffs(i[inc][in_range], f[inc][in_range], c[inc][in_range])
            self._k[l] = self._c
            self._k_range[l] = hsv_range(i[inc][in_range], 2.5, 3, 3)
            self._k_valid[l] = True

        if self._debug[0].enable('color_regression_result'):
            fig = self._debug[0].figure('color_regression_result', True)
            for m, l in enumerate(["Hue Median Correction", "Saturation Mean Correction", \
                                   "Value Mean Correction", "Hue Alignment Correction", \
                                   "Saturation Std Dev Correction", "Value Std Dev Correction"]):
                ax = fig.add_subplot(2, 3, m+1)
                ax.title.set_text(l)
                ax.bar(range(0, self._num_means), err[:,m], color=hsv_to_rgb_hex(self._kmeans_center))

        self._debug = dbg

    def correct_hsv(self, hsv, c_global, c_local):
        self._kmeans()
        c_eqr = coordinates.polar_to_eqr_3d(c_local).astype(np.float32)
        labels = coordinates.eqr_interp_3d(c_eqr, self._kmeans_label, cv.INTER_NEAREST)

        if self._debug[0].enable('color_final_kmeans'):
            self._debug[0].figure('color_final_kmeans') \
                .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1) \
                .hist(labels, self._num_means, [0, self._num_means])

        delta = np.zeros(hsv.shape[:-1] + (4,))
        corrected = np.zeros((self._num_means))
        for l in range(0, self._num_means):
            inc = labels == l
            if np.count_nonzero(inc) == 0: continue

            if not self._k_valid[l]:
                continue

            in_range = np.full(hsv.shape[:-1], False)
            in_range[inc] = hsv_in_range(hsv[inc], self._k_range[l])
            corrected[l] = np.count_nonzero(in_range)

            x = self._arrange_hsv(hsv[in_range], c_local[in_range])
            dc = np.clip(x @ self._k[l], 0, 1) - x
            delta[in_range] = dc[:,0:4]

        kernel = np.ones((11,11),np.float32)/(11*11)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug[0].enable('color_delta'):
            norm = (delta + 1) / 2
            f = plt.figure()
            for i in range(4):
                f.add_subplot(2, 2, i + 1).imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))

        if self._debug[0].enable('color_correction'):
            fig = self._debug[0].figure('color_correction')
            fig.add_subplot(2, 4, self._debug[1] + 1) \
               .bar(range(0, self._num_means), corrected, color=hsv_to_rgb_hex(self._kmeans_center))

        return trig_norm_to_hsv(np.clip(hsv_to_trig_norm(hsv) + delta, 0, 1))

class ColorTransitionMeaned(ColorTransformMeaned):
    def __init__(self, image, seam_left, seam_right, debug):
        super().__init__(image, debug);
        self._fade_dist = 10 * math.pi / 180

        self._seam_left = seam_left
        self._seam_right = seam_right

    def _distance_from_seam(self, seam, polar):
        phi = seam[:, 0]
        theta = seam[:, 1]
        slope = (theta[1:] - theta[:-1]) / (phi[1:] - phi[:-1])
        offset = theta[:-1] - slope * phi[:-1]

        seam_theta = np.zeros((polar.shape[0]))
        phi = polar[:,0,0:1]
        phi_n = phi.shape[0]
        slope_n = slope.shape[0]

        f_mat = np.ones((phi_n, slope_n + 1)) * seam[:, 0]
        in_range = np.logical_and(phi < f_mat[:,1:], phi >= f_mat[:,:-1])
        f_slope = (np.ones((phi_n, slope_n)) * slope)[in_range]
        f_offset = (np.ones((phi_n, offset.shape[0])) * offset)[in_range]

        in_range = np.any(in_range, axis=1)
        seam_theta[in_range] = phi[in_range,0] * f_slope + f_offset

        return seam_theta.reshape(polar.shape[0], 1) * np.ones(polar.shape[:-1]) - polar[...,1]


    def correct_hsv(self, hsv, c_global, c_local):
        df = np.zeros(hsv.shape[:-1])
        d_left = self._distance_from_seam(self._seam_left - [0, math.pi/2], c_global)
        d_right = self._distance_from_seam(self._seam_right, c_global)

        df += 1.0 - np.clip(np.absolute(d_left / self._fade_dist), 0, 1)
        df += 1.0 - np.clip(np.absolute(d_right / self._fade_dist), 0, 1)

        x = self._arrange_hsv(hsv, c_local)
        cor = x + df.reshape(df.shape + (1,)) * (np.clip(x @ self._c, 0, 1) - x)

        r = np.zeros(hsv.shape)
        r[...,0] = 180 / (2 * math.pi) * (np.arctan2(cor[...,0] * 2 - 1, cor[...,1] * 2 - 1) % (2 * math.pi))
        r[...,1:3] = np.clip(cor[...,2:4], 0, 1) * 255
        return r

class ColorCorrection():
    def __init__(self, images, config, debug):
        self._images = images
        self._debug = debug
        self._config = config
        self._align_thres = config.color_correction_align[0]
        self._color_thres = config.color_correction_align[1]

    def _match_along_seams(self, matches, seams):
        more_matches = []
        for i in range(4):
            m = matches[i]
            added = np.zeros((seams[2*i].shape[0], m.shape[1]))
            added[:,0:2] = seams[2*i] + [0, math.pi]
            added[:,2:4] = seams[2*i] + [0, math.pi]
            added[:,4:6] = seams[(2*i+1) % 8] + [0, math.pi]
            added[:,6:8] = seams[(2*i+1) % 8] + [0, math.pi]
            mm = np.vstack([m, added])

            S = 5
            for i in range(S):
                mid = i / S * added[:-1] + (S - i) / S * added[1:]
                mm = np.vstack([mm, mid])

            more_matches.append(mm)

        return more_matches

    def match_colors(self, matches, seams):
        transforms = [ColorTransformArea(img, (self._debug, i, 3, 3)) \
                      for i, img in enumerate(self._images)]
        matches = self._match_along_seams(matches, seams)
        return self._regression(matches, transforms)

    def match_colors_kmeans(self, matches, seams, means):
        transforms = [ColorTransformKMeansRegression(img, means, (self._debug, i, 3, 3)) \
                      for i, img in enumerate(self._images)]
        matches = self._match_along_seams(matches, seams)
        return self._regression(matches, transforms)

    def fade_colors(self, matches, seams, fade):
        transforms = [ColorTransitionMeaned(im, seams[(i-2)%8], seams[i], (self._debug, i, 3, 3)) \
                      for i, im in enumerate(self._images)]
        for t in transforms:
            t._fade_dist = fade
        matches = self._match_along_seams(matches, seams)
        return self._regression(matches, transforms)

    def _close_to_target(self, color, target):
        inc = target[:,3] > self._align_thres
        for i in range(4):
            inc = np.logical_and(inc, np.absolute(target[:,1] - color[:,i,1]) < self._color_thres)
            inc = np.logical_and(inc, np.absolute(target[:,2] - color[:,i,2]) < self._color_thres)
        return inc

    def _regression(self, matches, transforms):
        colors = []
        targets = []
        coords = []

        fig_color = plt.figure() if self._debug.enable('color') else None
        for i, m in enumerate(matches):
            ll = (2*i-2) % 8
            lr = 2*i % 8
            rl = (2*i-1) % 8
            rr = (2*i+1) % 8

            n = transforms[ll].coord_sizing(m.shape[0])
            color = np.zeros((n, 4, 3))
            coord = np.zeros((n, 4, 2))
            target = np.zeros((n, 4, 3))
            for j, idx in enumerate([ll, lr, rl, rr]):
                shift = [0, math.pi / 2] if j % 2 == 1 else [0, 0]
                color[:,j], coord[:,j] = transforms[idx].get_uncorrected_hsv(m[:,2*j:2*j+2] - shift)
                target[:,j] = transforms[idx].get_target_hsv(color[:,j], coord[:,j])

            target_avg = average_hsv(target)
            inc = self._close_to_target(color, target_avg)

            targets.append(target_avg[inc,:3])
            colors.append(color[inc])
            coords.append(coord[inc])

            if self._debug.enable('color'):
                h = 600
                t = target_avg[inc,:3].reshape(target_avg[inc].shape[0], 1, 3)
                black = np.full((t.shape), 0)
                img = np.concatenate([color[inc], target[inc], t, black], axis=1)
                img = img[img[:,8,0].argsort()]
                img = img[img[:,8,2].argsort(kind='mergesort')]
                w = math.ceil(img.shape[0] / h)
                img = np.concatenate([img, np.full((w*h - img.shape[0], img.shape[1], 3), 0)])
                img = img.reshape((h, w * img.shape[1], 3)).astype(np.uint8)
                fig_color.add_subplot(1, 4, i+1).imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))

        for i in range(8):
            l = int(i/2)
            r = (l+1) % 4
            cl = 2 * (i % 2) + 1 # column within the matched set of 4 images.
            cr = 2 * (i % 2) # column within the matched set of 4 images.
            color = np.concatenate([colors[l][:,cl], colors[r][:,cr]])
            target = np.concatenate([targets[l], targets[r]])
            coord = np.concatenate([coords[l][:,cl], coords[r][:,cr]])

            transforms[i].compute_hsv_coeffs(color, target, coord)

        return transforms
