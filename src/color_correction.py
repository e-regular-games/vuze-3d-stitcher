#!/usr/bin/python

import coordinates
import cv2 as cv
from datetime import datetime
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.spatial import KDTree
from skimage.exposure import match_histograms

class StopWatch():
    def __init__(self, name):
        self._a = datetime.now()
        self._name = name

    def stop(self):
        print(self._name, datetime.now() - self._a)

# opencv uses HSV ranges: (0, 179), (0, 255), (0, 255)
def normal_hsv(hsv):
    return hsv / [179, 255, 255]

def normal_bgr(bgr):
    return bgr / [255, 255, 255]

# a is assumed to be a minimum of 2 dimensions
def to_1d(a):
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])

def to_2d(a):
    return a.reshape((a.shape[0],) + (1,) + a.shape[1:])

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

def hsv_error(i, f):
    it = hsv_to_trig_norm(i)
    ft = hsv_to_trig_norm(f)

    dt = ft - it
    return np.sqrt(np.sum(dt*dt, axis=-1))

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

# expects: img as a 3d matrix with height, width, channels
# expects: img_hist, target_hist as a 2d lists of pixels with value by channels
def match_hist(img, img_hist, target_hist):
    n = img.shape[0] * img.shape[1]
    target_channels = [np.zeros((n, 1)) for i in range(3)]
    for c in range(3):
        s = 0
        for i in range(256):
            cnt = int(target_hist[i,c] * n)
            target_channels[c][s:s+cnt,0] = np.full((cnt), i)
            s += cnt

    target = np.concatenate(target_channels, axis=-1).reshape(img.shape)
    return match_histograms(img, target, channel_axis=-1)


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

# return a value such that 100*r% of values in array a are less than it.
def bottom_nth(a, r):
    return np.sort(a)[int(r * a.shape[0])]

def bottom_mth(a, r):
    return np.median(a) + r * np.std(a)

def middle_nth(a, r):
    n = a.shape[0]
    s = np.sort(a)
    l = s[int((0.5 - r/2) * n)]
    u = s[int((0.5 + r/2) * n)]
    return np.logical_and(l < s, s < u)

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

class ColorTransform():
    def __init__(self, image, debug):
        self._image = image

        self._debug = debug
        self._x_size = 10
        self._c = np.zeros((self._x_size, self._x_size))
        self._range = None
        self._range_max_density = None

        reducer = 20
        dims = (int(image.shape[1] / reducer), int(image.shape[0] / reducer))
        self._image_sm = cv.resize(image, dims, interpolation=cv.INTER_AREA)

    def coord_sizing(self, n):
        return n

    # @returns a tuple with (hsv, polar), note: the coordinates are returned
    # because derived classes may choose to expand the number of coordinates and pixels
    def get_uncorrected_hsv(self, polar):
        eqr = coordinates.polar_to_eqr(polar, self._image.shape)
        bgr = coordinates.eqr_interp(eqr, self._image)
        hsv = cv.cvtColor(bgr.reshape(bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                    .reshape(bgr.shape[0], 3)
        return hsv, polar

    # @returns the hsv value of the area surrounding each polar point.
    # the size of the area is based on the reducer variable in the constructor.
    def get_uncorrected_hsv_area(self, polar):
        eqr = coordinates.polar_to_eqr(polar, self._image_sm.shape)
        if len(polar.shape) == 3:
            bgr = coordinates.eqr_interp_3d(eqr, self._image_sm, method=cv.INTER_LINEAR)
            return cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        else:
            bgr = coordinates.eqr_interp(eqr, self._image_sm, method=cv.INTER_LINEAR)
            return cv.cvtColor(bgr.reshape(bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                     .reshape(bgr.shape[0], 3)

    def get_target_hsv(self, i, c):
        return i

    def compute_hsv_coeffs(self, i, f, c):
        x = self._arrange_hsv(i, c)
        y = self._arrange_hsv(f, c)

        if self._debug.enable('color_regression'):
            ax = self._debug.subplot('color_regression_before')
            ax.hist(hsv_error(i, f))

        Q, R = np.linalg.qr(x)
        self._c = np.linalg.inv(R).dot(np.transpose(Q)) @ y

        self._range = KDTree(x[:,:4])
        densities = np.array(self._range.query_ball_point(x[:,:4], 0.1, return_length=True))
        self._range_max_density = bottom_nth(densities, 0.4)

        y_a = trig_norm_to_hsv(self._delta_tnorm(x[:,:4], c) + x[:,:4])

        if self._debug.enable('color_regression'):
            ax = self._debug.subplot('color_regression_after')
            ax.hist(hsv_error(y_a, f))

        if self._debug.verbose:
            print('regression error', i.shape[0], \
                  np.sum(hsv_error(i, f)) / i.shape[0], \
                  np.sum(hsv_error(y_a, f)) / i.shape[0])

        return hsv_error(y_a, f)

    def correct_hsv(self, hsv, c_global, c_local, flt):
        return trig_norm_to_hsv(self.correct_tnorm(hsv_to_trig_norm(hsv), c_global, c_local, flt))

    def correct_tnorm(self, tnorm, c_global, c_local, flt):
        delta = self.delta_tnorm(tnorm, c_global, c_local, flt)

        kernel = np.ones((11,11),np.float32)/(11*11)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug.enable('color_delta'):
            norm = (delta + 1) / 2
            f = plt.figure()
            for i in range(4):
                f.add_subplot(2, 2, i + 1).imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))

        return np.clip(tnorm + delta, 0, 1)

    # operates on a matrix of trig norm, global coords, local coords, and
    # a boolean filter of which elements to calculate.
    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        delta = np.zeros(tnorm.shape)
        delta[flt] = self._delta_tnorm(tnorm[flt], c_local[flt])
        return delta

    # operates on a list of trig norm and coordinates.
    # will use the density of the initial regression at that color point
    # to decrease the influence of the regression on the estimated delta.
    # @returns a list of changes to trig norm values.
    def _delta_tnorm(self, tnorm, c_local):
        x = self._arrange_tnorm(tnorm, c_local)
        density = np.array(self._range.query_ball_point(tnorm, 0.1, return_length=True))
        density = density.reshape(density.shape + (1,)) / self._range_max_density
        density = np.clip((density - 0.05) / 0.95, 0, 1)
        return density * ((x @ self._c)[:,:4] - tnorm)

    def _arrange_hsv(self, hsv, c):
        return self._arrange_tnorm(hsv_to_trig_norm(hsv), c)

    def _arrange_tnorm(self, tnorm, c):
        means = self.get_uncorrected_hsv_area(c)
        return np.concatenate([
            tnorm,
            np.full(tnorm.shape[:-1] + (1,), 1),
            tnorm[...,0:1] * tnorm[...,1:2],
            hsv_to_trig_norm(means)
        ], axis=-1)

    def correct_bgr(self, bgr, c_global, c_local, flt):
        hsv = cv.cvtColor(bgr.astype(np.uint8), cv.COLOR_BGR2HSV)
        hsv = self.correct_hsv(hsv, c_global, c_local, flt).astype(np.uint8)
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

class ColorTransformArea(ColorTransform):
    def __init__(self, size, image, debug):
        super().__init__(image, debug)
        self._size = size

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


class ColorTransformMeaned(ColorTransform):
    def __init__(self, size, image, debug):
        super().__init__(image, debug)
        self._size = size

    def get_uncorrected_hsv(self, polar):
        rpp = math.pi / self._image.shape[0]
        size = self._size

        mean_tnorm = np.zeros(polar.shape[:-1] + (4,))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp
                pts = polar + [dphi, dtheta]
                eqr = coordinates.polar_to_eqr(pts, self._image.shape)
                bgr = coordinates.eqr_interp(eqr, self._image)
                hsv = cv.cvtColor(bgr.reshape(bgr.shape[0], 1, 3), cv.COLOR_BGR2HSV) \
                        .reshape(bgr.shape[0], 3)
                mean_tnorm += (1/(size*size)) * hsv_to_trig_norm(hsv)
        return trig_norm_to_hsv(mean_tnorm), polar

class ColorTransformAreaMeaned(ColorTransform):
    def __init__(self, size, image, debug):
        super().__init__(image, debug)
        self._size = size
        self._meaned = ColorTransformMeaned(size, image, debug)

    def coord_sizing(self, n):
        return n * self._size * self._size

    def get_uncorrected_hsv(self, polar):
        rpp = math.pi / self._image.shape[0]
        size = self._size

        all_polar = np.zeros((0, 2))
        for i in range(size):
            dphi = (i-(size-1)/2) * rpp * size
            for j in range(size):
                dtheta = (j-(size-1)/2) * rpp * size
                all_polar = np.concatenate([all_polar, polar + [dphi, dtheta]])

        return self._meaned.get_uncorrected_hsv(all_polar)

class ColorTransformKMeansRegression(ColorTransform):
    def __init__(self, means, image, debug):
        super().__init__(image, debug);

        self._num_means = means
        self._debug_w = math.ceil(math.sqrt(3 * self._num_means / 2))
        self._debug_h = math.ceil(self._debug_w * 2 / 3)

        self._transforms = [ColorTransform(image, debug) for t in range(means)]
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

        if self._debug.enable('kmeans'):
            self._debug.subplot('kmeans') \
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

        if self._debug.enable('color_regression_kmeans'):
            _, _, bars = self._debug.subplot('color_regression_kmeans') \
                             .hist(labels, self._num_means, [0, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                b.set_facecolor(colors[j])

        if self._debug.enable('color_regression'):
            self._debug.figure('color_regression_0', True)
            self._debug.figure('color_regression_1', True)

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

            if in_range_c < 100:
                print('skip kmean:', l, 'too few inputs:', in_range_c)
                continue

            self._debug = (dbg[0], l, self._debug_h, self._debug_w)

            errs = self._transforms[l] \
                       .compute_hsv_coeffs(i[inc][in_range], f[inc][in_range], c[inc][in_range])
            err[l] = np.mean(errs)
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

    def correct_hsv(self, hsv, c_global, c_local, flt):
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
            dc = np.clip(x @ self._transforms[l]._c, 0, 1) - x
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

class ColorTransformKMeansReducedRegression(ColorTransformAreaMeaned):
    def __init__(self, image, debug):
        super().__init__(3, image, debug);

        self._num_means = 0
        self._debug_w = None
        self._debug_h = None
        self._k = None
        self._k_range = None
        self._k_valid = None
        self._k_range_den = None

        self._kmeans_center = None
        self._kmeans = None
        self._temp_transform = ColorTransformAreaMeaned(3, image, self._debug.none())

    def _kmeans_n(self, i, n):
        I = hsv_to_trig_norm(i)
        kmeans = KMeans(n_clusters=n, random_state=0).fit(I)
        labels = kmeans.labels_
        center = trig_norm_to_hsv(kmeans.cluster_centers_)
        return labels, center, kmeans

    def _coeffs_for_label(self, i, f, c, flt, transform, center):
        flt_c = np.count_nonzero(flt)
        if flt_c < 100:
            return np.sum(hsv_error(i[flt], f[flt])), False, None, 0

        err = transform.compute_hsv_coeffs(i[flt], f[flt], c[flt])

        tnorm = hsv_to_trig_norm(i[flt])
        kdt = KDTree(tnorm)
        knn = np.array(kdt.query_ball_point(tnorm, 0.1, return_length=True))
        return np.sum(err), True, kdt, np.max(knn)

    def compute_hsv_coeffs(self, i, f, c):
        if self._debug[0].verbose:
            print('original error', np.sum(hsv_error(i, f)) / i.shape[0])

        best = None
        n = 4
        while best is None or best[0] > n - 5:
            labels, centers, kmeans = self._kmeans_n(i, n)

            errs = np.zeros((n))
            k = np.zeros((n, self._x_size, self._x_size))
            valid = np.full((n), False)
            rg = [None] * n
            rg_den = [0] * n

            for l in range(0, n):
                inc = labels == l
                errs[l], valid[l], rg[l], rg_den[l] = self._coeffs_for_label(i, f, c, inc, self._temp_transform, centers[l])
                k[l] = self._temp_transform._c

            if best is None or best[1] > np.sum(errs) / i.shape[0]:
                best = (n, np.sum(errs) / i.shape[0], labels)
                self._kmeans_center = centers
                self._k = k
                self._k_valid = valid
                self._kmeans = kmeans
                self._k_range = rg
                self._k_range_den = rg_den
            n += 1

        if self._debug[0].verbose:
            print('best', best[0:2])

        self._num_means = best[0]
        labels = best[2]
        self._debug_w = math.ceil(math.sqrt(3 * self._num_means / 2))
        self._debug_h = math.ceil(self._debug_w * 2 / 3)

        if self._debug[0].enable('color_regression_kmeans'):
            _, _, bars = self._debug[0] \
                             .figure('color_regression_kmeans') \
                             .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1) \
                             .hist(labels, self._num_means, [0, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                b.set_facecolor(colors[j])

        return best[1] * i.shape[0]

    def correct_hsv(self, hsv, c_global, c_local, flt):
        avg_5 = np.ones((5,5),np.float32)/(5*5)
        tnorm = hsv_to_trig_norm(cv.filter2D(hsv, -1, avg_5))

        labels = np.full(hsv.shape[:-1], -1) # use -1 because labels can be 0
        weight = np.zeros(hsv.shape[:-1])
        labels[flt] = self._kmeans.predict(tnorm[flt])
        for l in range(self._num_means):
            flt_l = labels == l
            out_range = np.full(labels.shape, False)
            if self._k_range[l] is not None:
                density = np.array(self._k_range[l].query_ball_point(tnorm[flt_l], 0.1, return_length=True))
                density = density / self._k_range_den[l]
                density[density > 1] = 1
                weight[flt_l] = density
                out_range[flt_l] = density < 0.1
            labels[out_range] = -1

        if self._debug[0].enable('color_correction'):
            _, _, bars = self._debug[0] \
                             .figure('color_correction_raw') \
                             .add_subplot(2, 4, self._debug[1] + 1) \
                             .hist(labels[flt], self._num_means + 1, [-1, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                if j == 0: continue # first bar is the non-labeled
                b.set_facecolor(colors[j-1])

        if self._debug[0].enable('color_correction'):
            self._debug[0].figure('color_correction_weight') \
                          .add_subplot(2, 4, self._debug[1] + 1) \
                          .imshow(weight.reshape(weight.shape + (1,)) * np.ones((1, 1, 3)))

        delta = np.zeros(hsv.shape[:-1] + (4,))
        corrected = np.zeros((self._num_means))
        for l in range(self._num_means):
            if not self._k_valid[l]:
                continue

            in_range = labels == l
            if np.count_nonzero(in_range) == 0:
                continue

            #dt = hsv_to_trig_norm(hsv[in_range]) - hsv_to_trig_norm(self._kmeans_center[l])
            #dt = np.sqrt(np.sum(dt * dt, axis=-1))
            # let's use a cubic equation to model the amount of change in the color for a pixel
            # base the equation on the distance of the color from the center of the mean
            # w = a/3x^3 - a/2rx^2 + 1
            #r = self._k_range[l,0]
            #a = 6 / (r * r * r)
            #weight = a * np.power(dt, 3) / 3 - a * r * np.power(dt, 2) / 2 + 1

            x = self._arrange_hsv(trig_norm_to_hsv(tnorm[in_range]), c_local[in_range])
            dc = np.clip(x @ self._k[l], 0, 1) - x
            delta[in_range] = weight.reshape(weight.shape + (1,))[in_range] * dc[:,0:4]
            #delta[in_range] = dc[:,0:4]

        kernel = np.ones((11,11),np.float32)/(11*11)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug[0].enable('color_delta'):
            norm = (delta + 1) / 2
            f = plt.figure()
            for i in range(4):
                f.add_subplot(2, 2, i + 1).imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))


        return trig_norm_to_hsv(np.clip(hsv_to_trig_norm(hsv) + delta, 0, 1))

class ColorTransformKMeansDynamic(ColorTransform):
    def __init__(self, image, debug):
        super().__init__(image, debug);

        self._kmeans_fixed = None

    def compute_hsv_coeffs(self, i, f, c):
        n = 4
        best = None # (num_means, errors, transform)
        while best is None or best[0] > n - 5:
            t = ColorTransformKMeansFixed(n, self._image, self._debug.none())
            errors = t.compute_hsv_coeffs(i, f, c)

            if best is None or np.sum(best[1]) > np.sum(errors):
                best = (n, errors, t)
            n += 1

        # re-compute to get the proper debug output.
        n = best[0]
        print('kmeans dynamic', n)
        self._kmeans_fixed = ColorTransformKMeansFixed(n, self._image, self._debug)
        return self._kmeans_fixed.compute_hsv_coefffs(i, f, c)

    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        return self._kmeans_fixed.delta_tnorm(tnorm, c_global, c_local, flt)

class ColorTransformKMeansFixed(ColorTransform):
    def __init__(self, num_means, image, debug):
        super().__init__(image, debug);

        self._num_means = num_means

        dbg_w = math.ceil(math.sqrt(3 * num_means / 2))
        dbg_h = math.ceil(dbg_w * 2 / 3)
        self._transforms = [ColorTransform(image, debug.set_subplot(dbg_h, dbg_w, i+1)) \
                            for i in range(num_means)]

        self._kmeans_center = None
        self._kmeans = None

    def compute_hsv_coeffs(self, i, f, c):
        n = self._num_means
        tnorm = hsv_to_trig_norm(i)
        self._kmeans = KMeans(n_clusters=n, random_state=0).fit(tnorm)
        self._kmeans_center = trig_norm_to_hsv(self._kmeans.cluster_centers_)
        labels = self._kmeans.labels_

        if self._debug.enable('color_regression_kmeans'):
            _, _, bars = self._debug.subplot('color_regression_kmeans') \
                             .hist(labels, n, [0, n])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                b.set_facecolor(colors[j])

        errors = np.zeros(i.shape[:-1])
        for l in range(n):
            flt = labels == l
            if np.count_nonzero(flt) > 100:
                errors[flt] = self._transforms[l].compute_hsv_coeffs(i[flt], f[flt], c[flt])
            else:
                self._transforms[l] = None
                errors[flt] = hsv_error(i[flt], f[flt])

        if self._debug.verbose:
            print('regression error kmeans', i.shape[0], \
                  np.sum(hsv_error(i, f)) / i.shape[0], \
                  np.sum(errors) / i.shape[0])

        return errors

    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        labels = np.full(tnorm.shape[:-1], -1) # use -1 because labels can be 0
        labels[flt] = self._kmeans.predict(tnorm[flt])

        if self._debug.enable('color_correction'):
            _, _, bars = self._debug \
                             .figure('color_correction_raw') \
                             .hist(labels[flt], self._num_means + 1, [-1, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                if j == 0: continue # first bar is the non-labeled
                b.set_facecolor(colors[j-1])

        delta = np.zeros(tnorm.shape)
        for l in range(self._num_means):
            if self._transforms[l] is None:
                continue

            in_range = labels == l
            if np.count_nonzero(in_range) == 0:
                continue

            delta += self._transforms[l].delta_tnorm(tnorm, c_global, c_local, in_range)

        return delta

class ColorTransformSided(ColorTransform):
    def __init__(self, image, debug):
        super().__init__(image, debug)

        self._left = ColorTransformKMeansFixed(10, image, debug.set_subplot(1, 2, 1))
        self._right = ColorTransformKMeansFixed(10, image, debug.set_subplot(1, 2, 2))

    def compute_hsv_coeffs(self, i, f, c):
        left = c[...,1] <= math.pi
        right = c[...,1] > math.pi

        errors = np.zeros(i.shape[:-1])
        errors[left] = self._left.compute_hsv_coeffs(i[left], f[left], c[left])
        errors[right] = self._right.compute_hsv_coeffs(i[right], f[right], c[right])

        if self._debug.verbose:
            print('regression error sided', i.shape[0], \
                  np.sum(hsv_error(i, f)) / i.shape[0], \
                  np.sum(errors) / i.shape[0])

        return errors

    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        left_delta = self._left.delta_tnorm(tnorm, c_global, c_local, flt)
        right_delta = self._right.delta_tnorm(tnorm, c_global, c_local, flt)

        right_weight = np.zeros(flt.shape + (1,))
        right_weight[flt,0] = np.clip((c_local[flt,1] - 3*math.pi/4) / (math.pi/2), 0, 1)
        left_weight = np.zeros(flt.shape + (1,))
        left_weight[flt,0] = 1.0 - right_weight[flt,0]

        return left_weight * left_delta + right_weight * right_delta

class ColorCorrection():
    def __init__(self, images, config, debug):
        self._images = images
        self._debug = debug

    def _match_along_seams(self, matches, seams):
        def create_matches(seam_left, seam_right, theta_offset):
            bnd = math.pi / 6
            delta = 40

            phi_l = np.linspace(bnd/(delta-1), bnd, delta-1)
            phi_m = np.linspace(bnd, math.pi-bnd, delta)
            phi_u = np.linspace(math.pi-bnd, math.pi-bnd/(delta-1), delta-1)
            phi = np.concatenate([phi_l, phi_m, phi_u])

            intersect_left = coordinates.seam_intersect(seam_left, phi) + theta_offset
            intersect_right = coordinates.seam_intersect(seam_right, phi) + theta_offset

            added = np.zeros((phi.shape[0], 8))
            added[:,0] = phi
            added[:,1] = intersect_left
            added[:,2] = phi
            added[:,3] = intersect_left
            added[:,4] = phi
            added[:,5] = intersect_right
            added[:,6] = phi
            added[:,7] = intersect_right
            return added

        more_matches = []
        for i in range(4):
            mm = matches[i]
            added = create_matches(seams[2*i] + [0, math.pi], seams[(2*i+1) % 8] + [0, math.pi], 0)
            mm = np.vstack([mm, added])
            added = create_matches(seams[2*i] + [0, math.pi], seams[(2*i+1) % 8] + [0, math.pi], -4 * math.pi / 180)
            mm = np.vstack([mm, added])
            added = create_matches(seams[2*i] + [0, math.pi], seams[(2*i+1) % 8] + [0, math.pi], 4 * math.pi / 180)
            mm = np.vstack([mm, added])
            more_matches.append(mm)

        return more_matches

    def match_colors(self, matches, seams):
        transforms = [ColorTransformArea(5, img, (self._debug, i, 3, 3)) \
                      for i, img in enumerate(self._images)]
        matches = self._match_along_seams(matches, seams)
        return self._regression(matches, transforms)

    def match_colors_kmeans(self, matches, seams):
        transforms = [ColorTransformKMeansReducedRegression(img, (self._debug, i, 3, 3)) \
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

    def _within_tolerance(self, color, area, idx):
        tnorm = hsv_to_trig_norm(color)
        dist = np.zeros((color.shape[0], 6))
        p = 0
        for o in range(3):
            d = tnorm[...,o+1:,:] - tnorm[...,o,:].reshape((tnorm.shape[0], 1, tnorm.shape[-1]))
            dist[:,p:p+3-o] = np.sqrt(np.sum(d * d, axis=-1))
            p += 3-o


        k = 10
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dist)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers_keep = []
        centers_mx = np.max(centers, axis=-1)
        centers_sorted = np.argsort(centers_mx)[:7].astype(np.int)
        kept = 0
        for center in centers_sorted:
            centers_keep.append(center)
            kept += np.count_nonzero(labels == center)
            if kept > 0.5 * labels.shape[0]:
                break

        centers_keep = np.array(centers_keep)

        if self._debug.enable('color'):
            _, _, bars = self._debug.figure('color_dist_kmeans') \
                                    .add_subplot(2, 2, idx+1).hist(labels, k, [0, k])
            for i in centers_keep:
                bars[i].set_facecolor('#00ff80')

        return np.any(labels[..., None] == centers_keep[None, ...], axis=1)



        tnorm_area = hsv_to_trig_norm(area)
        d = tnorm - tnorm_area
        dist = np.sqrt(np.sum(d*d, axis=-1))
        dist = np.mean(dist, axis=-1)
        inc_area = dist < bottom_nth(dist, 0.3)

        return np.logical_and(inc_dist, inc_area)

    def _get_targets(self, target):
        tnorm = hsv_to_trig_norm(target)
        t = np.mean(tnorm, axis=1)
        return trig_norm_to_hsv(t)

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
            area = np.zeros((n, 4, 3))
            coord = np.zeros((n, 4, 2))
            target = np.zeros((n, 4, 3))
            for j, idx in enumerate([ll, lr, rl, rr]):
                shift = [0, math.pi / 2] if j % 2 == 1 else [0, 0]
                color[:,j], coord[:,j] = transforms[idx].get_uncorrected_hsv(m[:,2*j:2*j+2] - shift)
                area[:,j] = transforms[idx].get_uncorrected_hsv_area(coord[:,j])
                target[:,j] = transforms[idx].get_target_hsv(color[:,j], coord[:,j])

            inc = self._within_tolerance(color, area, i)
            target_single = self._get_targets(target)

            if self._debug.verbose:
                print('sanitize color', np.count_nonzero(inc), color.shape[0])

            targets.append(target_single[inc])
            colors.append(color[inc])
            coords.append(coord[inc])

            if self._debug.enable('color'):
                h = 600
                t = target_single[inc].reshape(target_single[inc].shape[0], 1, 3)
                black = np.full((t.shape), 0)
                img = np.concatenate([color[inc], area[inc], t, black], axis=1)
                img = img[np.argsort(img[:,8,2])]
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


class ColorCorrectionRegion():
    def __init__(self, images, transforms, config, debug):
        self._images = images
        self._debug = debug
        self._transforms = transforms

    def _generate_regions(self, seams):
        h = self._images[0].shape[0]
        w = self._images[0].shape[1]
        y = np.arange(0, h)
        phi = y / (h-1) * math.pi
        theta_range = 8 / 180 * math.pi
        x_range = theta_range / (2 * math.pi) * w
        x_delta = np.arange(math.floor(-1 * x_range / 2), math.ceil(x_range / 2))
        theta_delta = x_delta / (w-1) * 2 * math.pi

        regions = []
        coords = []
        for i, seam in enumerate(seams):
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
            f.canvas.set_window_title('color_slices')
            img = np.concatenate(regions, axis=1).astype(np.uint8)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        return regions, coords

    def _regression(self, regions, coords, hists, hist_targets):
        transforms = []
        dbg_color = []
        for i in range(8):
            left_bgr = regions[(2*i+1) % 16]
            left_coord = coords[(2*i+1) % 16]
            left_hist = hists[(2*i+1) % 16]
            left_target_hist = hist_targets[int(i/2)]
            left_target = match_hist(left_bgr, left_hist, left_target_hist)

            right_bgr = regions[(2*i+4) % 16]
            right_coord = coords[(2*i+4) % 16]
            right_hist = hists[(2*i+4) % 16]
            right_target_hist = hist_targets[int(i/2+1)%4]
            right_target = match_hist(right_bgr, right_hist, right_target_hist)

            bgr = np.concatenate([left_bgr, right_bgr])
            hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
            target = np.concatenate([left_target, right_target])
            target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
            coord = np.concatenate([left_coord, right_coord])

            t = ColorTransformSided(self._images[i], self._debug.window('image ' + str(i)))
            pick = np.random.choice(hsv.shape[0] * hsv.shape[1], size=8000, replace=False)
            t.compute_hsv_coeffs(to_1d(hsv)[pick], to_1d(target_hsv)[pick], to_1d(coord)[pick])
            transforms.append(t)

            print('.', end='', flush=True)

            if self._debug.enable('color'):
                black = np.zeros((left_bgr.shape[0], 1, 3))
                dbg_color.append(left_bgr)
                dbg_color.append(black)
                dbg_color.append(left_target)
                dbg_color.append(black)
                dbg_color.append(black)
                dbg_color.append(right_bgr)
                dbg_color.append(black)
                dbg_color.append(right_target)
                dbg_color.append(black)
                dbg_color.append(black)
                dbg_color.append(black)
                dbg_color.append(black)

        if self._debug.enable('color'):
            f = plt.figure()
            f.canvas.set_window_title('color_with_target')
            img = np.concatenate(dbg_color, axis=1).astype(np.uint8)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        return transforms

    def match_colors_kmeans(self, matches, seams):
        return self.match_colors(matches, seams)

    def match_colors(self, matches, seams):
        regions, coords = self._generate_regions(seams)

        hists = []
        for bgr in regions:
            hist = np.zeros((256, 3))
            for c, clr in enumerate(['b', 'g', 'r']):
                hist[:,c] = cv.calcHist([bgr], [c], None, [256], [0,256])[:,0]
            hists.append(hist)

        hist_targets = []
        for i in range(4):
            hist_mean = hists[4*i] + hists[4*i+1] + hists[4*i+2] + hists[4*i+3]
            hist_mean /= np.sum(hist_mean, axis=0)
            hist_targets.append(hist_mean)

        if self._debug.enable('color_histogram'):
            f, axs = plt.subplots(4, 5)
            f.canvas.set_window_title('color_histogram')
            for c, clr in enumerate(['b', 'g', 'r']):
                for i in range(4):
                    for j in range(4):
                        axs[i, j].plot(hists[4*i+j][:,c], color=clr)
                    axs[i, 4].plot(hist_targets[i][:,c], color=clr)

            axs[0, 0].title.set_text('Left Eye - Left of Seam')
            axs[0, 1].title.set_text('Left Eye - Right of Seam')
            axs[0, 2].title.set_text('Right Eye - Left of Seam')
            axs[0, 3].title.set_text('Right Eye - Right of Seam')
            axs[0, 4].title.set_text('Target')

        return self._regression(regions, coords, hists, hist_targets)
