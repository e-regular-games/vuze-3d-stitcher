#!/usr/bin/python

import coordinates
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

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

        if self._debug[0].verbose:
            print('regression error', np.sum(hsv_error(i, f)) / i.shape[0], \
                  np.sum(hsv_error(y_a, f)) / i.shape[0])

        return hsv_error(y_a, f)

    def correct_hsv(self, hsv, c_global, c_local, flt):
        r = np.zeros(hsv.shape)
        r[flt] = self._correct_hsv(hsv[flt], c_local[flt])
        return r

    def _correct_hsv(self, hsv, c_local):
        x = self._arrange_hsv(hsv, c_local)
        cor = np.clip(x @ self._c, 0, 1)
        return trig_norm_to_hsv(cor)

    def _arrange_hsv(self, hsv, c):
        hue = hsv[...,0:1] * math.pi / 180.0 * 2.0
        means = self.get_uncorrected_hsv_area(c)
        return np.concatenate([
            hsv_to_trig_norm(hsv),
            np.full(hsv.shape[:-1] + (1,), 1),
            (np.sin(hue) / 2 + 0.5) * (np.cos(hue) / 2 + 0.5),
            hsv_to_trig_norm(means)
        ], axis=-1)

    def correct_bgr(self, bgr, c_global, c_local, flt):
        hsv = cv.cvtColor(bgr.astype(np.uint8), cv.COLOR_BGR2HSV)
        hsv_c = self.correct_hsv(hsv, c_global, c_local, flt).astype(np.uint8)
        return cv.cvtColor(hsv_c, cv.COLOR_HSV2BGR)


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

        self._transforms = [ColorTransform() for t in range(means)]
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
            _, _, bars = self._debug[0] \
                             .figure('color_regression_kmeans') \
                             .add_subplot(self._debug[2], self._debug[3], self._debug[1] + 1) \
                             .hist(labels, self._num_means, [0, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                b.set_facecolor(colors[j])

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

            if in_range_c < 100:
                print('skip kmean:', l, 'too few inputs:', in_range_c)
                continue

            self._debug = (dbg[0], l, self._debug_h, self._debug_w)

            err[l] = self._transforms[l] \
                         .compute_hsv_coeffs(i[inc][in_range], f[inc][in_range], c[inc][in_range])
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
    def __init__(self, image, config, debug):
        super().__init__(3, image, debug);

        self._config = config
        self._num_means = 0
        self._debug_w = None
        self._debug_h = None
        self._k = None
        self._k_range = None
        self._k_valid = None

        self._kmeans_center = None
        self._kmeans = None
        self._temp_transform = ColorTransformAreaMeaned(3, image, (debug[0].cleared(),))

    def _kmeans_n(self, i, n):
        I = hsv_to_trig_norm(i)
        kmeans = KMeans(n_clusters=n, random_state=0).fit(I)
        labels = kmeans.labels_
        center = trig_norm_to_hsv(kmeans.cluster_centers_)
        return labels, center, kmeans

    def _coeffs_for_label(self, i, f, c, flt, transform, center):
        flt_c = np.count_nonzero(flt)
        if flt_c < 100:
            return np.sum(hsv_error(i[flt], f[flt])), False, 0, 0

        err = transform.compute_hsv_coeffs(i[flt], f[flt], c[flt])

        d = hsv_to_trig_norm(i[flt]) - hsv_to_trig_norm(center)
        dist = np.sqrt(np.sum(d * d, axis=-1))
        lim_d = bottom_nth(dist, 0.9)

        area = self.get_uncorrected_hsv_area(c[flt])
        da = hsv_to_trig_norm(area) - hsv_to_trig_norm(i[flt])
        dist_a = np.sqrt(np.sum(da * da, axis=-1))
        lim_da = bottom_nth(dist_a, 0.9)

        return np.sum(err), True, lim_d, lim_da

    def compute_hsv_coeffs(self, i, f, c):
        if self._debug[0].verbose:
            print('original error', np.sum(hsv_error(i, f)) / i.shape[0])

        best = None
        n = 4
        while best is None or best[0] > n - 8:
            labels, centers, kmeans = self._kmeans_n(i, n)

            errs = np.zeros((n))
            k = np.zeros((n, self._x_size, self._x_size))
            valid = np.full((n), False)
            rg = np.zeros((n, 2))

            for l in range(0, n):
                inc = labels == l
                errs[l], valid[l], rg[l,0], rg[l,1] = self._coeffs_for_label(i, f, c, inc, self._temp_transform, centers[l])
                k[l] = self._temp_transform._c

            if best is None or best[1] > np.sum(errs) / i.shape[0]:
                best = (n, np.sum(errs) / i.shape[0], labels)
                self._kmeans_center = centers
                self._k = k
                self._k_valid = valid
                self._kmeans = kmeans
                self._k_range = rg
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

        labels[flt] = self._kmeans.predict(tnorm[flt])
        diff = tnorm[flt] - hsv_to_trig_norm(self._kmeans_center)[labels[flt]]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        outlier = np.full(hsv.shape[:-1], False)
        outlier[flt] = dist >= self._k_range[labels[flt],0]
        labels[outlier] = -1

        if self._debug[0].enable('color_correction'):
            _, _, bars = self._debug[0] \
                             .figure('color_correction_raw') \
                             .add_subplot(2, 4, self._debug[1] + 1) \
                             .hist(labels[flt], self._num_means + 1, [-1, self._num_means])
            colors = hsv_to_rgb_hex(self._kmeans_center)
            for j, b in enumerate(bars):
                if j == 0: continue # first bar is the non-labeled
                b.set_facecolor(colors[j-1])

        delta = np.zeros(hsv.shape[:-1] + (4,))
        corrected = np.zeros((self._num_means))
        for l in range(0, self._num_means):
            if not self._k_valid[l]:
                continue

            in_range = labels == l
            if np.count_nonzero(in_range) == 0:
                continue

            area = self.get_uncorrected_hsv_area(c_local[in_range])
            da = hsv_to_trig_norm(area) - tnorm[in_range]
            dist_a = np.sqrt(np.sum(da * da, axis=-1))
            in_range[in_range] = dist_a < self._k_range[l,1]

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
            #delta[in_range] = weight.reshape((dc.shape[0], 1)) * dc[:,0:4]
            delta[in_range] = dc[:,0:4]

        kernel = np.ones((11,11),np.float32)/(11*11)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug[0].enable('color_delta'):
            norm = (delta + 1) / 2
            f = plt.figure()
            for i in range(4):
                f.add_subplot(2, 2, i + 1).imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))


        return trig_norm_to_hsv(np.clip(hsv_to_trig_norm(hsv) + delta, 0, 1))

class ColorTransitionMeaned(ColorTransformMeaned):
    def __init__(self, image, seam_left, seam_right, debug):
        super().__init__(5, image, debug);
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
        transforms = [ColorTransformKMeansReducedRegression(img, self._config, (self._debug, i, 3, 3)) \
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
            if kept > 0.3 * labels.shape[0]:
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
