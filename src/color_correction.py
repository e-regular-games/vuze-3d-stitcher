#!/usr/bin/python

import coordinates
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from skimage.exposure import match_histograms

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
    ], axis=-1).astype(np.float32)

def trig_norm_to_hsv(t):
    hue = 180 / (2 * math.pi) * (np.arctan2(t[...,0:1] * 2 - 1, t[...,1:2] * 2 - 1) % (2 * math.pi))
    return np.concatenate([
        hue,
        t[...,2:3] * 255,
        t[...,3:4] * 255
    ], axis=-1).astype(np.uint8)

def hsv_error(i, f):
    it = hsv_to_trig_norm(i)
    ft = hsv_to_trig_norm(f)

    dt = ft - it
    return np.sqrt(np.sum(dt*dt, axis=-1)).astype(np.float32)

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

# return a value such that 100*r% of values in array a are less than it.
def bottom_nth(a, r):
    return np.sort(a)[int(r * a.shape[0])]

def hsv_to_rgb_hex(hsv):
    n = hsv.shape[0]
    rgb = cv.cvtColor(hsv.reshape((n, 1, 3)).astype(np.uint8), cv.COLOR_HSV2RGB).reshape((n, 3))
    rgb = rgb[:,0] * 0x10000 + rgb[:,1] * 0x100 + rgb[:,2]
    strings = np.vectorize(lambda x: np.base_repr(x, 16))(rgb)
    return np.char.add('#', np.char.zfill(strings, 6))

class ColorTransform():
    def __init__(self, image, debug):
        self._image = image
        self._debug = debug

        self._c = None
        self._range = None
        self._range_max_density = None
        self._range_max_dist = None

    def compute_hsv_coeffs(self, i, f, c):
        x = self._arrange_hsv(i, c)
        y = self._arrange_hsv(f, c)

        try:
            Q, R = np.linalg.qr(x)
            self._c = np.linalg.inv(R).dot(np.transpose(Q)) @ y
        except:
            print('error computing coefficients')
            return hsv_error(i, f)

        self._range = KDTree(x[:,:4])
        densities = np.array(self._range.query_ball_point(x[:,:4], 0.1, return_length=True))
        self._range_max_density = bottom_nth(densities, 0.4)
        self._range_max_dist = 1.5 * np.max(hsv_error(i, f))

        y_a = trig_norm_to_hsv(self._delta_tnorm(x[:,:4], c) + x[:,:4])

        if self._debug.enable('color_regression'):
            b=10
            err0 = hsv_error(i, f)
            err1 = hsv_error(y_a, f)
            err_max = np.max(err0)
            h0, edges = np.histogram(err0, bins=b, range=(0, err_max))
            h1, _ = np.histogram(err1, bins=b, range=(0, err_max))
            err_center = (edges[:-1] + edges[1:]) / 2
            clr0_center = np.zeros((b, 3))
            clr1_center = np.zeros((b, 3))
            for c in range(0, b):
                in0_bin = np.logical_and(err0 < edges[c+1], err0 >= edges[c])
                in1_bin = np.logical_and(err1 < edges[c+1], err1 >= edges[c])
                if np.count_nonzero(in0_bin) > 0:
                    clr0_center[c] = trig_norm_to_hsv(np.mean(hsv_to_trig_norm(f[in0_bin]), axis=0))
                if np.count_nonzero(in1_bin) > 0:
                    clr1_center[c] = trig_norm_to_hsv(np.mean(hsv_to_trig_norm(f[in1_bin]), axis=0))
            self._debug.set_subplot(2, 1, 1) \
                       .subplot('color_regression') \
                       .bar(err_center, h0, color=hsv_to_rgb_hex(clr0_center), width=err_max/10)
            self._debug.set_subplot(2, 1, 2) \
                       .subplot('color_regression') \
                       .bar(err_center, h1, color=hsv_to_rgb_hex(clr1_center), width=err_max/10)

        self._debug.log('regression error', i.shape[0], \
                        np.sum(hsv_error(i, f)) / i.shape[0], \
                        np.sum(hsv_error(y_a, f)) / i.shape[0])

        return hsv_error(y_a, f)

    def correct_hsv(self, hsv, c_global, c_local, flt):
        return trig_norm_to_hsv(self.correct_tnorm(hsv_to_trig_norm(hsv), c_global, c_local, flt))

    def correct_tnorm(self, tnorm, c_global, c_local, flt):
        delta = self.delta_tnorm(tnorm, c_global, c_local, flt)

        kernel = np.ones((11,11), np.float32)/(11*11)
        delta = cv.filter2D(delta, -1, kernel)
        if self._debug.enable('color_delta'):
            norm = (delta + 1) / 2
            self._debug.figure('color_delta', True)
            for i in range(4):
                self._debug.set_subplot(2, 2, i+1) \
                    .subplot('color_delta') \
                    .imshow(norm[...,i:i+1] * np.ones((1, 1, 3)))

        return np.clip(tnorm + delta, 0, 1)

    # operates on a matrix of trig norm, global coords, local coords, and
    # a boolean filter of which elements to calculate.
    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        delta = np.zeros(tnorm.shape, np.float32)
        delta[flt] = self._delta_tnorm(tnorm[flt], c_local[flt])
        return delta

    # operates on a list of trig norm and coordinates.
    # will use the density of the initial regression at that color point
    # to decrease the influence of the regression on the estimated delta.
    # @returns a list of changes to trig norm values.
    def _delta_tnorm(self, tnorm, c_local):
        if self._c is None:
            return np.zeros(tnorm.shape, np.float32)
        x = self._arrange_tnorm(tnorm, c_local)
        density = np.array(self._range.query_ball_point(x[:,:4], 0.1, return_length=True))
        density = density.reshape(density.shape + (1,)) / self._range_max_density
        density = np.clip((density - 0.05) / 0.95, 0, 1)

        delta = (x @ self._c)[:,:4] - tnorm
        x = None

        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        out_of_bounds = dist > self._range_max_dist
        out_count = np.count_nonzero(out_of_bounds)
        if out_count > 0:
            delta[out_of_bounds] *= (self._range_max_dist / dist[out_of_bounds]).reshape((out_count, 1)) * np.ones((out_count, 4))

        out_of_bounds = None
        dist = None

        return density * delta

    def _arrange_hsv(self, hsv, c):
        return self._arrange_tnorm(hsv_to_trig_norm(hsv), c)

    def _arrange_tnorm(self, tnorm, c):
        return np.concatenate([
            tnorm,
            np.full(tnorm.shape[:-1] + (1,), 1),
            tnorm[...,0:1] * tnorm[...,1:2]
        ], axis=-1).astype(np.float32)

    def correct_bgr(self, bgr, c_global, c_local, flt):
        hsv = cv.cvtColor(bgr.astype(np.uint8), cv.COLOR_BGR2HSV)
        hsv = self.correct_hsv(hsv, c_global, c_local, flt).astype(np.uint8)
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

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

        self._debug.log('regression error kmeans', i.shape[0], \
                        np.sum(hsv_error(i, f)) / i.shape[0], \
                        np.sum(errors) / i.shape[0])

        return errors

    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        labels = np.full(tnorm.shape[:-1], -1) # use -1 because labels can be 0
        labels[flt] = self._kmeans.predict(tnorm[flt])

        if self._debug.enable('color_correction'):
            _, _, bars = self._debug \
                             .subplot('color_correction_raw') \
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

        self._debug.log('regression error sided', i.shape[0], \
                        np.sum(hsv_error(i, f)) / i.shape[0], \
                        np.sum(errors) / i.shape[0])

        return errors

    def cubic_fade(self, x, r):
        x = np.clip(x, 0, r)
        a = 6 / (r * r * r)
        return  a * r * np.power(x, 2) / 2 - a * np.power(x, 3) / 3

    def delta_tnorm(self, tnorm, c_global, c_local, flt):
        left_delta = self._left.delta_tnorm(tnorm, c_global, c_local, flt)
        right_delta = self._right.delta_tnorm(tnorm, c_global, c_local, flt)

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

        hists = []
        for bgr in regions:
            hist = np.zeros((256, 3))
            for c, clr in enumerate(['b', 'g', 'r']):
                hist[:,c] = cv.calcHist([bgr], [c], None, [256], [0,256])[:,0]
            hists.append(hist)

        hist_targets = []
        for i in range(4):
            # todo: should this use trig_norm
            region_mean = 0.25 * regions[4*i] + 0.25 * regions[4*i+1] \
                + 0.25 * regions[4*i+2] + 0.25 * regions[4*i+3]
            hist_mean = np.zeros((256, 3))
            bgr = region_mean.astype(np.uint8)
            for c, clr in enumerate(['b', 'g', 'r']):
                hist_mean[:,c] = cv.calcHist([bgr], [c], None, [256], [0,256])[:,0]
            hist_targets.append(hist_mean / np.sum(hist_mean, axis=0))

        if self._debug.enable('color_histogram'):
            f, axs = plt.subplots(4, 5)
            f.canvas.manager.set_window_title('color_histogram')
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

    def _regression(self, regions, coords, hists, hist_targets):
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
            left_hist = hists[(2*i+1) % 16]
            left_target_hist = hist_targets[int(i/2)]
            left_target = match_hist(left_bgr, left_hist, left_target_hist)

            right_bgr = regions[(2*i+4) % 16].astype(np.uint8)
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
            np.random.seed(0)
            pick = np.random.choice(hsv.shape[0] * hsv.shape[1], size=8000, replace=False)
            t.compute_hsv_coeffs(to_1d(hsv)[pick], to_1d(target_hsv)[pick], to_1d(coord)[pick])
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


class ColorCorrectionMatches(ColorCorrection):
    def __init__(self, images, matches, debug):
        super().__init__(images, debug)
        self._matches = matches
        self._region_size = 21

    def _expand_coord(self, plr, wr):
        rpp = math.pi / self._images[0].shape[0]
        size = self._region_size

        w = wr * size
        h = math.ceil(plr.shape[0] / wr) * size
        all_polar = np.zeros((h, w, 2))

        d = np.arange(size)
        dtheta, dphi = np.meshgrid(d, d)
        dphi = (dphi - (size-1)/2) * rpp
        dtheta = (dtheta - (size-1)/2) * rpp

        block = np.zeros((size, size, 2))
        block[:,:,0] = dphi
        block[:,:,1] = dtheta

        for i in range(plr.shape[0]):
            l = (i % 3) * size
            r = ((i % 3) + 1) * size
            t = math.floor(i / 3) * size
            b = math.floor(i / 3 + 1) * size
            all_polar[t:b,l:r] = plr[i] + block

        return all_polar

    def _generate_regions(self):
        regions = []
        coords = []

        for i, m in enumerate(self._matches):
            ll = (2*i-2) % 8
            lr = 2*i % 8
            rl = (2*i-1) % 8
            rr = (2*i+1) % 8

            for j, idx in enumerate([ll, lr, rl, rr]):
                shift = [0, math.pi / 2] if j % 2 == 1 else [0, 0]
                coord = self._expand_coord(m[:,2*j:2*j+2], 3) - shift
                coords.append(coord)

                eqr = coordinates.polar_to_eqr(coord, self._images[idx].shape)
                bgr = coordinates.eqr_interp_3d(eqr.astype(np.float32), self._images[idx]) \
                    .astype(np.uint8)
                regions.append(bgr)

        if self._debug.enable('color'):
            f = plt.figure()
            f.canvas.manager.set_window_title('color_slices')
            for i in range(len(self._matches)):
                img = np.concatenate(regions[4*i:4*i+4], axis=1).astype(np.uint8)
                f.add_subplot(1, 4, i+1).imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        return regions, coords

class ColorCorrectionSeams(ColorCorrection):
    def __init__(self, images, transforms, seams, debug):
        super().__init__(images, debug)
        self._transforms = transforms
        self._seams = seams

    def _generate_regions(self):
        h = self._images[0].shape[0]
        w = self._images[0].shape[1]
        y = np.arange(0, h)
        phi = y / (h-1) * math.pi
        theta_range = 10 / 180 * math.pi
        x_range = theta_range / (2 * math.pi) * w
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
