
import coordinates
import transform
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular

class RefineSeams():
    """
    Refine the seams across all 8 images of the Vuze Camera.
    a, b are equirectangular images with 360 degrees
    and 0deg is the center of the image.
    a is on the left, b is on the right
    the seem is assumed to exist at 45deg in a, and -45deg in b.
    """
    def __init__(self, images, verbose, display):
        self._images = images
        self._transforms = []
        self.verbose = verbose
        self.display = display
        for i, img in enumerate(images):
            t = transform.Transform(verbose, display)
            t.label = 'lens:' + str(i+1)
            self._transforms.append(t)

    def set_verbose(self, v):
        self.verbose = v
        for t in self._transforms:
            t.verbose = v

    def _match_between_eyes(self, imgs_left, imgs_right, threshold):
        sift = cv.SIFT_create()
        all_polar_pts = np.zeros((0, 8))

        imgs = imgs_left + imgs_right
        thetas = [45, -45, 45, -45]
        for lat in [60, 0, -60]:
            rl = [None] * 4
            inv = [None] * 4
            kp = [None] * 4
            des = [None] * 4
            for i in range(4):
                rl[i], inv[i] = Equirectangular(imgs[i]).GetPerspective(60, thetas[i], lat, 1000, 1000);
                gray = cv.cvtColor(rl[i], cv.COLOR_BGR2GRAY)
                kp[i], des[i] = sift.detectAndCompute(gray, None)

            matches = [None]*3
            kp_indices = np.zeros((len(kp[0]), 4), dtype=np.int) - 1
            kp_indices[:, 0] = np.arange(0, len(kp[0]))
            for i in range(1, 4):
                matches = self._determine_matches(des[i], des[0], threshold)
                for m in matches:
                    kp_indices[m.trainIdx, i] = m.queryIdx

            kp_indices = kp_indices[(kp_indices != -1).all(axis=1), :]
            rl_pts = np.zeros((kp_indices.shape[0], 2 * kp_indices.shape[1]))
            for i in range(kp_indices.shape[0]):
                for j in range(kp_indices.shape[1]):
                    rl_pts[i][2*j:2*j+2] = kp[j][kp_indices[i,j]].pt

            lat_polar_pts = np.zeros(rl_pts.shape)
            for j in range(kp_indices.shape[1]):
                eq_pts = inv[j].GetEquirectangularPoints(rl_pts[:,2*j:2*j+2])
                lat_polar_pts[:,2*j:2*j+2] = coordinates.eqr_to_polar(eq_pts, imgs[j].shape)
                lat_polar_pts[:,2*j+1] -= (thetas[j] - thetas[0]) * math.pi / 180

            all_polar_pts = np.concatenate([all_polar_pts, lat_polar_pts])

        return all_polar_pts

    def _determine_matching_points(self, img_a, img_b, theta_a, theta_b, threshold):
        sift = cv.SIFT_create()

        imgs = (img_a, img_b)
        all_polar_pts = np.zeros((0, 4))
        for lat in [60, 0, -60]:
            rl_a, inv_a = Equirectangular(imgs[0]).GetPerspective(60, theta_a, lat, 1000, 1000);
            rl_b, inv_b = Equirectangular(imgs[1]).GetPerspective(60, theta_b, lat, 1000, 1000);
            gray_a = cv.cvtColor(rl_a, cv.COLOR_BGR2GRAY)
            gray_b = cv.cvtColor(rl_b, cv.COLOR_BGR2GRAY)
            kp_a, des_a = sift.detectAndCompute(gray_a, None)
            kp_b, des_b = sift.detectAndCompute(gray_b, None)
            inv = (inv_a, inv_b)

            matches = self._determine_matches(des_b, des_a, threshold)

            if len(matches) == 0:
                continue

            rl_pts = np.zeros((len(matches), 4))
            for i, m in enumerate(matches):
                rl_pts[i, 0:2] = kp_a[m.trainIdx].pt
                rl_pts[i, 2:4] = kp_b[m.queryIdx].pt

            lat_polar_pts = np.zeros(rl_pts.shape)
            for i in range(2):
                eq_pts = inv[i].GetEquirectangularPoints(rl_pts[:,2*i:2*i+2])
                lat_polar_pts[:,2*i:2*i+2] = coordinates.eqr_to_polar(eq_pts, imgs[i].shape)
            all_polar_pts = np.concatenate([all_polar_pts, lat_polar_pts])

        # adjust for the difference in theta's
        all_polar_pts[:, 3] -= (theta_b - theta_a) * math.pi / 180
        return all_polar_pts

    def _determine_matches(self, des_a, des_b, threshold):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des_a, des_b, k=2)

        def by_distance(e):
            return e.distance

        good_matches = []
        for m,n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        good_matches.sort(key=by_distance)
        return good_matches


    def show_polar_plot(self, polar_a, polar_b):
        plt.figure()
        xa = np.sin(polar_a[..., 0]) * np.cos(polar_a[..., 1])
        ya = np.sin(polar_a[..., 0]) * np.sin(polar_a[..., 1])
        za = np.cos(polar_a[..., 0])

        ax = plt.axes(projection ='3d')
        ax.plot3D(xa, ya, za, 'bo')

        xb = np.sin(polar_b[..., 0]) * np.cos(polar_b[..., 1])
        yb = np.sin(polar_b[..., 0]) * np.sin(polar_b[..., 1])
        zb = np.cos(polar_b[..., 0])

        ax.plot3D(xb, yb, zb, 'ro')

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim(-1, 1)

    # idx is the index of the stitch line: 0:4
    # returns line_left, line_right as np.arrays of phi, theta coordinates
    def compute_eye_stitch_line(self, ts_left, ts_right, idx):
        ll = 2 * idx
        lr = (ll + 2) % 8
        rl = ll + 1
        rr = lr + 1

        img_left = [self._images[ll], self._images[lr]]
        img_right = [self._images[rl], self._images[rr]]

        # matches = numpy.array([ll, lr, rl, rr])
        matches = self._match_between_eyes(img_left, img_right, 0.75)

        # reject outliers
        keep = np.full((matches.shape[0]), True)
        for m in range(2, 8, 2):
            diff = matches[:,m:m+2] - matches[:,0:2]
            s = np.std(diff, axis=0)
            mn = np.median(diff, axis=0)
            accept = [2, 2]
            valid = np.logical_and(diff > (mn - accept * s), diff < (mn + accept * s)).all(axis=1)
            keep = np.logical_and(valid, keep)
        matches = matches[keep]

        left_side = np.array([0, math.pi / 2])
        pts_ll = ts_left.apply(matches[:, 0:2]) - [0, math.pi]
        pts_lr = ts_left.apply(matches[:, 2:4] - left_side) + left_side - [0, math.pi]
        pts_rl = ts_right.apply(matches[:, 4:6]) - [0, math.pi]
        pts_rr = ts_right.apply(matches[:, 6:8] - left_side) + left_side - [0, math.pi]

        def limit_range(l, r, num_std):
            middle = (l + r) / 2
            m_theta = np.median(middle[:,1])
            d_theta = np.std(middle[:,1])
            m_diff = np.mean(l-r, axis=0)
            d_diff = np.std(l-r, axis=0)
            if self.verbose:
                print('limit_range', 'theta_median', m_theta, 'theta_stddev', d_theta, 'diff_mean', m_diff, 'diff_stddev', d_diff)
            return np.logical_and(middle[:, 1] > m_theta - num_std * d_theta, middle[:, 1] < m_theta + num_std * d_theta)

        def sort_middle(l, r):
            middle = (l + r) / 2
            m_theta = np.median(middle[:, 1])
            sort_middle = middle[middle[:, 0].argsort()]
            return np.concatenate([np.array([[0, m_theta]]), sort_middle, np.array([[math.pi, m_theta]])])

        def unique_filter(pts):
            _, unique = np.unique(pts[:,0], return_index=True)
            include = np.full((pts.shape[0]), False)
            include[unique] = True
            return include

        def max_slope(pts, mx):
            slope = np.absolute((pts[1:,1] - pts[:-1,1]) / (pts[1:,0] - pts[:-1,0]))
            return np.concatenate([slope < mx, [True]])

        include_l = limit_range(pts_ll, pts_lr, 0.75)
        include_r = limit_range(pts_rl, pts_rr, 0.75)
        include = np.logical_and(include_l, include_r)

        pts_l = sort_middle(pts_ll[include], pts_lr[include])
        pts_r = sort_middle(pts_rl[include], pts_rr[include])

        include = np.logical_and(unique_filter(pts_l), unique_filter(pts_r))
        pts_l = pts_l[include]
        pts_r = pts_r[include]

        include = np.logical_and(max_slope(pts_l, 1.0), max_slope(pts_r, 1.0))
        pts_l = pts_l[include]
        pts_r = pts_r[include]

        print('stitch points: ', pts_l.shape[0], pts_r.shape[0])

        return pts_l, pts_r

    def _align(self, matches):
        m0 = range(0, 2)
        m1 = range(2, 4)
        # reject outliers
        for i in range(len(matches)):
            m = matches[i]
            diff = m[:,m1] - m[:,m0]
            s = np.std(diff, axis=0)
            mn = np.mean(diff, axis=0)
            accept = [1.5, 3]
            valid = np.logical_and(diff > (mn - accept * s), diff < (mn + accept * s)).all(axis=1)
            matches[i] = m[valid]

        middles = []
        for m in matches:
            middles.append((m[:,m0] + m[:,m1]) / 2)

        left_side = np.array([0, math.pi / 2])

        transforms = []
        for l in range(len(matches)):
            r = (l + 1) % len(matches)

            polar_a = np.concatenate([matches[l][:,m1] - left_side, matches[r][:,m0]])
            polar_f = np.concatenate([middles[l] - left_side, middles[r]])

            t = transform.Transform(self.verbose, self.display)
            t.label = 'align:' + str(l)
            t.calculate_theta_c(polar_a, polar_f)

            polar_a[:,1] = polar_f[:,1]
            t.calculate_phi_c(polar_a, polar_f)

            transforms.append(t)

        return transforms

    # returns an array of Transform objects aligned to each side.
    def align_left(self):
        matches = [
            self._determine_matching_points(self._images[6], self._images[0], 45, -45, 0.6),
            self._determine_matching_points(self._images[0], self._images[2], 45, -45, 0.6),
            self._determine_matching_points(self._images[2], self._images[4], 45, -45, 0.6),
            self._determine_matching_points(self._images[4], self._images[6], 45, -45, 0.6)
        ]

        for m in range(4):
            ll = (2*m-2) % 8
            lr = (2*m)
            matches[m] = np.concatenate([self._transforms[ll].apply_phi_lr_c(matches[m][:,0:2]),
                                         self._transforms[lr].apply_phi_lr_c(matches[m][:,2:4])], axis=1)

        transforms = self._align(matches)
        for i, t in enumerate(transforms):
            self._transforms[2*i].theta_coeffs = t.theta_coeffs;
            self._transforms[2*i].phi_coeffs = t.phi_coeffs;

    # returns an array of Transform objects aligned to each side.
    def align_right(self):
        matches = [
            self._determine_matching_points(self._images[7], self._images[1], 45, -45, 0.6),
            self._determine_matching_points(self._images[1], self._images[3], 45, -45, 0.6),
            self._determine_matching_points(self._images[3], self._images[5], 45, -45, 0.6),
            self._determine_matching_points(self._images[5], self._images[7], 45, -45, 0.6)
        ]

        for m in range(4):
            rl = (2*m-1) % 8
            rr = (2*m+1) % 8
            matches[m] = np.concatenate([self._transforms[rl].apply_phi_lr_c(matches[m][:,0:2]),
                                self._transforms[rr].apply_phi_lr_c(matches[m][:,2:4])], axis=1)

        transforms = self._align(matches)
        for i, t in enumerate(transforms):
            self._transforms[2*i+1].theta_coeffs = t.theta_coeffs;
            self._transforms[2*i+1].phi_coeffs = t.phi_coeffs;

    def align_all(self):
        left = [
            self._determine_matching_points(self._images[6], self._images[0], 45, -45, 0.6),
            self._determine_matching_points(self._images[0], self._images[2], 45, -45, 0.6),
            self._determine_matching_points(self._images[2], self._images[4], 45, -45, 0.6),
            self._determine_matching_points(self._images[4], self._images[6], 45, -45, 0.6)
        ]
        right = [
            self._determine_matching_points(self._images[7], self._images[1], 45, -45, 0.6),
            self._determine_matching_points(self._images[1], self._images[3], 45, -45, 0.6),
            self._determine_matching_points(self._images[3], self._images[5], 45, -45, 0.6),
            self._determine_matching_points(self._images[5], self._images[7], 45, -45, 0.6)
        ]

        matches = []
        for m in range(4):
            ll = (2*m-2) % 8
            lr = (2*m)
            rl = (2*m-1) % 8
            rr = (2*m+1) % 8
            l = np.concatenate([self._transforms[ll].apply_phi_lr_c(left[m][:,0:2]),
                                self._transforms[lr].apply_phi_lr_c(left[m][:,2:4])], axis=1)
            r = np.concatenate([self._transforms[rl].apply_phi_lr_c(right[m][:,0:2]),
                                self._transforms[rr].apply_phi_lr_c(right[m][:,2:4])], axis=1)
            matches.append(np.concatenate([l, r]))

        transforms = self._align(matches)
        for i, t in enumerate(transforms):
            self._transforms[2*i].theta_coeffs = t.theta_coeffs;
            self._transforms[2*i+1].theta_coeffs = t.theta_coeffs;
            self._transforms[2*i].phi_coeffs = t.phi_coeffs;
            self._transforms[2*i+1].phi_coeffs = t.phi_coeffs;


    def align_verticle(self):
        img_pairs = []
        for i in range(0, len(self._images), 2):
            img_pairs.append((self._images[i], self._images[i+1]))
            img_pairs.append((self._images[i+1], self._images[i]))

        for i, pair in enumerate(img_pairs):
            matches = np.zeros((0,4))
            for a in [-90, -30, 30, 90]:
                m = self._determine_matching_points(pair[0], pair[1], a, a, 0.5)
                matches = np.concatenate([matches, m])

            middle = (matches[:,0:2] + matches[:,2:4]) / 2

            self._transforms[i].calculate_phi_lr_c(matches[:,0:2], middle)
