#!/usr/bin/python

# Operates on 180deg FOV equirectangular images only!
import copy
import getopt
import sys
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular
from skimage import exposure

class ProgramOptions:

    def __init__(self):
        self.verbose = False
        self.config = ""

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'vc:',
            ["config=", "verbose"])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            elif o in ("-v", "--verbose"):
                self.verbose = True

    def valid(self):
        return len(self.config) > 0


class Config:

    def __init__(self, file_path):
        self.input = ""
        self.output = ""
        self.resolution = 1080
        self.reference = 1

        f = open(file_path, 'r')
        for l in f.readlines():
            cmd = l.strip().split(',')
            if cmd[0] == 'in' and len(cmd) == 2:
                self.input = cmd[1]
            if cmd[0] == 'out' and len(cmd) == 2:
                self.output = cmd[1]
            if cmd[0] == 'resolution' and len(cmd) == 2:
                self.resolution = int(cmd[1])
            if cmd[0] == 'reference' and len(cmd) == 2:
                self.reference = int(cmd[1])

    def valid(self):
        return self.input != '' or self.output != ''

options = ProgramOptions()
if not options.valid():
    exit(1)

config = Config(options.config)
if not config.valid():
    exit(1)

def eqr_to_polar(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    phi = (c[:, 1:2] / h) * math.pi;
    theta = c[:, 0:1] / w * shape[1] / shape[0] * math.pi;
    return np.concatenate([phi, theta], axis=-1)

def polar_to_eqr(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    r = c.copy()
    r[:, 0] = w * c[:, 1] / math.pi * shape[0] / shape[1]
    r[:, 1] = h * (c[:, 0] / math.pi)
    return r

# where all inputs are np.array objects
# https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
def cubic_roots(a, b, c, d):
    p = c/a - b*b/(3*a*a)
    q = 2*b*b*b/(27*a*a*a) - b*c/(3*a*a) + d/a

    # determine the number of roots
    # I only care about the case where there is 1 root.
    ds = q*q/4 + p*p*p/27

    # ensure the final constant is an array.
    D = np.ones(ds.shape)*b/(3*a)

    s = ds > 0
    root = np.zeros(ds.shape)
    root[s] = np.cbrt(-q[s]/2 - np.sqrt(ds[s])) + np.cbrt(-q[s]/2 + np.sqrt(ds[s])) - D[s]
    return root

def quadratic_roots(a, b, c):
    c = b * b - 4 * a * c
    return (-b + np.sqrt(c)) / (2 * a)

def roots(c):
    if c.shape[1] == 4:
        return cubic_roots(c[:,3], c[:,2], c[:,1], c[:,0])
    elif c.shape[1] == 3:
        return quadratic_roots(c[:,2], c[:,1], c[:,0])


class Transform():
    def __init__(self):
        self.theta_coeffs_order = 3
        tc_cnt = (self.theta_coeffs_order + 1)
        self.theta_coeffs = np.zeros(tc_cnt * tc_cnt)

        self.phi_coeffs_order = 2
        pc_cnt = (self.phi_coeffs_order + 1)
        self.phi_coeffs = np.zeros(pc_cnt * pc_cnt)

        self.phi_lr_order = 2
        plr_cnt = (self.phi_lr_order + 1)
        self.phi_lr_c = np.zeros(plr_cnt * plr_cnt)

        self.visualize = True

    def _apply(self, x1, x2, order, c):
        cnt = order + 1
        x = np.zeros((x1.shape[0], cnt * cnt))
        for t in range(cnt):
            for p in range(cnt):
                x[:,t*cnt + p] = np.power(x2, t) * np.power(x1, p)

        return x.dot(c)

    # assumes x2 is fixed, and x1f is the value of x1 + k
    # where k is the result of applying the constants c
    # to a multivariate polynomial of x1,x2 with order.
    def _reverse(self, x1f, x2, order, c):
        cnt = order + 1

        coeffs = c * np.ones((x1f.shape[0], c.shape[0]))
        coeffs[:,0] -= x1f
        coeffs[:,1] += 1

        C = np.zeros((x1f.shape[0], cnt))
        for t in range(cnt):
            for p in range(cnt):
                C[:,p] += coeffs[:,t*cnt+p] * np.power(x2, t)

        # compute the original x1
        return roots(C)


    def _regression(self, x1, x2, order, y):
        cnt = order + 1

        if self.visualize:
            plt.figure('Difference')
            ax = plt.axes(projection ='3d')
            ax.plot3D(x1, x2, y, 'b+', markersize=1)
            plt.xlim([-math.pi/2, math.pi/2])
            plt.ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.2, 0.2)

        x = np.zeros((x1.shape[0], cnt * cnt))
        for t in range(cnt):
            for p in range(cnt):
                x[:,t*cnt + p] = np.power(x2, t) * np.power(x1, p)

        # QR decomposition
        Q, R = np.linalg.qr(x)
        c = np.linalg.inv(R).dot(np.transpose(Q)).dot(y)
        print('constants:', c)

        err = y - self._apply(x1, x2, order, c)
        print('error:', np.mean(err), np.std(err))

        rev = x1 - self._reverse(self._apply(x1, x2, order, c) + x1, x2, order, c)
        print('reverse (expect 0,0):', np.mean(rev), np.std(rev))

        if self.visualize:
            plt.figure('Difference After Adjustment')
            ax = plt.axes(projection ='3d')
            ax.plot3D(x1, x2, err, 'b+', markersize=1)
            plt.xlim([-math.pi/2, math.pi/2])
            plt.ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.1, 0.1)
            plt.show()

        return c

    # matches is a np.array with phi_l, theta_l, phi_r, theta_r coordinate pairs
    # which are assumed to refer to the same points in each image.
    # use these pairs to calculate phi_lr_c coeffecients which cause
    # the phi values for l and r to meet at a mid-point
    def calculate_phi_lr_c(self, c_0, c_1):
        theta = c_0[:,1] - math.pi
        phi = c_0[:,0] - math.pi / 2
        diff = c_1[:,0] - c_0[:,0]

        self.phi_lr_c = self._regression(phi, theta, self.phi_lr_order, diff)

    def apply_phi_lr_c(self, c):
        r = c.copy()
        theta = r[:,1] - math.pi
        phi = r[:,0] - math.pi / 2
        r[:,0] += self._apply(phi, theta, self.phi_lr_order, self.phi_lr_c)
        return r

    def reverse_phi_lr_c(self, c):
        r = c.copy()
        phi = r[:,0] - math.pi / 2
        theta = r[:,1] - math.pi

        r[:,0] = self._reverse(phi, theta, self.phi_lr_order, self.phi_lr_c) + math.pi / 2
        return r

    def calculate_theta_c(self, c_0, c_1):
        theta = c_0[:,1] - math.pi
        phi = c_0[:,0] - math.pi / 2
        diff = c_1[:,1] - c_0[:,1]

        self.theta_coeffs = self._regression(theta, phi, self.theta_coeffs_order, diff)

    def apply_theta_c(self, c):
        r = c.copy()
        theta = r[:,1] - math.pi
        phi = r[:,0] - math.pi / 2
        r[:,1] += self._apply(theta, phi, self.theta_coeffs_order, self.theta_coeffs)
        return r

    def reverse_theta_c(self, c):
        r = c.copy()
        phi = r[:,0] - math.pi / 2
        theta = r[:,1] - math.pi

        r[:,1] = self._reverse(theta, phi, self.theta_coeffs_order, self.theta_coeffs) + math.pi
        return r

    def calculate_phi_c(self, c_0, c_1):
        theta = c_0[:,1] - math.pi
        phi = c_0[:,0] - math.pi / 2
        diff = c_1[:,0] - c_0[:,0]
        self.phi_coeffs = self._regression(phi, theta, self.phi_coeffs_order, diff)

    def apply_phi_c(self, c):
        r = c.copy()
        theta = r[:,1] - math.pi
        phi = r[:,0] - math.pi / 2
        r[:,0] += self._apply(phi, theta, self.phi_coeffs_order, self.phi_coeffs)
        return r

    def reverse_phi_c(self, c):
        r = c.copy()
        phi = r[:,0] - math.pi / 2
        theta = r[:,1] - math.pi

        r[:,0] = self._reverse(phi, theta, self.phi_coeffs_order, self.phi_coeffs) + math.pi / 2
        return r

    def apply(self, c):
        r = self.apply_phi_lr_c(c)
        r = self.apply_theta_c(r)
        r = self.apply_phi_c(r)
        return r

    def reverse(self, c):
        r = self.reverse_phi_c(c)
        r = self.reverse_theta_c(r)
        r = self.reverse_phi_lr_c(r)
        return r

class SpliceImages():
    def __init__(self, images):
        self._images = images

        # tansform(i) of image (i)
        self._transforms = [None]*len(images)

        # seam(i) between images (i) and (i+1)
        self._stitches = [None]*len(images)
        self._st_slopes = [None]*len(images)
        self._st_offset = [None]*len(images)

    def set_transform(self, idx, t):
        self._transforms[idx] = t

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

        polar = eqr_to_polar(eq, shape)
        result = np.zeros((px_count, 3), dtype=np.uint8)
        for s in range(len(self._stitches)):
            print('filter to segment ' + str(s))
            pts = self._filter_to_stitch(polar, s)
            print('convert points to local space')
            pts[:,1] -= s * math.pi / 2
            pts[:,1] += math.pi
            pts[:,1] = pts[:,1] % (2 * math.pi)
            local_pts_polar = self._transforms[s].reverse(pts)
            local_pts_eqr = polar_to_eqr(local_pts_polar, self._images[s].shape)
            print('determine pixel colors ' + str(s))
            pixels = self._interp(local_pts_eqr, self._images[s])
            result[local_pts_eqr[:,2].astype(np.int)] = pixels

        return result.reshape(shape)

class RefineSeam():
    # a, b are equirectangular images with 360 degrees
    # and 0deg is the center of the image.
    # a is on the left, b is on the right
    # the seem is assumed to exist at 45deg in a, and -45deg in b.
    def __init__(self, images):
        self._images = images
        self._transforms = []
        for i in images:
            self._transforms.append(Transform())

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
                rl[i], inv[i] = Equirectangular(imgs[i]).GetPerspective(60, thetas[i], lat, 800, 800);
                kp[i], des[i] = sift.detectAndCompute(rl[i], None)

            matches = [None]*3
            kp_indices = np.zeros((len(kp[0]), 4), dtype=np.int) - 1
            kp_indices[:, 0] = np.arange(0, len(kp[0]))
            for i in range(1, 4):
                matches = self._determine_matches(des[i], des[0], threshold, 500)
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
                lat_polar_pts[:,2*j:2*j+2] = eqr_to_polar(eq_pts, imgs[j].shape)
                lat_polar_pts[:,2*j+1] -= (thetas[j] - thetas[0]) * math.pi / 180

            all_polar_pts = np.concatenate([all_polar_pts, lat_polar_pts])

        return all_polar_pts

    def _determine_matching_points(self, img_a, img_b, theta_a, theta_b, threshold):
        sift = cv.SIFT_create()

        imgs = (img_a, img_b)
        all_polar_pts = np.zeros((0, 4))
        for lat in [60, 0, -60]:
            rl_a, inv_a = Equirectangular(imgs[0]).GetPerspective(60, theta_a, lat, 800, 800);
            rl_b, inv_b = Equirectangular(imgs[1]).GetPerspective(60, theta_b, lat, 800, 800);
            kp_a, des_a = sift.detectAndCompute(rl_a, None)
            kp_b, des_b = sift.detectAndCompute(rl_b, None)
            inv = (inv_a, inv_b)

            matches = self._determine_matches(des_b, des_a, threshold, 100)

            if len(matches) == 0:
                continue

            rl_pts = np.zeros((len(matches), 4))
            for i, m in enumerate(matches):
                rl_pts[i, 0:2] = kp_a[m.trainIdx].pt
                rl_pts[i, 2:4] = kp_b[m.queryIdx].pt

            lat_polar_pts = np.zeros(rl_pts.shape)
            for i in range(2):
                eq_pts = inv[i].GetEquirectangularPoints(rl_pts[:,2*i:2*i+2])
                lat_polar_pts[:,2*i:2*i+2] = eqr_to_polar(eq_pts, imgs[i].shape)
            all_polar_pts = np.concatenate([all_polar_pts, lat_polar_pts])

        # adjust for the difference in theta's
        all_polar_pts[:, 3] -= (theta_b - theta_a) * math.pi / 180
        return all_polar_pts

    def _determine_matches(self, des_a, des_b, threshold, limit):
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

        return good_matches[:limit]


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
            accept = [2, 3]
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

            t = Transform()
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
            matches[m] = np.concatenate([self._transform_v(matches[m][:,0:2], (2*m-2)%8),
                                         self._transform_v(matches[m][:,2:4], 2*m)], axis=1)

        transforms = self._align(matches)
        for i in range(4):
            transforms[i].phi_offset = self._verticle_offset[2*i]
        return transforms

    # returns an array of Transform objects aligned to each side.
    def align_right(self):
        matches = [
            self._determine_matching_points(self._images[7], self._images[1], 45, -45, 0.6),
            self._determine_matching_points(self._images[1], self._images[3], 45, -45, 0.6),
            self._determine_matching_points(self._images[3], self._images[5], 45, -45, 0.6),
            self._determine_matching_points(self._images[5], self._images[7], 45, -45, 0.6)
        ]

        for m in range(4):
            matches[m] = np.concatenate([self._transform_v(matches[m][:,0:2], (2*m-1)%8),
                                         self._transform_v(matches[m][:,2:4], 2*m+1)], axis=1)

        transforms = self._align(matches)
        for i in range(4):
            transforms[i].phi_offset = self._verticle_offset[2*i+1]
        return transforms

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


np.set_printoptions(suppress=True)
print('loading images')
images = []
for l in range(1, 9):
    images.append(cv.imread(config.input + '_' + str(l) + '_eq360.JPG'))

print('matching exposure')
# match histograms using the reference image.
for i in range(len(images)):
    if i != config.reference:
        images[i] = exposure.match_histograms(images[i], images[config.reference], channel_axis=2)

print('refining seams')
seam = RefineSeam(images)
seam.align_verticle()
#tsl = seam.align_left()
#tsr = seam.align_right()
seam.align_all()
ts = seam._transforms;

splice_left = SpliceImages(images[0:8:2])
splice_right = SpliceImages(images[1:8:2])
for s in range(4):
    print('computing seam ' + str(s))
    st_l, st_r = seam.compute_eye_stitch_line(ts[2*s], ts[2*s+1], s)
    st_l[:,1] += s * math.pi / 2
    st_r[:,1] += s * math.pi / 2
    splice_left.set_stitch(s, st_l)
    splice_right.set_stitch(s, st_r)

    splice_left.set_transform(s, ts[2*s])
    splice_right.set_transform(s, ts[2*s+1])

print('generate left eye')
left = splice_left.generate(config.resolution)

print('generate right eye')
right = splice_right.generate(config.resolution)

combined = np.concatenate([left, right])
cv.imwrite(config.output + '.JPG', combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

plt.show()
