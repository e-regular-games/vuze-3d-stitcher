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


class Transform():
    def __init__(self):
        self.theta_coeffs = np.zeros(4)
        self.phi_coeffs = np.zeros(3)
        self.phi_offset = 0

    def _determine_theta(self, p_i, p_f):
        theta = p_i[:, 1] - math.pi
        phi = p_i[:, 0]
        theta_f = p_f[:, 1] - math.pi
        k = theta_f - theta

        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot3D(phi, theta, k, 'bo', markersize=1)

        # the valus for dk appear to match with: dk=c1.*(th-2).*th.*(th+2)-c2.*(ph-3).*ph.*th;
        # putting in terms of constant coeffs
        # k / th = c1*th^2 + c2*th + c3*ph^2 + c4*ph

        y = k / theta
        x = np.zeros((phi.shape[0], 4))
        x[:,0] = theta * theta
        x[:,1] = theta
        x[:,2] = phi * phi
        x[:,3] = phi

        # QR decomposition
        Q, R = np.linalg.qr(x)
        c = np.linalg.inv(R).dot(np.transpose(Q)).dot(y)
        self.theta_coeffs = c

        khat = x.dot(c) * theta

        err = k - khat
        print('theta coeffs', c)
        print('theta stats', np.std(err), np.mean(err))

    def _determine_phi(self, p_i, p_f):
        theta_f = p_f[:, 1] - math.pi
        phi = p_i[:, 0]
        phi_f = p_f[:, 0]

        # the dPhi only changes with respect to theta: dphi = a*th^2 + b*th + c
        y = phi_f - phi
        x = np.zeros((phi.shape[0], 4))
        x[:,0] = theta_f * theta_f * theta_f
        x[:,1] = theta_f * theta_f
        x[:,2] = theta_f
        x[:,3] = 1

        # QR decomposition
        Q, R = np.linalg.qr(x)
        c = np.linalg.inv(R).dot(np.transpose(Q)).dot(y)
        self.phi_coeffs = c

        yhat = x.dot(c)

        err = y - yhat
        print('phi coeffs', c)
        print('phi stats', np.std(err), np.mean(err))

    # p_i and p_f are polar coordinates (phi, theta)
    # where (pi/2, pi) is the center of the image
    def determine(self, p_i, p_f):
        self._determine_theta(p_i, p_f)
        self._determine_phi(p_i, p_f)

    # c columns assumed as: phi, theta, ...
    def polar(self, c):
        r = c.copy()
        theta = c[:,1] - math.pi
        phi = c[:,0] + self.phi_offset

        r[:,0] = phi

        x_th = np.zeros((phi.shape[0], 4))
        x_th[:,0] = theta * theta
        x_th[:,1] = theta
        x_th[:,2] = phi * phi
        x_th[:,3] = phi

        # r is the original value before subtracting pi, so all we have to do is add k.
        r[:, 1] += x_th.dot(self.theta_coeffs) * theta
        theta_1 = r[:,1] - math.pi

        x_ph = np.zeros((phi.shape[0], 4))
        x_ph[:,0] = theta_1 * theta_1 * theta_1
        x_ph[:,1] = theta_1 * theta_1
        x_ph[:,2] = theta_1
        x_ph[:,3] = 1
        r[:, 0] += x_ph.dot(self.phi_coeffs)

        return r

    # assumes c is polar coordinates (phi, theta) with the center of the image
    # at (pi/2, pi)
    def unpolar(self, c):
        r = c.copy()
        theta_1 = c[:,1] - math.pi
        phi = c[:,0] - self.phi_offset
        r[:,0] = phi

        # find roots of: c1 * th^3 + c2 * th^2 + (c + 1) * th - th_1
        # where c = c3 * ph^2 + c4 * ph
        a_th = self.theta_coeffs[0]
        b_th = self.theta_coeffs[1]
        c_th = self.theta_coeffs[2] * phi * phi + self.theta_coeffs[3] * phi + 1
        d_th = -1 * theta_1

        # compute the original theta
        r[:,1] = cubic_roots(a_th, b_th, c_th, d_th) + math.pi

        x_ph = np.zeros((c.shape[0], 4))
        x_ph[:,0] = theta_1 * theta_1 * theta_1
        x_ph[:,1] = theta_1 * theta_1
        x_ph[:,2] = theta_1
        x_ph[:,3] = 1
        r[:,0] -= x_ph.dot(self.phi_coeffs)

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
            local_pts_polar = self._transforms[s].unpolar(pts)
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
        self._verticle_offset = np.zeros((8))

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

            matches = self._determine_matches(des_b, des_a, threshold, 40)

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

    def _transform_v(self, pts, idx):
        r = pts.copy()
        r[:,0] += self._verticle_offset[idx]
        return r

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
        pts_ll = ts_left.polar(self._transform_v(matches[:, 0:2], ll)) - [0, math.pi]
        pts_lr = ts_left.polar(self._transform_v(matches[:, 2:4], lr) - left_side) + left_side - [0, math.pi]
        pts_rl = ts_right.polar(self._transform_v(matches[:, 4:6], rl)) - [0, math.pi]
        pts_rr = ts_right.polar(self._transform_v(matches[:, 6:8], rr) - left_side) + left_side - [0, math.pi]

        def limit_range(l, r, num_std):
            middle = (l + r) / 2
            m_theta = np.median(middle[:, 1])
            d_theta = np.std(middle[:, 1])
            print('median', m_theta, 'stddev', d_theta)
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
            t.determine(polar_a, polar_f)
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
            l = np.concatenate([self._transform_v(left[m][:,0:2], (2*m-2)%8),
                                self._transform_v(left[m][:,2:4], 2*m)], axis=1)
            r = np.concatenate([self._transform_v(right[m][:,0:2], (2*m-1)%8),
                                self._transform_v(right[m][:,2:4], 2*m+1)], axis=1)
            matches.append(np.concatenate([l, r]))

        return self._align(matches)

    def align_verticle(self):
        kv = np.zeros(len(self._images))
        for i in range(0, len(self._images), 2):
            matches = self._determine_matching_points(self._images[i], self._images[i+1], 0, 0, 0.6)

            # reject outliers
            diff = matches[:,2:4] - matches[:,0:2]
            s = np.std(diff, axis=0)
            mn = np.median(diff, axis=0)
            accept = [3, 2]
            valid = np.logical_and(diff > (mn - accept * s), diff < (mn + accept * s)).all(axis=1)
            matches = matches[valid]

            phi_middle = (matches[:,0] + matches[:,2]) / 2
            kv[i] = np.median(phi_middle - matches[:,0])
            kv[i+1] = np.median(phi_middle - matches[:,2])

        print('verticle offset:', kv)
        self._verticle_offset = kv


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
tsl = seam.align_left()
tsr = seam.align_right()
#ts = seam.align_all()

splice_left = SpliceImages(images[0:8:2])
splice_right = SpliceImages(images[1:8:2])
for s in range(4):
    print('computing seam ' + str(s))
    st_l, st_r = seam.compute_eye_stitch_line(tsl[s], tsr[s], s)
    st_l[:,1] += s * math.pi / 2
    st_r[:,1] += s * math.pi / 2
    splice_left.set_stitch(s, st_l)
    splice_right.set_stitch(s, st_r)

    splice_left.set_transform(s, tsl[s])
    splice_right.set_transform(s, tsr[s])

print('generate left eye')
left = splice_left.generate(config.resolution)

print('generate right eye')
right = splice_right.generate(config.resolution)

combined = np.concatenate([left, right])
cv.imwrite(config.output + '.JPG', combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

plt.show()
