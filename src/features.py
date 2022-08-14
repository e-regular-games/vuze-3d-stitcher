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

        f = open(file_path, 'r')
        for l in f.readlines():
            cmd = l.strip().split(',')
            if cmd[0] == 'in' and len(cmd) == 2:
                self.input = cmd[1]
            if cmd[0] == 'out' and len(cmd) == 2:
                self.output = cmd[1]

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


class Transform():
    def __init__(self):
        # constants for the question phi = slope*theta + offset
        self.phi_slope = 0.0
        self.phi_offset = 0.0

        # scaling factor for theta around the center of the image which is
        # assumed to be at theta = 0
        self.theta_scale = 1.0

    # c columns assumed as: phi, theta, ...
    def polar(self, c):
        r = c.copy()
        r[:, 0] = c[:, 0] + self.phi_slope * c[:, 1] + self.phi_offset
        r[:, 1] = self.theta_scale * c[:, 1]
        return r

    def unpolar(self, c):
        r = c.copy()
        r[:, 1] = (c[:, 1] - math.pi) / self.theta_scale
        r[:, 0] = c[:, 0] - self.phi_slope * r[:, 1] - self.phi_offset

        r[:, 1] += math.pi
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

    def _match_between_sides(self, img_a, img_b, theta_a, theta_b, threshold):
        pass

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

            matches = self._determine_matches(des_b, des_a, threshold, 20)

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

    def compute_theta_scaling(self, polar, polar_f):
        theta = polar[..., 1]
        theta_f = polar_f[..., 1]

        # theta_f = theta * k -> solve for k, using linear regression
        k = (theta @ theta_f) / (theta @ theta)
        print('theta_k: ', k)

        polar[..., 1] *= k

        err = polar_f[:, 1] - polar[:, 1]
        print('error:', np.mean(err), np.std(err))

        return k, polar

    def compute_phi_rate(self, polar, polar_f):
        theta = polar[..., 1]
        dphi = polar_f[..., 0]

        # TODO do we need to calculate a constant b to fit y = dphi * theta + b

        # solve dphi = theta * k
        k = (theta @ dphi) / (theta @ theta)
        print('dphi_k: ', k)

        polar[..., 0] += (polar[..., 1] * k)

        err = polar_f[:, 0] - polar[:, 0]
        print('error:', np.mean(err), np.std(err))

        return k, polar

    def show_polar_plot(self, polar_a, polar_b):
        plt.figure()
        xa = np.sin(polar_a[..., 0]) * np.cos(polar_a[..., 1])
        ya = np.sin(polar_a[..., 0]) * np.sin(polar_a[..., 1])
        za = np.cos(polar_a[..., 0])

        ax = plt.axes(projection ='3d')
        ax.plot3D(xa, ya, za, 'bo', markersize=1)

        xb = np.sin(polar_b[..., 0]) * np.cos(polar_b[..., 1])
        yb = np.sin(polar_b[..., 0]) * np.sin(polar_b[..., 1])
        zb = np.cos(polar_b[..., 0])

        ax.plot3D(xb, yb, zb, 'ro', markersize=1)

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim(-1, 1)

    def transform(self, pts, k):
        # expect k to be an array of form [dphi_k, theta_k]
        result = np.zeros(pts.shape)
        result[..., 1] = (pts[..., 1] - math.pi) * k[1]
        result[..., 0] = pts[..., 0] + result[..., 1] * k[0]
        return result


    def compute_stitch_line(self, polar_k, matches, idx):
        l = idx
        r = (idx + 1) % len(matches)

        left_side = np.array([0, math.pi / 2])
        pts_l = self.transform(matches[l][:,0:2], polar_k[l])
        pts_r = self.transform(matches[l][:,2:4] - left_side, polar_k[r]) + left_side

        middle = (pts_l + pts_r) / 2
        m_theta = np.median(middle[:, 1])
        d_theta = np.std(middle[:, 1])
        print('median', m_theta, 'stddev', d_theta)

        include = np.logical_and(middle[:, 1] > m_theta - d_theta, middle[:, 1] < m_theta + d_theta)
        pts_l = pts_l[include]
        pts_r = pts_r[include]
        middle = middle[include]

        sort_middle = middle[middle[:, 0].argsort()]
        unique_phi, indices = np.unique(sort_middle[:, 0], return_index=True)
        sort_middle = sort_middle[indices]

        pts = np.concatenate([np.array([[0, m_theta]]), sort_middle, np.array([[math.pi, m_theta]])])
        return pts

    # idx is the index of the stitch line: 0:4
    # returns line_left, line_right as np.arrays of phi, theta coordinates
    def compute_eye_stitch_line(self, polar_k, kv, idx):
        ll = 2 * idx
        lr = (ll + 2) % 8
        rl = ll + 1
        rr = lr + 1

        img_left = [self._images[ll], self._images[lr]]
        img_right = [self._images[rl], self._images[rr]]

        # matches = numpy.array([ll, lr, rl, rr])
        matches = self._match_between_eyes(img_left, img_right, 0.75)

        matches[:, 0:4] = self.transform_v(matches[:, 0:4], [kv[ll], kv[lr]])
        matches[:, 4:8] = self.transform_v(matches[:, 4:8], [kv[rl], kv[rr]])

        # reject outliers
        keep = np.full((matches.shape[0]), True)
        for m in range(2, 8, 2):
            diff = matches[:,m:m+2] - matches[:,0:2]
            s = np.std(diff)
            mn = np.median(diff)
            valid = np.logical_and(diff > (mn - 2 * s), diff < (mn + 2 * s)).all(axis=1)
            keep = np.logical_and(valid, keep)
        matches = matches[keep]

        left_side = np.array([0, math.pi / 2])
        pts_ll = self.transform(matches[:,0:2], polar_k[idx])
        pts_lr = self.transform(matches[:,2:4] - left_side, polar_k[(idx + 1) % 4]) + left_side
        pts_rl = self.transform(matches[:,4:6], polar_k[idx])
        pts_rr = self.transform(matches[:,6:8] - left_side, polar_k[(idx + 1) % 4]) + left_side

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
            unique_phi, indices = np.unique(sort_middle[:, 0], return_index=True)
            sort_middle = sort_middle[indices]
            return np.concatenate([np.array([[0, m_theta]]), sort_middle, np.array([[math.pi, m_theta]])])

        include_l = limit_range(pts_ll, pts_lr, 0.75)
        include_r = limit_range(pts_rl, pts_rr, 0.75)
        include = np.logical_and(include_l, include_r)

        pts_l = sort_middle(pts_ll[include], pts_lr[include])
        pts_r = sort_middle(pts_rl[include], pts_rr[include])
        self.show_polar_plot(pts_l, pts_r)

        print('stitch points: ', pts_l.shape[0], pts_r.shape[0])

        return pts_l, pts_r

    def _align(self, matches):
        m0 = range(0, 2)
        m1 = range(2, 4)
        # reject outliers
        for m in matches:
            diff = m[:,m1] - m[:,m0]
            s = np.std(diff)
            mn = np.mean(diff)
            valid = np.logical_and(diff > (mn - 2 * s), diff < (mn + 2 * s)).all(axis=1)
            m = m[valid]

        middles = []
        diffs = []
        for m in matches:
            middles.append((m[:,m0] + m[:,m1]) / 2)
            diffs.append((m[:,m1] - m[:,m0]) / 2)

        left_side = np.array([0, math.pi / 2])

        polar_k = np.zeros((4, 2))
        for l in range(len(matches)):
            r = (l + 1) % len(matches)

            polar_a = np.concatenate([matches[l][:,m1] - left_side, matches[r][:,m0]])
            polar_f = np.concatenate([middles[l] - left_side, middles[r]])

            # the offset is to ensure scaling is done from the center of the image
            polar_a -= np.array([math.pi / 2, math.pi])
            polar_f -= np.array([math.pi / 2, math.pi])
            polar_k[l][1], polar_a = self.compute_theta_scaling(polar_a, polar_f)

            polar_f = np.concatenate([diffs[l], diffs[r]])
            polar_k[l][0], polar_a = self.compute_phi_rate(polar_a, polar_f)
            polar_a += np.array([math.pi / 2, 0])

            # try with the transform function
            #polar_a = self.transform(np.concatenate([matches[l][:,m1] - left_side, matches[r][:,m0]]), polar_k[l])
            #polar_b = np.concatenate([middles[l] - left_side, middles[r]])
            #polar_b -= np.array([0, math.pi])
            #self.show_polar_plot(polar_a, polar_b)

        return polar_k, matches

    def transform_v(self, m, kv):
        r = m.copy()
        r[:,0] += kv[0]
        r[:,2] += kv[1]
        return r

    # returns polar_k, which is a 4x2 matrix with each row dphi_k, theta_k
    def align_left(self, kv):
        matches = [
            self._determine_matching_points(self._images[0], self._images[2], 45, -45, 0.6),
            self._determine_matching_points(self._images[2], self._images[4], 45, -45, 0.6),
            self._determine_matching_points(self._images[4], self._images[6], 45, -45, 0.6),
            self._determine_matching_points(self._images[6], self._images[0], 45, -45, 0.6)
        ]

        for m in range(len(matches)):
            matches[m] = self.transform_v(matches[m], [kv[2*m], kv[(2*m+2) % 8]])

        return self._align(matches)

    # returns polar_k, which is a 4x2 matrix with each row dphi_k, theta_k
    def align_right(self, kv):
        matches = [
            self._determine_matching_points(self._images[1], self._images[3], 45, -45, 0.6),
            self._determine_matching_points(self._images[3], self._images[5], 45, -45, 0.6),
            self._determine_matching_points(self._images[5], self._images[7], 45, -45, 0.6),
            self._determine_matching_points(self._images[7], self._images[1], 45, -45, 0.6)
        ]

        for m in range(len(matches)):
            matches[m] = self.transform_v(matches[m], [kv[2*m+1], kv[(2*m+3) % 8]])

        return self._align(matches)

    def align_all(self, kv):
        left = [
            self._determine_matching_points(self._images[0], self._images[2], 45, -45, 0.6),
            self._determine_matching_points(self._images[2], self._images[4], 45, -45, 0.6),
            self._determine_matching_points(self._images[4], self._images[6], 45, -45, 0.6),
            self._determine_matching_points(self._images[6], self._images[0], 45, -45, 0.6)
        ]
        right = [
            self._determine_matching_points(self._images[1], self._images[3], 45, -45, 0.6),
            self._determine_matching_points(self._images[3], self._images[5], 45, -45, 0.6),
            self._determine_matching_points(self._images[5], self._images[7], 45, -45, 0.6),
            self._determine_matching_points(self._images[7], self._images[1], 45, -45, 0.6)
        ]

        matches = []
        for m in range(4):
            l = self.transform_v(left[m], [kv[2*m], kv[(2*m+2) % 8]])
            r = self.transform_v(right[m], [kv[2*m+1], kv[(2*m+3) % 8]])
            matches.append(np.concatenate([l, r]))

        return self._align(matches)

    def align_verticle(self):
        kv = np.zeros(len(self._images))
        for i in range(0, len(self._images), 2):
            matches = self._determine_matching_points(self._images[i], self._images[i+1], 0, 0, 0.5)

            # reject outliers
            diff = matches[:,2:4] - matches[:,0:2]
            s = np.std(diff)
            mn = np.median(diff)
            valid = np.logical_and(diff > (mn - 2 * s), diff < (mn + 2 * s)).all(axis=1)
            matches = matches[valid]

            phi_middle = (matches[:,0] + matches[:,2]) / 2
            kv[i] = np.median(phi_middle - matches[:,0])
            kv[i+1] = np.median(phi_middle - matches[:,2])
        return kv

    def show_matches(self, idx_a, idx_b, theta_a, theta_b, threshold):
        kp_a = self._determine_keypoints(self._images[idx_a], theta_a)
        kp_b = self._determine_keypoints(self._images[idx_b], theta_b)
        matches = self.determine_matches(kp_a, kp_b, threshold)

        kps_a = []
        kps_b = []
        for m in matches:
            pt_a = kp_a.keypoints[m.queryIdx]
            pt_a.pt = kp_a.equirect_points[m.queryIdx]
            kps_a.append(pt_a)

            pt_b = kp_b.keypoints[m.trainIdx]
            pt_b.pt = kp_b.equirect_points[m.trainIdx]
            kps_b.append(pt_b)

            m.queryIdx = len(kps_a) - 1
            m.trainIdx = len(kps_b) - 1

        lol_matches = []
        for m in matches:
            lol_matches.append([m])
        res = cv.drawMatchesKnn(self._images[idx_a], kps_a, self._images[idx_b], kps_b, lol_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure()
        plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))


    def print_keypoint_stats(self):
        d = self._coord_b - self._coord_a
        print('dtheta: mean: ', np.mean(d[..., 1]), ', var: ', np.var(d[..., 1]))
        print('dphi: mean: ', np.mean(d[..., 0]), ', var: ', np.var(d[..., 0]))



np.set_printoptions(suppress=True)
print('loading images')
images = []
for l in range(1, 9):
    images.append(cv.imread(config.input + '_' + str(l) + '_eq360.JPG'))

seam = RefineSeam(images)
kv = seam.align_verticle()
ka, ma = seam.align_all(kv)

splice_left = SpliceImages(images[0:8:2])
splice_right = SpliceImages(images[1:8:2])
for s in range(4):
    print('computing seam ' + str(s))
    st_l, st_r = seam.compute_eye_stitch_line(ka, kv, s)
    st_l[:,1] += s * math.pi / 2
    st_r[:,1] += s * math.pi / 2
    splice_left.set_stitch(s, st_l)
    splice_right.set_stitch(s, st_r)

    tl = Transform()
    tl.phi_slope = ka[s, 0]
    tl.theta_scale = ka[s, 1]
    tl.phi_offset = kv[2*s]

    splice_left.set_transform(s, tl)

    tr = Transform()
    tr.phi_slope = ka[s, 0]
    tr.theta_scale = ka[s, 1]
    tr.phi_offset = kv[2*s+1]
    splice_right.set_transform(s, tr)

print('generate left eye')
left = splice_left.generate(1080)

print('generate right eye')
right = splice_right.generate(1080)

combined = np.concatenate([left, right])
cv.imwrite(config.output + '.JPG', combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

plt.show()
