#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
import threading
from Equirec2Perspec import Equirectangular
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.spatial import KDTree
#from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation
from linear_regression import LinearRegression
from feature_matcher import FeatureMatcher2
from feature_matcher import FeatureMatcher4
import threading


def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

def trim_outliers(i, d):
    m = np.mean(i)
    std = np.std(i)

    inc = np.logical_and(m - d*std < i, i < m + d*std)
    return i[inc], inc


class RegionFeatures():
    def __init__(self):
        self.to_equ = None
        self.rectilinear = None
        self.gray = None
        self.keypoints = None
        self.descripts = None

def switch_axis(p):
    # adjust from the scripts perception of the world axis, y-front to the vuze camera
    # perception which is y-up.
    return np.matmul(np.transpose(p), np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]))

def radius_compute(left, right, p_l, p_r, minimum=0.1, maximum=10):
    n = left.shape[0]
    v_p_lr = p_l - p_r
    d_p_lr = np.sqrt(v_p_lr.dot(np.transpose(v_p_lr)))

    m_l = np.zeros((n, 3))
    m_r = np.zeros((n, 3))

    m_l = coordinates.polar_to_cart(left, 1)
    a_l = m_l[:,0:1]
    b_l = m_l[:,1:2]
    c_l = m_l[:,2:3]

    m_r = coordinates.polar_to_cart(right, 1)
    a_r = m_r[:,0:1]
    b_r = m_r[:,1:2]
    c_r = m_r[:,2:3]

    # v = p + r * m
    # find the point on v_l and v_r such that the points are closest
    # r_l and r_r are the radius along each line that results in the closest point
    # if the point is r_l = r_r = 0, ignore the point,
    m_d = np.cross(m_r, m_l)

    # normalize m_d, so that r_d will be the closest distance between v_l and v_r
    m_d_n = np.linalg.norm(m_d, axis=1).reshape((n, 1))
    parallel = (m_d_n == 0)[:,0]
    m_d_n[parallel] = 1
    if np.count_nonzero(parallel) > 0:
        print('radius_compute: parallel points:', np.count_nonzero(parallel))

    m_d = m_d / m_d_n
    m_d[parallel] = np.nan

    a_d = m_d[:,0:1]
    b_d = m_d[:,1:2]
    c_d = m_d[:,2:3]

    a_pl = p_l[0,0]
    b_pl = p_l[0,1]
    c_pl = p_l[0,2]

    a_pr = p_r[0,0]
    b_pr = p_r[0,1]
    c_pr = p_r[0,2]

    r_d = (a_l*b_pl*c_r - a_l*b_pr*c_r - a_l*b_r*c_pl + a_l*b_r*c_pr - a_pl*b_l*c_r + a_pl*b_r*c_l + a_pr*b_l*c_r - a_pr*b_r*c_l + a_r*b_l*c_pl - a_r*b_l*c_pr - a_r*b_pl*c_l + a_r*b_pr*c_l)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)
    r_r = (a_d*b_l*c_pl - a_d*b_l*c_pr - a_d*b_pl*c_l + a_d*b_pr*c_l - a_l*b_d*c_pl + a_l*b_d*c_pr + a_l*b_pl*c_d - a_l*b_pr*c_d + a_pl*b_d*c_l - a_pl*b_l*c_d - a_pr*b_d*c_l + a_pr*b_l*c_d)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)
    r_l = (-a_d*b_pl*c_r + a_d*b_pr*c_r + a_d*b_r*c_pl - a_d*b_r*c_pr + a_pl*b_d*c_r - a_pl*b_r*c_d - a_pr*b_d*c_r + a_pr*b_r*c_d - a_r*b_d*c_pl + a_r*b_d*c_pr + a_r*b_pl*c_d - a_r*b_pr*c_d)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)

    #v_l = p_l + r_l * m_l
    #v_r = p_r + r_r * m_r
    #r = np.linalg.norm((v_l + v_r) / 2 - (p_l + p_r) / 2, axis=1).reshape((n, 1))

    r_l = np.linalg.norm(r_l * m_l, axis=1).reshape((n, 1))
    r_r = np.linalg.norm(r_r * m_r, axis=1).reshape((n, 1))

    r_l[parallel] = maximum
    r_l[r_l < minimum] = minimum

    r_r[parallel] = maximum
    r_r[r_r < minimum] = minimum

    r_d[parallel] = d_p_lr
    return r_l, r_r, r_d


class DepthCalibration():
    def __init__(self, debug=None):
        self._patches = None
        self._debug = debug

        self._coords = np.zeros((2, 0, 2), np.float32)
        self._expected = np.zeros((0, 2), np.float32)
        self._r_expected = np.zeros((0, 1), np.float32)
        self._r_initial = np.zeros((0, 1), np.float32)

        self._mode = 'linreg'
        self._linreg = None
        self._rotation = None

    def set_patches(self, patches):
        self._patches = patches;
        return self

    # mode is either 'kabsch' or 'linreg'
    def set_mode(self, mode):
        self._mode = mode
        return self

    def to_dict(self):
        r = {}
        if self._mode is not None:
            r['mode'] = self._mode
        if self._linreg is not None:
            r['linreg'] = self._linreg.to_dict()
        if self._rotation is not None:
            r['rotation'] = self._rotation.tolist()
        return r

    def from_dict(self, d):
        if 'linreg' in d:
            self._linreg = LinearRegression().from_dict(d['linreg'])
        if 'mode' in d:
            self._mode = d['mode']
        if 'rotation' in d:
            self._rotation = np.array(d['rotation'])
        return self

    # returns a tuple (coords, valid)
    # assumes img is the middle of an equirectangular image
    # where coordinates is in polar with 0 radians along the positive x-axis
    def _find_colors(self, img, patches):
        n = len(patches)
        coords = np.zeros((n, 2))
        valid = np.zeros((n,), dtype=np.bool)
        for pi, p in enumerate(patches):
            ci = int(p["color"], 16)
            c = np.array([(ci >> 0) & 0xff, (ci >> 8) & 0xff, (ci >> 16) & 0xff])
            center = self._find_color(img, c)
            if center is not None:
                coords[pi, 0] = center[1]
                coords[pi, 1] = center[0]
                valid[pi] = True

        coords = coordinates.eqr_to_polar(coords, img.shape)
        coords[:,1] = math.pi - coords[:,1]
        return coords, valid

    # returns the theta and phi of the center of the color patch within the image
    def _find_color(self, img, c):
        tol = 10
        ind = np.nonzero(np.logical_and(np.logical_and(np.abs(img[...,0].astype(np.float32) - c[0]) < tol, \
                                                       np.abs(img[...,1].astype(np.float32) - c[1]) < tol), \
                                        np.abs(img[...,2].astype(np.float32) - c[2]) < tol))
        if ind[0].shape[0] == 0 or ind[1].shape[0] == 0:
            return None
        return np.array([np.mean(ind[0]), np.mean(ind[1])])

    # images a and b and their positions.
    # note: adjustments are made to image a only.
    def add_coordinates(self, a, b, p_a, p_b):
        a = get_middle(a)
        b = get_middle(b)
        p_a = switch_axis(p_a)
        p_b = switch_axis(p_b)
        coords, expected, r_expected = \
            self._determine_coordinates(a, b, p_a, p_b, self._patches)
        r_initial, _, _ = radius_compute(coords[0], coords[1], p_a, p_b)

        self._coords = np.concatenate([self._coords, coords], axis=1)
        self._expected = np.concatenate([self._expected, expected], axis=0)
        self._r_expected = np.concatenate([self._r_expected, r_expected], axis=0)
        self._r_initial = np.concatenate([self._r_initial, r_initial], axis=0)

    def apply(self, img):
        if self._mode == 'linreg':
            return self._apply_linreg(self._linreg, img)
        if self._mode == 'kabsch':
            return self._apply_kabsch(self._rotation, img)
        return None

    def finalize(self):
        r_exp = self._r_expected
        r = self._r_initial
        print('initial squared error:', np.sum((r-r_exp)*(r-r_exp)) / r_exp.shape[0])
        print('samples:', self._expected.shape[0])

        if self._debug.enable('depth-samples'):
            c = self._r_expected * coordinates.polar_to_cart(self._coords[0], 1)
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(c[:,0], c[:,1], c[:,2], marker='.')

        if self._debug.enable('depth-error'):
            d = self._coords[0] - self._expected
            f = plt.figure()
            f.add_subplot(1, 2, 1, projection='3d') \
             .scatter(self._coords[0][:,0], self._coords[0][:,1], d[:,0], marker='.')
            f.add_subplot(1, 2, 2, projection='3d') \
             .scatter(self._coords[0][:,0], self._coords[0][:,1], d[:,1], marker='.')


        if self._mode == 'linreg':
            self._finalize_linreg(self._coords[0], self._expected)
        if self._mode == 'kabsch':
            self._finalize_kabsch(self._coords[0], self._expected)

        return self

    def _determine_coordinates(self, a, b, p_a, p_b, patches):
        r_expected = np.zeros((len(patches), 1))
        for pi, p in enumerate(patches):
            r_expected[pi,0] = p['distance']

        n = len(patches)
        coords = np.zeros((2, n, 2))
        valid = np.zeros((2, n), dtype=np.bool)
        for ii, i in enumerate([a, b]):
            coords[ii], valid[ii] = self._find_colors(i, patches)

        valid_pair = np.logical_and(valid[0], valid[1])
        coords = coords[:,valid_pair]
        r_expected = r_expected[valid_pair]
        n = np.count_nonzero(valid_pair)

        cart_a = coordinates.polar_to_cart(coords[0], 1)
        cart_b = coordinates.polar_to_cart(coords[1], 1)
        p_a_exp = p_a + r_expected * cart_a
        p_b_exp = p_b + r_expected * cart_b
        p_exp = 0.5 * (p_a_exp + p_b_exp)
        cart_a_exp = p_exp - p_a
        cart_a_exp = cart_a_exp / np.linalg.norm(cart_a_exp, axis=1).reshape((n, 1))
        expected = coordinates.cart_to_polar(cart_a_exp)

        return coords, expected, r_expected

    def _apply_linreg(self, linreg, img):
        shape_full = (img.shape[0], 2*img.shape[0])
        shape_half = (shape_full[0], int(shape_full[1] / 2))

        center_pts = coordinates.polar_points_3d(shape_half)
        center_pts = linreg.evaluate(center_pts)
        center_pts = coordinates.polar_to_eqr_3d(center_pts, shape_full)
        center_pts -= [shape_full[1]/4, 0]
        return coordinates.eqr_interp_3d(center_pts, img)

    def _apply_kabsch(self, rot, img):
        s = (img.shape[0], img.shape[1])
        center_pts = coordinates.polar_points_3d(s)
        center_pts_cart = coordinates.polar_to_cart([0, 3/2*math.pi] + np.array([1, -1]) * center_pts, 1)
        center_pts_cart = np.transpose(np.matmul(rot, np.transpose(center_pts_cart.reshape((s[0] * s[1], 3))))).reshape(s + (3,))
        center_pts = [0, 3/2*math.pi] + np.array([1, -1]) * coordinates.cart_to_polar(center_pts_cart)
        center_pts_eqr = coordinates.polar_to_eqr_3d(center_pts, (s[0], 2*s[0]))
        center_pts_eqr -= [s[1]/2, 0]
        return coordinates.eqr_interp_3d(center_pts_eqr, img)

    def _finalize_linreg(self, act, exp):
        # convert back to coordinates with the image centered at pi
        exp = [0, 3*math.pi/2] + [1, -1] * exp
        act = [0, 3*math.pi/2] + [1, -1] * act

        self._linreg = LinearRegression(np.array([2, 4]), False)
        err = self._linreg.regression(exp, act)
        print('linear regression depth squared error:', np.sum(err*err, axis=0))

    def _finalize_kabsch(self, act, exp):
        cart_a = coordinates.polar_to_cart(act, 1)
        cart_a_exp = coordinates.polar_to_cart(exp, 1)

        rot, rssd = Rotation.align_vectors(cart_a, cart_a_exp)
        print('kabsch rssd', rssd)

        est = np.transpose(np.matmul(rot.as_matrix(), np.transpose(cart_a_exp)))
        err = est - cart_a
        print('kabsch cart init:', np.sum((cart_a - cart_a_exp)*(cart_a - cart_a_exp)))
        print('kabsch cart err:', np.sum(err * err))
        self._rotation = rot.as_matrix()

    def result_info(self, a, b, p_a, p_b):
        a = get_middle(a)
        b = get_middle(b)
        p_a = switch_axis(p_a)
        p_b = switch_axis(p_b)

        info = np.zeros((4,), np.float32)
        coords, expected, r_exp = \
            self._determine_coordinates(a, b, p_a, p_b, self._patches)
        r, _, d = radius_compute(coords[0], coords[1], p_a, p_b)
        info[0] = np.sum((r-r_exp)*(r-r_exp)) / r_exp.shape[0]
        info[2] = np.sum(d*d)

        b_adj = self.apply(b)
        if self._debug.enable('depth-cal-finalize'):
            self._debug.subplot('depth-cal-left').imshow(a)
            self._debug.subplot('depth-cal-original').imshow(b)
            self._debug.subplot('depth-cal-right').imshow(b_adj)

        coords, expected, r_exp = \
            self._determine_coordinates(a, b_adj, p_a, p_b, self._patches)
        r, _, d = radius_compute(coords[0], coords[1], p_a, p_b)
        info[1] = np.sum((r-r_exp)*(r-r_exp)) / r_exp.shape[0]
        info[3] = np.sum(d*d)
        return info

class DepthMap():

    # depth map for a constant radius sphere
    def __init__(self, radius, position=None):
        self._r = radius
        if position is not None:
            self._p = position.reshape((3, 1))
        else:
            self._p = np.zeros((3,1), np.float32)

    def plot(self, img):
        f = plt.figure()
        img0 = get_middle(img)
        f.add_subplot(1, 2, 1).imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))

        ax = f.add_subplot(1, 2, 2)
        pts = coordinates.polar_points_3d((1024, 1024))
        r = self.eval(pts)
        if np.min(r) != np.max(r) and np.mean(r) != np.max(r) and np.mean(r) != np.min(r):
            clr = colors.TwoSlopeNorm(vmin=np.min(r), vcenter=np.mean(r), vmax=np.max(r))
            pos = ax.imshow(r, norm=clr, cmap='summer', interpolation='none')
        else:
            pos = ax.imshow(r, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

    # c is a set of polar coordinates, radiating from the center of the sphere.
    # this is a special case of the intersect function
    # returns the radius at each point in c.
    def eval(self, c):
        cart = coordinates.polar_to_cart(c, 1).reshape(c.shape[:-1] + (3, 1))
        P = self.intersect(self._p, cart)
        return np.linalg.norm(P - self._p[...,0], axis=-1)

    # intersect with lens sphere, in cartesian space.
    # c and slope are cartesian vectors
    # can be higher dimensional as long as the last dimension of each
    # is [...,3,1]
    def intersect(self, c, slope):
        offset = c - self._p
        disc = np.sum(slope * offset, axis=-2) ** 2 \
            - (np.linalg.norm(offset, axis=-2)**2 - self._r**2)
        d = -np.sum(slope * offset, axis=-2) + np.sqrt(disc)

        P = (c + d.reshape((d.shape + (1,))) * slope)
        return P.reshape(slope.shape[:-2] + (3,))

    def to_dict(self):
        d = {'type': 'constant'}
        d['r'] = self._r
        d['p'] = self._p.tolist()
        return d

    @staticmethod
    def from_dict(d):
        if not 'type' in d:
            return DepthMap(1)

        if d['type'] == 'constant':
            return DepthMap(d['r'], np.array(d['p'], np.float32))
        elif d['type'] == 'cloud':
            return DepthMapCloud.from_dict(d)
        elif d['type'] == 'ellipsoid':
            return DepthMapEllipsoid.from_dict(d)

        return DepthMap(1)

class DepthMapEllipsoid(DepthMap):

    # depth map for an ellipsoid with r=[a, b, c] for the scaling
    # along the x, y, z axes.
    def __init__(self, radius, position=None):
        super().__init__(radius.reshape((3, 1)), position)

    # intersect with lens ellipsoid
    # c and slope are cartesian vectors
    # can be higher dimensional as long as the last dimension of each
    # is [...,3,1]
    def intersect(self, c, slope):
        offset = c - self._p

        rr = self._r**2
        qa = np.sum(slope**2 / rr, axis=-2)
        qb = np.sum(2 * slope * offset / rr, axis=-2)
        qc = np.sum(offset**2 / rr, axis=-2) - 1
        d = (-qb + np.sqrt(qb**2 - 4*qa*qc)) / (2 * qa)

        P = (c + d.reshape((d.shape + (1,))) * slope)
        return P.reshape(slope.shape[:-2] + (3,))

    def to_dict(self):
        d = {'type': 'ellipsoid'}
        d['r'] = self._r.tolist()
        d['p'] = self._p.tolist()
        return d

    @staticmethod
    def from_dict(d):
        if not 'type' in d or d['type'] != 'ellipsoid':
            return DepthMap(1)
        return DepthMapEllipsoid(np.array(d['r'], np.float32), \
                                 np.array(d['p'], np.float32))


class DepthMapCloud(DepthMap):
    def __init__(self, polar, r):
        super().__init__(5)

        self._pts = polar
        self._tree = KDTree(coordinates.polar_to_cart(polar, 1))
        self._area = min(len(polar), 4)
        self._r = r

    def to_dict(self):
        d = {'type': 'cloud'}
        d['points'] = self._pts.tolist()
        d['area'] = self._area
        d['r'] = self._r.tolist()
        return d

    @staticmethod
    def from_dict(d):
        if not 'type' in d or d['type'] != 'cloud':
            return DepthMap(1)

        plr = np.array(d['points'], np.float32)
        r = np.array(d['r'], np.float32)

        d = DepthMapCloud(plr, r)
        d.set_area(d['area'])
        return d

    # the integer number of nearest neighbors to consider during evaluation of a point.
    def set_area(self, a):
        self._area = min(len(self._pts), a)
        return self

    def plot(self, img):
        f = plt.figure()
        img0 = get_middle(img)
        f.add_subplot(1, 2, 1).imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))

        ax = f.add_subplot(1, 2, 2)
        pts = coordinates.polar_points_3d((1024, 1024))
        r = self.eval(pts)
        clr = colors.TwoSlopeNorm(vmin=np.min(r), vcenter=np.mean(r), vmax=np.max(r))
        pos = ax.imshow(r, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

    def average_r(arr):
        pts = arr[0]._pts
        r = np.zeros(arr[0]._r.shape, np.float32)
        for m in arr:
            r += (m._r / len(arr))
        return DepthMapCloud(pts, r)

    def merge(arr):
        pts = []
        r = []
        for m in arr:
            pts += [m._pts]
            r += [m._r]

        return DepthMapCloud(np.concatenate(pts), np.concatenate(r))

    def filter(self, indices):
        return DepthMapCloud(self._pts[indices], self._r[indices])

    def eval(self, c):
        c_1 = c
        if len(c.shape) == 3:
            c_1 = coordinates.to_1d(c)

        c_1 = coordinates.polar_to_cart(c_1, 1)
        dist, idx = self._tree.query(c_1, k=self._area, workers=8)

        if self._area == 1:
            return self._r[idx].reshape(c.shape[:-1])

        limit = (np.max(dist, axis=-1) + np.min(dist, axis=-1)).reshape(dist.shape[:-1] + (1,))
        dist = (limit - dist) ** 2
        return (np.sum(self._r[idx].reshape(dist.shape) * dist, axis=-1) \
            / np.sum(dist, axis=-1)).reshape(c.shape[:-1])


class DepthMapper():
    def __init__(self, debug):
        self._debug = debug

        self._min = 1.2
        self._mid = 10.0
        self._max = 60.0

    def _kernel(self, size):
        v = np.arange(size) - int(size/2)
        x, y = np.meshgrid(v, v)
        k = np.zeros((size, size), np.float32)
        in_circle = np.sqrt(x*x + y*y) < size/2
        k[in_circle] = 1
        return k

    def _filter_radii(self, r, d, err):
        inc = np.abs(d) < err
        return r[inc], inc

    def _generate_map(self, img, polar, polar_r):
        mn = polar_r < self._mid
        mx = polar_r > self._mid

        r = polar_r.copy()
        """
        if np.count_nonzero(mn) > 0 and np.min(polar_r) < self._min:
            mn_scale = (self._mid - self._min) / (self._mid - np.min(polar_r))
            r[mn] = self._mid - (self._mid - r[mn]) * mn_scale

        if np.count_nonzero(mx) > 0 and np.max(polar_r) > self._max:
            mx_scale = (self._max - self._mid) / (np.max(polar_r) - self._mid)
            r[mx] = (r[mx] - self._mid) * mx_scale + self._mid
        """

        m = DepthMapCloud(polar, r)

        return m


class DepthMapperSlice(DepthMapper):
    def __init__(self, images, positions, debug):
        super().__init__(debug)

        self._images = images
        self._positions = [
            switch_axis(positions[0]),
            switch_axis(positions[1]),
            np.matmul(switch_axis(positions[2]), \
                      np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
            np.matmul(switch_axis(positions[3]), \
                      np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
        ]
        self._maps = None
        self._matches = None
        self._max_err_dist = 0.02

    # must call map() first.
    def matches(self):
        return self._matches

    def map(self):
        if self._maps is not None:
            return self._maps

        matches = FeatureMatcher4(self._images[0:4:2], self._images[1:4:2], self._debug) \
            .matches()
        self._matches = np.zeros(matches.shape, np.float32)
        self._matches[:,0] = matches[:,0]
        self._matches[:,1] = matches[:,2]
        self._matches[:,2] = matches[:,1]
        self._matches[:,3] = matches[:,3]
        matches = self._matches

        dim = self._images[0].shape[0]
        maps = [[] for i in range(4)]
        inc_all = np.ones((self._matches.shape[0],), bool)
        shift = np.array([0, math.pi/2], np.float32)
        pairs = [(0, 2), (1, 3)]
        for p in pairs:
            r_l, r_r, d = radius_compute(matches[:,p[0]], \
                                         matches[:,p[1]], \
                                         self._positions[p[0]], \
                                         self._positions[p[1]])

            r_l, inc = self._filter_radii(r_l, d, self._max_err_dist)
            r_r, _ = self._filter_radii(r_r, d, self._max_err_dist)
            inc_all = np.logical_and(inc[:,0], inc_all)

            maps[p[0]] += [self._generate_map(self._images[p[0]], \
                                              matches[inc[:,0], p[0]] - int(p[0]/2)*shift, r_l)]
            maps[p[1]] += [self._generate_map(self._images[p[1]], \
                                              matches[inc[:,0], p[1]] - int(p[1]/2)*shift, r_r)]

        self._matches = self._matches[inc_all]
        self._debug.log('map points:', self._matches.shape[0])
        self._maps = [DepthMapCloud.average_r(m) for m in maps]

        if self._debug.enable('depth-map'):
            for m, img in zip(self._maps, self._images):
                m.plot(img)

        return self._maps


class DepthMapper2D(DepthMapper):

    def __init__(self, img_left, img_right, p_left, p_right, debug):
        super().__init__(debug)

        self._img_left = img_left
        self._img_right = img_right
        self._p_left = switch_axis(p_left)
        self._p_right = switch_axis(p_right)

        self._map_left = None
        self._map_right = None

    def map(self):
        if self._map_left is None or self._map_right is None:
            matches = FeatureMatcher2(self._img_left, self._img_right, self._debug).matches()

            self._debug.log('matches', matches.shape[0])
            r_l, r_r, d = radius_compute(matches[:,0], matches[:,1], self._p_left, self._p_right)

            self._map_left = self._generate_map(self._img_left, matches[:,0], r_l)
            self._map_right = self._generate_map(self._img_right, matches[:,1], r_r)

        if self._debug.enable('depth-map'):
            self._map_left.plot(self._img_left)
            self._map_right.plot(self._img_right)


        return (self._map_left, self._map_right)
