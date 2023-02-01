#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
import threading
from Equirec2Perspec import Equirectangular
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
#from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation

def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

def choose_closest(r, x):
    final = r[...,0]
    delta = np.absolute(r[...,0] - x)
    for i in range(1, r.shape[-1]):
        pick = np.absolute(r[...,i] - x) < delta
        final[pick] = r[...,i][pick]
        delta[pick] = np.absolute(r[...,i] - x)[pick]
    return final

# where all inputs are np.array objects
# https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
def cubic_roots(a, b, c, d):
    p = c/a - b*b/(3*a*a)
    q = 2*b*b*b/(27*a*a*a) - b*c/(3*a*a) + d/a
    ds = q*q/4 + p*p*p/27

    D = np.ones(ds.shape)*b/(3*a)

    root = np.zeros(ds.shape + (3,))

    s = ds > 0
    if np.count_nonzero(s) > 0:
        root[s,:] = (np.cbrt(-q[s]/2 - np.sqrt(ds[s])) + np.cbrt(-q[s]/2 + np.sqrt(ds[s])) - D[s])[...,None] * np.ones((root.shape))[s,:]

    m = np.logical_and(ds == 0, p != 0)
    if np.count_nonzero(m) > 0:
        root[m,0] = 3*q[m]/p[m]
        root[m,1:3] = -3*q[m]/(2*p[m])[...,None] * np.ones(ds.shape + (2,))[m,:]

    t = np.logical_and(ds < 0, p < 0)
    if np.count_nonzero(t) > 0:
        for k in range(3):
            root[t,k] = 2 * np.sqrt(-p[t]/3) * np.cos(1/3*np.arccos(3*q[t]/(2*p[t])*np.sqrt(-3/p[t])) - 2*math.pi*k/3) - D[t]

    return root

def quadratic_roots(a, b, c):
    C = b * b - 4 * a * c

    res = np.zeros(C.shape + (2,))
    res[...,0] = (-b + np.sqrt(C)) / (2 * a)
    res[...,1] = (-b - np.sqrt(C)) / (2 * a)
    return res

def roots(c, x):
    if c.shape[-1] == 4:
        rts = cubic_roots(c[...,3], c[...,2], c[...,1], c[...,0])
        return choose_closest(rts, x)
    elif c.shape[-1] == 3:
        rts = quadratic_roots(c[...,2], c[...,1], c[...,0])
        return choose_closest(rts, x)
    elif c.shape[-1] == 2:
        return -c[...,0] / c[...,1]

class LinearRegression():
    # order an array of X integers for the order of each independent variable.
    # an order of 2 indicates that terms x^2 and x will be used along with a constant.
    def __init__(self, order=np.array([2])):
        self._order = order
        self._vars = order.shape[0]

        C = 1
        for o in range(self._vars):
            self._order[o] = order[o]+1
            C *= self._order[o]
        self._num_terms = C
        self._coeffs = np.zeros((self._num_terms, 1))

        # the powers of each dependent variable for each term
        self._powers = np.zeros((self._num_terms, self._vars))
        for i in range(C):
            self._powers[i] = self._index_to_powers(i)

    def regression(self, x, y):
        terms = self._terms(x)
        Q, R = np.linalg.qr(terms)
        self._coeffs = np.linalg.inv(R).dot(np.transpose(Q)).dot(y)

        approx = terms.dot(self._coeffs)
        return approx - y

    def evaluate(self, x):
        terms = self._terms(x)
        return terms.dot(self._coeffs)

    # only applies if order is 3 or less
    # only finds the solution for a single variable (0-indexed)
    def reverse(self, x, var):
        # note: at this point self._order is the number of coeffs, not the order
        if self._order[var] > 4:
            return np.zeros(y.shape)

        C = np.zeros(x.shape[:-1] + (self._order[var],))
        C[...,0] = -1 * x[...,var]
        for i in range(self._num_terms):
            t = np.ones(x.shape[:-1])
            for v in range(self._vars):
                if v == var: continue
                t *= np.power(x[...,v], self._powers[i,v])
            ci = int(self._powers[i,var])
            C[..., ci] += t * self._coeffs[i,0]

        return roots(C, x[...,var])

    def to_dict(self):
        return {
            "order": (self._order - 1).tolist(),
            "coeffs": self._coeffs.tolist()
        }

    def from_dict(self, d):
        order = np.array(d["order"])
        self.__init__(order)
        self._coeffs = np.array(d["coeffs"])
        return self

    def _index_to_powers(self, i):
        p = np.zeros((1, self._vars))
        for oi, o in enumerate(self._order):
            p[0,oi] = i % o
            i = int(i / o)
        return p

    def _terms(self, x):
        terms = np.ones(x.shape[:-1] + (self._num_terms,), np.float32)
        for i in range(self._num_terms):
            for v in range(self._vars):
                terms[...,i] *= np.power(x[...,v], self._powers[i,v])
        return terms

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

def radius_compute(left, right, p_l, p_r, maximum=10):
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

    v_l = p_l + r_l * m_l
    v_r = p_r + r_r * m_r
    r = np.linalg.norm((v_l + v_r) / 2 - (p_l + p_r) / 2, axis=1).reshape((n, 1))

    #alpha = np.arccos(np.sum(v_l * v_r, axis=1) / (np.linalg.norm(v_l, axis=1) * np.linalg.norm(v_r, axis=1)))
    r[parallel] = maximum
    r_d[parallel] = d_p_lr
    return r, r_d


class DepthCalibration():
    def __init__(self, img_left=None, img_right=None, p_left=None, p_right=None, debug=None):
        self._img_left = img_left
        self._img_right = img_right
        if p_left is not None:
            self._p_left = switch_axis(p_left)

        if p_right is not None:
            self._p_right = switch_axis(p_right)

        self._debug = debug

        self._coords = None
        self._expected = None
        self._r_expected = None

        self._mode = 'linrex'
        self._linreg = None
        self._rotation = None

    def to_dict(self):
        r = {}
        if self._coords is not None:
            r['coords'] = self._coords.tolist()
        if self._expected is not None:
            r['expected'] = self._expected.tolist()
        if self._r_expected is not None:
            r['distance'] = self._r_expected.tolist()
        if self._mode is not None:
            r['mode'] = self._mode
        if self._linreg is not None:
            r['linreg'] = self._linreg.to_dict()
        if self._rotation is not None:
            r['rotation'] = self._rotation.tolist()
        return r

    def merge_dict(self, d):
        if 'coords' not in d or 'expected' not in d or 'distance' not in d:
            return
        if self._coords is None:
            self._coords = np.array(d['coords'])
        else:
            self._coords = np.concatenate([self._coords, np.array(d['coords'])], axis=1)

        if self._expected is None:
            self._expected = np.array(d['expected'])
        else:
            self._expected = np.concatenate([self._expected, np.array(d['expected'])], axis=0)

        if self._r_expected is None:
            self._r_expected = np.array(d['distance'])
        else:
            self._r_expected = np.concatenate([self._r_expected, np.array(d['distance'])], axis=0)

    def from_dict(self, d):
        if 'coords' in d:
            self._coords = np.array(d['coords'])
        if 'expected' in d:
            self._expected = np.array(d['expected'])
        if 'distance' in d:
            self._r_expected = np.array(d['distance'])
        if 'linreg' in d:
            self._linreg = LinearRegression().from_dict(d['linreg'])
        if 'mode' in d:
            self._mode = d['mode']
        if 'rotation' in d:
            self._rotation = np.array(d['rotation'])
        return self

    def initial_sqr_err(self):
        r, d = radius_compute(self._coords[0], self._coords[1], self._p_left, self._p_right)
        r_exp = self._r_expected
        err = np.sum((r-r_exp)*(r-r_exp))
        print('initial squared error:', err)
        print('initial distance at intersect (r_d*r_d):', np.sum(d*d))
        return err

    # returns a tuple (coords, valid)
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
        coords[:,1] = 3*math.pi/2 - coords[:,1]
        return coords, valid

    # returns the theta and phi of the center of the color patch within the image
    def _find_color(self, img, c):
        tol = 10
        ind = np.nonzero(np.logical_and(np.logical_and(np.abs(img[...,0].astype(np.float32) - c[0]) < tol, np.abs(img[...,1].astype(np.float32) - c[1]) < tol), np.abs(img[...,2].astype(np.float32) - c[2]) < tol))
        if ind[0].shape[0] == 0 or ind[1].shape[0] == 0:
            print('color not found: ', hex(c[0] + (c[1] << 8) + (c[2] << 16)))
            return None
        return np.array([np.mean(ind[0]), np.mean(ind[1])])

    def determine_coordinates(self, patches):
        self._coords, self._expected, self._r_expected = \
            self._determine_coordinates(self._img_left, self._img_right, patches)
        self.initial_sqr_err()

    def apply(self, right):
        if self._mode == 'linreg':
            return self._apply_linreg(self._linreg, right)
        if self._mode == 'kabsch':
            return self._apply_kabsch(self._rotation, right)
        return None

    def finalize(self, patches):
        print('samples:', self._expected.shape[0])

        if self._debug.enable('depth-samples'):
            c = self._r_expected * coordinates.polar_to_cart(self._coords[0], 1)
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(c[:,0], c[:,1], c[:,2], marker='.')

        if self._mode == 'linreg':
            self._finalize_linreg(patches)
        if self._mode == 'kabsch':
            self._finalize_kabsch(patches)

        self._print_result(patches)

        # clear out the data used to generate the linear regression coeffs
        self._coords = None
        self._expected = None
        self._r_expected = None

        return self

    def _determine_coordinates(self, left, right, patches):
        r_expected = np.zeros((len(patches), 1))
        for pi, p in enumerate(patches):
            r_expected[pi,0] = p['distance']

        n = len(patches)
        coords = np.zeros((2, n, 2))
        valid = np.zeros((2, n), dtype=np.bool)
        for ii, i in enumerate([left, right]):
            coords[ii], valid[ii] = self._find_colors(i, patches)

        valid_pair = np.logical_and(valid[0], valid[1])
        coords = coords[:,valid_pair]
        r_expected = r_expected[valid_pair]
        n = np.count_nonzero(valid_pair)

        expected = np.zeros((n,2))
        cart_left = coordinates.polar_to_cart(coords[0], 1)
        p_exp = self._p_left + r_expected * cart_left
        cart_right_exp = p_exp - self._p_right
        cart_right_exp = cart_right_exp / np.linalg.norm(cart_right_exp, axis=1).reshape((n, 1))
        expected = coordinates.cart_to_polar(cart_right_exp)

        return coords, expected, r_expected

    def _apply_linreg(self, linreg, right):
        shape_full = (right.shape[0], 2*right.shape[0])
        shape_half = (shape_full[0], int(shape_full[1] / 2))

        center_pts = coordinates.polar_points_3d(shape_half)
        center_pts = linreg.evaluate(center_pts)
        center_pts = coordinates.polar_to_eqr_3d(center_pts, shape_full)
        return coordinates.eqr_interp_3d(center_pts, right)

    def _apply_kabsch(self, rot, right):
        s = (right.shape[0], right.shape[1])
        center_pts = coordinates.polar_points_3d(s)
        center_pts_cart = coordinates.polar_to_cart([0, 3/2*math.pi] + np.array([1, -1]) * center_pts, 1)
        center_pts_cart = np.transpose(np.matmul(rot, np.transpose(center_pts_cart.reshape((s[0] * s[1], 3))))).reshape(s + (3,))
        center_pts = [0, 3/2*math.pi] + np.array([1, -1]) * coordinates.cart_to_polar(center_pts_cart)
        center_pts_eqr = coordinates.polar_to_eqr_3d(center_pts)
        return coordinates.eqr_interp_3d(center_pts_eqr, right)

    def _finalize_linreg(self, patches):
        # convert back to coordinates with the image centered at pi
        exp = self._expected.copy()
        exp[:,1] = 3*math.pi/2 - exp[:,1]
        act = self._coords[1].copy()
        act[:,1] = 3*math.pi/2 - act[:,1]

        self._linreg = LinearRegression(np.array([4, 4]))
        err = self._linreg.regression(exp, act)
        print('linear regression depth squared error:', np.sum(err*err, axis=0))

    def _finalize_kabsch(self, patches):
        cart_left = coordinates.polar_to_cart(self._coords[0], 1)
        cart_right = coordinates.polar_to_cart(self._coords[1], 1)
        cart_right_exp = coordinates.polar_to_cart(self._expected, 1)

        rot, rssd = Rotation.align_vectors(cart_right, cart_right_exp)
        print('kabsch rssd', rssd)

        est = np.transpose(np.matmul(rot.as_matrix(), np.transpose(cart_right_exp)))
        err = est - cart_right
        print('kabsch cart init:', np.sum((cart_right - cart_right_exp)*(cart_right - cart_right_exp)))
        print('kabsch cart err:', np.sum(err * err))
        self._rotation = rot.as_matrix()

    def _print_result(self, patches):
        right = self.apply(get_middle(self._img_right))
        if self._debug.enable('depth-cal-finalize'):
            f, ax = plt.subplots(2, 1)
            ax[0].imshow(self._img_right)
            ax[1].imshow(right)

        coords, expected, r_exp = \
            self._determine_coordinates(self._img_left, right, patches)
        r, d = radius_compute(coords[0], coords[1], self._p_left, self._p_right)
        print('depth distance squared error:', np.sum((r-r_exp)*(r-r_exp)))
        print('distance at intersect (r_d*r_d):', np.sum(d*d))

class DepthAtSeam():
    def __init__(self, seam, images, calibration):
        # seam is of {"FL", "FR", "BR", "BL"}
        # for Front, Left, Right, Back

        idx = ['fl', 'fr', 'br', 'bl'].index(seam.lower())
        if idx == -1:
            return

        self._left = [images[(2*idx-2) % 8], images[(2*idx) % 8]]
        self._right = [images[(2*idx-1) % 8], images[(2*idx+1) %8]]

        self._t = [calibration[(2*idx-2) % 8], \
                   calibration[(2*idx-1) % 8], \
                   calibration[(2*idx) % 8], \
                   calibration[(2*idx+1) % 8]]


class DepthMesh():

    def __init__(self, images, calibration, debug):
        self._images = images
        self._calibration = calibration
        self._debug = debug

    def generate(self):
        matches = self._features_determine()
        pass

    def _print_sample(self, left, right):
        n = left.shape[0]
        m_l = np.zeros((n, 3))
        m_r = np.zeros((n, 3))

        m_l[:,0] = np.sin(left[:,0]) * np.cos(math.pi/2 - left[:,1])
        m_l[:,1] = np.sin(left[:,0]) * np.sin(math.pi/2 - left[:,1])
        m_l[:,2] = np.cos(left[:,0])

        m_r[:,0] = np.sin(right[:,0]) * np.cos(math.pi/2 - right[:,1])
        m_r[:,1] = np.sin(right[:,0]) * np.sin(math.pi/2 - right[:,1])
        m_r[:,2] = np.cos(right[:,0])

        ch = 300
        choose = np.sort(np.random.choice(n-1, size=ch, replace=False))
        print(choose)
        print('sample_l = [')
        for r in range(ch):
            print('  ', end='')
            for c in range(3):
                if c < 2:
                    print(str(m_l[choose[r], c]), end=', ')
                else:
                    print(str(m_l[choose[r], c]), end=';\n')
        print(']\'')

        print('sample_r = [')
        for r in range(ch):
            print('  ', end='')
            for c in range(3):
                if c < 2:
                    print(str(m_r[choose[r], c]), end=', ')
                else:
                    print(str(m_r[choose[r], c]), end=';\n')
        print(']\'')

    # the left and right theta values as 1d vectors
    def _radius_compute(self, left, right, p_l, p_r):
        n = left.shape[0]
        m_l = np.zeros((n, 3))
        m_r = np.zeros((n, 3))

        m_l[:,0] = np.sin(left[:,0]) * np.cos(math.pi/2 - left[:,1])
        a_l = m_l[:,0:1]
        m_l[:,1] = np.sin(left[:,0]) * np.sin(math.pi/2 - left[:,1])
        b_l = m_l[:,1:2]
        m_l[:,2] = np.cos(left[:,0])
        c_l = m_l[:,2:3]

        m_r[:,0] = np.sin(right[:,0]) * np.cos(math.pi/2 - right[:,1])
        a_r = m_r[:,0:1]
        m_r[:,1] = np.sin(right[:,0]) * np.sin(math.pi/2 - right[:,1])
        b_r = m_r[:,1:2]
        m_r[:,2] = np.cos(right[:,0])
        c_r = m_r[:,2:3]

        #p_l = np.array([[-self._X, self._Y, self._Z]])
        #p_r = np.array([[self._X, self._Y, self._Z]])

        # v = p + r * m
        # find the point on v_l and v_r such that the points are closest
        # r_l and r_r are the radius along each line that results in the closest point
        # if the point is r_l = r_r = 0, ignore the point,
        m_d = np.cross(m_r, m_l)

        # normalize m_d, so that r_d will be the closest distance between v_l and v_r
        m_d = m_d / np.sqrt(np.sum(m_d * m_d, axis=1)).reshape((n, 1))

        a_d = m_d[:,0:1]
        b_d = m_d[:,1:2]
        c_d = m_d[:,2:3]

        a_pl = p_l[0,0]
        b_pl = p_l[0,1]
        c_pl = p_l[0,2]

        a_pr = p_r[0,0]
        b_pr = p_r[0,1]
        c_pr = p_r[0,2]

        print(p_l, p_r)

        r_d = (a_l*b_pl*c_r - a_l*b_pr*c_r - a_l*b_r*c_pl + a_l*b_r*c_pr - a_pl*b_l*c_r + a_pl*b_r*c_l + a_pr*b_l*c_r - a_pr*b_r*c_l + a_r*b_l*c_pl - a_r*b_l*c_pr - a_r*b_pl*c_l + a_r*b_pr*c_l)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)
        r_r = (a_d*b_l*c_pl - a_d*b_l*c_pr - a_d*b_pl*c_l + a_d*b_pr*c_l - a_l*b_d*c_pl + a_l*b_d*c_pr + a_l*b_pl*c_d - a_l*b_pr*c_d + a_pl*b_d*c_l - a_pl*b_l*c_d - a_pr*b_d*c_l + a_pr*b_l*c_d)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)
        r_l = (-a_d*b_pl*c_r + a_d*b_pr*c_r + a_d*b_r*c_pl - a_d*b_r*c_pr + a_pl*b_d*c_r - a_pl*b_r*c_d - a_pr*b_d*c_r + a_pr*b_r*c_d - a_r*b_d*c_pl + a_r*b_d*c_pr + a_r*b_pl*c_d - a_r*b_pr*c_d)/(a_d*b_l*c_r - a_d*b_r*c_l - a_l*b_d*c_r + a_l*b_r*c_d + a_r*b_d*c_l - a_r*b_l*c_d)

        v_l = p_l + r_l * m_l
        v_r = p_r + r_r * m_r

        print('distance between')
        print('r_d', np.mean(np.abs(r_d)), np.std(np.abs(r_d)))
        plt.hist(np.abs(r_d), bins=10)
        print(r_r)

        # want to minimize d
        # rotate around the y-axis using [ cosL 0 sinL; 0 1 0; -sinL 0 cosL ]
        # Only need to rotate 1 lens to meet the other.
        # note that cos^2(L) + sin^2(L) = 1
        # L^2_c + L^2_s = 1
        # syms rho_1 rho_2 z
        # R_l = subs([ rho_1 z rho_2; z 1 z; -rho_2 z rho_1 ], z, 0)

        # syms a_l b_l c_l a_r b_r c_r
        # m_l = [a_l; b_l; c_l]
        # m_r = [a_r; b_r; c_r]
        # m_d = cross(m_r, R_l *  m_l)
        # a_d = m_d(1)
        # b_d = m_d(2)
        # c_d = m_d(3)
        # A = (b_l*c_r - c_l*b_r) / (c_d*b_r - b_d*c_r)

        # syms X
        # r_l = 2 * X / (a_l + A*a_d - c_l*a_r/c_r - A*c_d*a_r/c_r)
        # m_d will now be in terms of L, and R

        v = (v_l + v_r) / 2
        inc = np.logical_and(r_r > 0, r_l > 0)[:,0]
        v = v[inc]

        r = np.sqrt(np.sum(v * v, axis=1))
        print('r', np.min(r), np.max(r))
        return v


    def _features_determine(self):
        points = np.zeros((0, 3), dtype=np.float32)
        for i in range(0, 2, 2):
            matches_plr = self._match_between_eyes(self._images[i], self._images[i+1], 0.75)

            #self._print_sample(matches_plr[:,0], matches_plr[:,1])

            p = self._radius_compute(matches_plr[:,0], matches_plr[:,1], switch_axis(self._calibration[i].t), switch_axis(self._calibration[i+1].t))
            return
            p[:,1] += i / 2 * math.pi / 2

            points = np.concatenate([points, p])

        points_flat = points.reshape((points.shape[0], 3))
        tree = KDTree(points_flat)
        overlapping = tree.query_radius(points_flat, 0.0001)

        include = np.full((points_flat.shape[0],), True, dtype=np.bool)
        skip = np.full((points_flat.shape[0],), False, dtype=np.bool)
        for i, near in enumerate(overlapping):
            if skip[i]: continue
            for o in near:
                if o != i:
                    include[o] = False
                    skip[o] = True

        if self._debug.verbose:
            print('overlapping features', np.count_nonzero(include), points.shape[0])

        points_flat = points_flat[include]
        points = points_flat.reshape(points_flat.shape[0:1] + points.shape[1:])

        if self._debug.enable('depth_points'):
            plt.figure()
            ax = plt.axes(projection ='3d')
            ax.plot3D(points[:,0], points[:,1], points[:,2], 'bo', markersize=1)

            plt.xlim([-self._d_far, self._d_far])
            plt.ylim([-self._d_far, self._d_far])
            ax.set_zlim(-self._d_far, self._d_far)

    def _match_between_eyes(self, img_left, img_right, threshold):
        sift = cv.SIFT_create()
        fig = plt.figure() if self._debug.enable('depth_matches') else None
        fc = 1

        imgs = [img_left] + [img_right]
        thetas = [-45, 0, 45]
        phis = [-45, 0, 45]
        features = [RegionFeatures() for i in range(len(thetas) * len(phis) * len(imgs))]
        all_polar_pts = np.zeros((0, len(imgs), 2), dtype=np.float32)

        def index(p, t, i):
            return p * len(thetas) * len(imgs) + t * len(imgs) + i

        for p, phi in enumerate(phis):
            for t, theta in enumerate(thetas):
                for i, img in enumerate(imgs):
                    f = features[index(p, t, i)]
                    f.rectilinear, f.to_equ = Equirectangular(img) \
                        .GetPerspective(60, theta, phi, 1000, 1000);
                    f.gray = cv.cvtColor(f.rectilinear, cv.COLOR_BGR2GRAY)
                    f.keypoints, f.descripts = sift.detectAndCompute(f.gray, None)

        for p, phi in enumerate(phis):
            for t, theta in enumerate(thetas):
                f_l = features[index(p, t, 0)]
                f_r = features[index(p, t, 1)]
                f = [f_l, f_r]

                if f_l.descripts is None or f_r.descripts is None: continue

                num_keypoints = len(f_r.keypoints)
                kp_indices = np.zeros((num_keypoints, 2), dtype=np.int) - 1
                kp_indices[:, 1] = np.arange(0, num_keypoints)

                matches = self._determine_matches(f_l.descripts, f_r.descripts, threshold)
                for m in matches:
                    kp_indices[m.trainIdx, 0] = m.queryIdx

                kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
                if kp_indices.shape[0] == 0: continue

                if self._debug.enable('depth_matches'):
                    plot = cv.drawMatches(f_l.gray, f_l.keypoints,
                                          f_r.gray, f_r.keypoints,
                                          matches, None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    fig.add_subplot(len(phis), len(thetas), fc).imshow(plot)
                    fc += 1

                rectilinear_pts = np.zeros(kp_indices.shape + (2,))
                for i in range(kp_indices.shape[0]):
                    for j in range(kp_indices.shape[1]):
                        rectilinear_pts[i][j] = f[j].keypoints[kp_indices[i,j]].pt

                polar_pts = np.zeros(rectilinear_pts.shape)
                for j in range(rectilinear_pts.shape[1]):
                    eq_pts = f[j].to_equ.GetEquirectangularPoints(rectilinear_pts[:,j])
                    polar_pts[:,j] = coordinates.eqr_to_polar(eq_pts, imgs[j].shape)
                    polar_pts[:,j] -= [0, math.pi]

                #_, inc = trim_outliers(polar_pts[:,0,0] - polar_pts[:,1,0], 1)
                #polar_pts = polar_pts[inc]

                all_polar_pts = np.concatenate([all_polar_pts, polar_pts])

        return all_polar_pts

    def _determine_matches(self, des_a, des_b, threshold):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des_a, des_b, k=2)

        def by_distance(e):
            return e.distance

        good_matches = []
        for ma in matches:
            if len(ma) == 2 and ma[0].distance < threshold * ma[1].distance:
                good_matches.append(ma[0])

        good_matches.sort(key=by_distance)
        return good_matches[:60]
