
import coordinates
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Transform():
    def __init__(self, debug):
        self._debug = debug

    # coordinates are a NxMx2 matrix of polar coordinates (phi, theta)
    def eval(self, c):
        return c

class Transforms(Transform):
    """
    Transforms applied in order, ie the evaluated coordinates from the
    first transform are passed to the second. The result from the last is returned.
    """
    def __init__(self, arr, debug):
        super().__init__(debug)
        self._transforms = arr

    def eval(self, c):
        res = c
        for t in self._transforms:
            res = t.eval(res)
        return res

class TransformLinReg(Transform):
    """
    Use linear regression to adjust the coordinates.
    """
    def __init__(self, debug):
        super().__init__(debug)
        self._linreg = None
        self._offset = [0, 0]

    # linreg should be in terms of 2 variables phi and theta.
    def set_regression(self, linreg):
        self._linreg = linreg
        return self

    # indices is an array of integers, if not provided the default is on all indices
    def set_offset(self, offset):
        self._offset = offset
        return self

    def eval(self, c):
        if self._linreg is None:
            return c
        return self._linreg.evaluate(c - self._offset) + self._offset


class TransformDepth(Transform):
    """
    Determine transforms for image coordinates based on polar coordinates system
    and the desired values.
    Apply or reverse the transforms as desired based on the determined coefficients.
    """

    def __init__(self, debug):
        super().__init__(debug)
        self._p_0 = None
        self._shift = None
        self._center = None
        self._eye = 1

        # set, _R, _interocular and _alpha
        self.override_params(0.064, 0.06)

        self._debug = debug

    def override_params(self, R, interocular):
        self._R = R
        self._interocular = interocular
        self._alpha = math.asin(self._interocular / (2 * self._R))
        return self

    # e is either [0, 1] for [left, right] respecitvely
    # center is a cartesian coordinate, normally the same as position
    def set_eye(self, e, center):
        if e == 0:
            self._eye = 1
        elif e == 1:
            self._eye = -1

        self._center = center
        return self

    def set_position(self, p_0):
        self._p_0 = p_0.reshape((1, 3))
        return self

    def set_depth(self, dmap):
        self._depth = dmap
        return self

    def _apply(self, c):
        dim = c.shape[0]
        r = self._depth.eval(c)

        c_adj = [0, 3*math.pi/2] + [1,-1]*c;
        P = self._p_0 + coordinates.polar_to_cart(c_adj, r)

        rho = coordinates.cart_to_polar(self._center)[0,1]

        d = np.sqrt(np.sum(P*P, axis=-1))
        qa = 1 + np.tan(self._alpha) * np.tan(self._alpha)
        qb = 2 * self._R
        qc = (self._R*self._R - d*d)
        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        beta = np.arccos((self._R + x) / d)

        plr_c = coordinates.cart_to_polar(P)
        plr_c[...,1] = rho - (plr_c[...,1] + self._eye * beta)
        return plr_c

    # Expects c to be a NxMx2 3d matrix where the last dimension is (phi, theta)
    def eval(self, c):
        rho = coordinates.cart_to_polar(self._center)[0,1]

        p_1_plr = [0, rho] + [1,-1] * (c - [0, math.pi])
        p_1 = coordinates.polar_to_cart(p_1_plr, self._R)

        r = self._depth.eval(c)
        P = p_1 + coordinates.polar_to_cart(p_1_plr - [0, self._eye * self._alpha], r)

        return [0, 3*math.pi/2] + [1,-1] * coordinates.cart_to_polar(P - self._p_0)
