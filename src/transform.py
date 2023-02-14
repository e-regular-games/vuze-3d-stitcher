
import coordinates
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class Transform():
    """
    Determine transforms for image coordinates based on polar coordinates system
    and the desired values.
    Apply or reverse the transforms as desired based on the determined coefficients.
    """

    def __init__(self, debug):
        self._left = None
        self._right = None
        self._p_0 = None
        self._shift = None

        # set, _R, _interocular and _alpha
        self.override_params(0.064, 0.06)

        self._debug = debug

    def override_params(self, R, interocular):
        self._R = R
        self._interocular = interocular
        self._alpha = math.asin(self._interocular / (2 * self._R))
        return self

    # left, right are Nx3 matrices with (phi, theta, r)
    def set_seams(self, left, right):
        self._left = left
        self._right = right
        return self

    def set_position(self, p_0):
        self._p_0 = p_0.reshape((1, 3))
        return self

    def set_depth(self, dmap):
        self._depth = dmap
        return self

    def _apply(self, c):
        dim = c.shape[0]

        #phi = c[:,0,0]
        #theta = c[...,1]
        #ls_theta = (coordinates.seam_intersect(self._left, phi) - math.pi/2).reshape((dim, 1))
        #ls_r = coordinates.seam_intersect(self._left[:,[0,2]], phi).reshape((dim, 1))
        #rs_theta = coordinates.seam_intersect(self._right, phi).reshape((dim, 1))
        #rs_r = coordinates.seam_intersect(self._right[:,[0,2]], phi).reshape((dim, 1))
        #drdt = ((rs_r - ls_r) / (rs_theta - ls_theta))
        #r = (theta - ls_theta) * drdt + ls_r
        #print('r', r)

        r = self._depth.eval(c)

        c_adj = [0, 3*math.pi/2] + [1,-1]*c;
        P = self._p_0 + coordinates.polar_to_cart(c_adj, r)

        #print('P', P)

        rho = coordinates.cart_to_polar(self._p_0)[0,1]
        #print('rho', rho)

        d = np.sqrt(np.sum(P*P, axis=-1))
        #print('d', d)
        qa = 1 + np.tan(self._alpha) * np.tan(self._alpha)
        qb = 2 * self._R
        qc = (self._R*self._R - d*d)
        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        beta = np.arccos((self._R + x) / d)

        #print('beta', beta)

        plr_c = coordinates.cart_to_polar(P)
        #print('plr_c', plr_c)

        plr_c[...,1] = rho - (plr_c[...,1] + beta)

        print('phi', np.min(plr_c[...,0]), np.max(plr_c[...,0]))
        print('theta', np.min(plr_c[...,1]), np.max(plr_c[...,1]))
        return plr_c

    # Expects c to be a NxMx2 3d matrix where the last dimension is (phi, theta)
    # returns an NxMx2 matrix of new (phi, theta)
    def apply(self, c):
        if self._shift  is None:
            self._shift = self._apply(np.array([[[math.pi/2, math.pi]]]))[0,0,1]

        print(self._shift)

        return self._apply(c) + [0, math.pi/2 - self._shift]

    def reverse(self, c):
        r = self.reverse_theta_c(c)
        r = self.reverse_phi_c(r)
        return r
