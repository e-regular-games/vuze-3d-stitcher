
import coordinates
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Generic class to transform coordinates C to C'.
# The base class performs no transformation.
class Transform():
    def __init__(self, debug):
        self._debug = debug

    # coordinates are a NxMx2 matrix of polar coordinates (phi, theta)
    # Coordinates are assumed to be in the transformed space C'.
    # returns coordinates in the space C.
    def reverse(self, c):
        return c

    # coordinates are a NxMx2 matrix of polar coordinates (phi, theta)
    # Coordinates are assumed to be in the non-transformed space C.
    # returns coordinates in the space C'.
    def forward(self, c):
        return c

    # Compute the C -> C' -> C and C' -> C -> C' with coordinates
    # between (0, pi/2) and (pi/2, 3pi/2) and print to the debug log
    # the mean squared error between the initial and final values.
    def validate(self):
        res = 20
        plr = coordinates.polar_points_3d((res, res))[1:-1,1:-1]

        ev = self.reverse(plr)
        ev0 = self.forward(ev)
        print('reverse->forward', np.mean((plr - ev0)**2, axis=(0, 1)))

        ap = self.forward(plr)
        ap0 = self.reverse(ap)
        print('forward->reverse', np.mean((plr - ap0)**2, axis=(0, 1)))


    def show(self, img):
        res = 1024
        plr = coordinates.polar_points_3d((res, res))
        plr = self.reverse(plr)
        eqr = coordinates.polar_to_eqr_3d(plr, (img.shape[0], 2*img.shape[0]))
        img1 = coordinates.eqr_interp_3d(eqr - [img.shape[0]/2, 0], img)

        plr = self.forward(plr)
        eqr = coordinates.polar_to_eqr_3d(plr, (img.shape[0], 2*img.shape[0]))
        img0 = coordinates.eqr_interp_3d(eqr - [img.shape[0]/2, 0], img)

        f = plt.figure()
        f.suptitle('Original')
        f.add_subplot(1, 1, 1).imshow(img)

        f = plt.figure()
        f.add_subplot(1, 1, 1).imshow(img1)
        f.suptitle('Transformed')

        f = plt.figure()
        f.add_subplot(1, 1, 1).imshow(img0)
        f.suptitle('Untransformed')

class Transforms(Transform):
    """
    Transforms applied in order, ie the evaluated coordinates from the
    first transform are passed to the second. The result from the last is returned.
    """
    def __init__(self, arr):
        super().__init__(None)
        self._transforms = arr

    def reverse(self, c):
        res = c
        for t in reversed(self._transforms):
            res = t.reverse(res)
        return res

    def forward(self, c):
        res = c
        for t in self._transforms:
            res = t.forward(res)
        return res

class TransformScale(Transform):
    def __init__(self, scale, debug):
        super().__init__(debug)
        self._scale = scale

    def reverse(self, c):
        return c / self._scale


    def forward(self, c):
        return c * self._scale

class TransformLinReg(Transform):
    """
    Use linear regression to adjust the coordinates.
    """
    def __init__(self, debug):
        super().__init__(debug)
        self._forward = None
        self._reverse = None
        self._offset = [0, 0]

    # linreg should be in terms of 2 variables phi and theta.
    def set_regression(self, forward, reverse):
        self._forward = forward
        self._reverse = reverse
        return self

    # indices is an array of integers, if not provided the default is on all indices
    def set_offset(self, offset):
        self._offset = offset
        return self

    def reverse(self, c):
        if self._reverse is None:
            return c

        return self._reverse.evaluate(c - self._offset) + self._offset

    def forward(self, c):
        if self._forward is None:
            return c

        return self._forward.evaluate(c - self._offset) + self._offset

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
        R = np.linalg.norm(self._p_0[:,0:2])

        c_adj = [0, 3*math.pi/2] + [1,-1]*c;
        P = self._p_0 + coordinates.polar_to_cart(c_adj, r)

        # assumes the camera plan is the x-y plane, ie z=0
        P_l = P * [1, 1, 0]
        rho = coordinates.cart_to_polar(self._p_0)[0,1]

        d = np.sqrt(np.sum(P_l*P_l, axis=-1)).reshape(c.shape[:-1] + (1,))
        qa = 1 + np.tan(self._alpha)**2
        qb = 2 * R
        qc = (R**2 - d*d)
        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        beta = np.arccos((R + x) / d)

        P_h_plr = coordinates.cart_to_polar(P_l)
        P_h_plr[...,1:2] += self._eye * beta
        theta = rho - (P_h_plr[...,1:2])

        P_h = coordinates.polar_to_cart(P_h_plr, R)

        P_hl = P_h - P_l
        n = np.sqrt(np.sum(P_hl*P_hl, axis=-1)).reshape(c.shape[:-1] + (1,))
        C_1 = P_hl * R / n + P_h

        above = (P[...,2:3] >= 0) + -1 * (P[...,2:3] < 0)
        phi = math.pi/2 - above * coordinates.angle(P - C_1, P_l - C_1)

        return np.concatenate([phi, theta], axis=-1) + [0, math.pi]


    def forward(self, c):
        shift = self._apply(np.array([[[math.pi/2, math.pi]]]))[0,0] - [math.pi/2, math.pi]
        return self._apply(c) - shift

    def reverse(self, c):
        p_0 = self._p_0[:,0:2]
        R = np.linalg.norm(p_0)

        r = self._depth.eval(c).reshape(c.shape[:-1] + (1,)) # TODO this should not be c...
        z = (r + R) * np.sin(math.pi/2 - c[...,0:1])
        y = np.sqrt((r + R)**2 - z**2) - R
        above = (c[...,0:1] < math.pi) + -1 * (c[...,0:1] >= math.pi)
        r = r * np.cos(np.arctan2(z, y))
        phi = math.pi/2 - above * np.arctan2(z, y)

        rho = coordinates.cart_to_polar(self._p_0 * [1, 1, 0])[0,1]

        p_1_l_plr = [math.pi/2, rho] + [0,-1] * (c - [0, math.pi])
        p_1_l = coordinates.polar_to_cart(p_1_l_plr, R)

        P_d = coordinates.polar_to_cart(p_1_l_plr - [0, self._eye * self._alpha], 1)

        m_1x = P_d[...,0:1]
        m_1y = P_d[...,1:2]

        P_0x = p_0[0,0]
        P_0y = p_0[0,1]

        P_1x = p_1_l[...,0:1]
        P_1y = p_1_l[...,1:2]

        # solve P_d + d_0 * m_1 == P_0 + r * m_2 for beta
        # where m_2 = [ cos(beta); sin(beta) ]
        beta = -2*np.arctan2((m_1x*r - np.sqrt(-P_0x**2*m_1y**2 + 2*P_0x*P_0y*m_1x*m_1y + 2*P_0x*P_1x*m_1y**2 - 2*P_0x*P_1y*m_1x*m_1y - P_0y**2*m_1x**2 - 2*P_0y*P_1x*m_1x*m_1y + 2*P_0y*P_1y*m_1x**2 - P_1x**2*m_1y**2 + 2*P_1x*P_1y*m_1x*m_1y - P_1y**2*m_1x**2 + m_1x**2*r**2 + m_1y**2*r**2)), (-P_0x*m_1y + P_0y*m_1x + P_1x*m_1y - P_1y*m_1x + m_1y*r))

        theta = 3*math.pi/2 - beta

        return np.concatenate([phi, theta], axis=-1)



    # Expects c to be a NxMx2 3d matrix where the last dimension is (phi, theta)
    def reverse_trig(self, c):

        p_0 = self._p_0 * [1, 1, 0]
        R = np.linalg.norm(p_0)

        r = self._depth.eval(c).reshape(c.shape[:-1] + (1,)) # TODO this should not be c...
        z = (r + R) * np.sin(math.pi/2 - c[...,0:1])
        y = np.sqrt((r + R)**2 - z**2) - R
        above = (c[...,0:1] < math.pi) + -1 * (c[...,0:1] >= math.pi)
        phi = math.pi/2 - above * np.arctan2(z, y)

        rho = coordinates.cart_to_polar(self._p_0)[0,1]

        p_1_l_plr = [math.pi/2, rho] + [0,-1] * (c - [0, math.pi])
        p_1_l = coordinates.polar_to_cart(p_1_l_plr, R)

        P_d = coordinates.polar_to_cart(p_1_l_plr - [0, self._eye * self._alpha], 1)

        # this math only works because we assume z=0 for all points.
        # ie we are operating in the x-y-axis plane
        angle_P_d = coordinates.angle(P_d, np.array([1, 0, 0], np.float32))

        rho_2 = coordinates.angle(p_1_l - p_0, np.array([1, 0, 0], np.float32))
        flip = np.logical_and((p_1_l - p_0)[...,1:2] < 0, rho_2 < math.pi/2)
        rho_2[flip] = -1 * rho_2[flip]
        rho_2[rho_2 > math.pi/2] = math.pi - rho_2[rho_2 > math.pi/2]

        rho_1 = angle_P_d - rho_2
        flip = rho_1 >= math.pi/2
        d = np.sqrt(np.sum((p_1_l - p_0)**2, axis=-1)).reshape(c.shape[:-1] + (1,))

        qa = 1 + np.tan(rho_1)**2
        qb = 2 * d
        qc = (d**2 - y**2)

        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        x[qa > 1000000] = 0

        beta = np.arccos((d + x) / y)
        beta[flip] = math.pi - beta[flip]

        theta = 3*math.pi/2 - (beta + rho_2)

        return np.concatenate([phi, theta], axis=-1)
