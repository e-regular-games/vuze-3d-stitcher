
import coordinates
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from linear_regression import LinearRegression

# Generic class to transform coordinates C to C'.
# The base class performs no transformation.
class Transform():
    def __init__(self, debug=None):
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
        res = 10
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

    def to_dict(self):
        return {'type': 'identity'}

    @staticmethod
    def from_dict(d, debug=None):
        if not 'type' in d:
            return Transform(debug)
        elif d['type'] == 'transforms':
            return Transforms.from_dict(d, debug)
        elif d['type'] == 'linreg':
            return TransformLinReg.from_dict(d, debug)
        elif d['type'] == 'depth':
            return TransformDepth.from_dict(d, debug)
        return Transform(debug)


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

    def to_dict(self):
        return {
            'type': 'transforms',
            'transforms': [t.to_dict() for t in self._transforms]
        }

    @staticmethod
    def from_dict(d, debug):
        if not 'type' in d or d['type'] != 'transforms':
            return Transform(debug)
        arr = [Transform.from_dict(t, debug) for t in d['transforms']]
        return Transforms(arr)

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

    def to_dict(self):
        r = {'type': 'linreg'}
        if self._forward is not None:
            r['forward'] = self._forward.to_dict()
        if self._reverse is not None:
            r['reverse'] = self._reverse.to_dict()
        r['offset'] = self._offset
        return r

    @staticmethod
    def from_dict(d, debug):
        if not 'type' in d or d['type'] != 'linreg':
            return Transform(debug)

        t = TransformLinReg(debug)
        t._offset = d['offset']
        if 'forward' in d:
            t._forward = LinearRegression().from_dict(d['forward'])
        if 'reverse' in d:
            t._reverse = LinearRegression().from_dict(d['reverse'])
        return t


class TransformDepth(Transform):
    """
    Determine transforms for image coordinates based on polar coordinates system
    and the desired values.
    Apply or reverse the transforms as desired based on the determined coefficients.
    """

    def __init__(self, debug):
        super().__init__(debug)
        self._p_0 = None
        self._eye = 1
        self._alpha = 0
        self._depth = None

        # overrides _alpha based on the
        self.set_interocular(0.064, 0.06)

        self._debug = debug

    def to_dict(self):
        d = {'type': 'depth'}
        d['p0'] = self._p_0.tolist()
        d['eye'] = self._eye
        d['alpha'] = self._alpha
        if self._depth is not None:
            d['depth'] = self._depth.to_dict()
        return d

    @staticmethod
    def from_dict(d, debug):
        if not 'type' in d or d['type'] != 'linreg':
            return Transform(debug)
        t = TransformDepth(debug)
        t._p_0 = np.array(self._p_0, np.float32)
        t._eye = d['eye']
        t._alpha = d['alpha']
        t._depth = DepthMap.from_dist(d['depth'])
        return t

    def set_interocular(self, R, interocular):
        self._alpha = math.asin(interocular / (2 * R))
        return self

    # e is either [0, 1] for [left, right] respecitvely
    # center is a cartesian coordinate, normally the same as position
    def set_eye(self, e):
        if e == 0:
            self._eye = 1
        elif e == 1:
            self._eye = -1
        return self

    def set_position(self, p_0):
        self._p_0 = p_0.reshape((1, 3))
        return self

    def set_depth(self, dmap):
        self._depth = dmap
        return self

    def forward(self, c):
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

        result = np.concatenate([phi, theta], axis=-1) + [0, math.pi]

        return result

    def reverse(self, c):
        p_0 = self._p_0[:,0:2]
        R = np.linalg.norm(p_0)

        r = self._depth.eval(c).reshape(c.shape[:-1] + (1,)) # TODO this should not be c...

        rho = coordinates.cart_to_polar(self._p_0)[0,1]
        c = [math.pi/2, rho] - (c - [0, math.pi])
        p_1 = np.zeros(c.shape[:-1] + (3, 1), np.float32)
        p_1[...,0,0] = R * np.cos(c[...,1])
        p_1[...,1,0] = R * np.sin(c[...,1])

        alpha = self._eye * -self._alpha
        R_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0], \
                            [math.sin(alpha), math.cos(alpha), 0], \
                            [0, 0, 1]], np.float32)

        # compute the direction of P from p_1 along the camera plane
        # normalize the direction since it will be used as a unit vector
        H_theta = R_alpha @ p_1
        H_theta = H_theta / np.linalg.norm(H_theta, axis=-2).reshape(c.shape[:-1] + (1, 1))

        R_H = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)

        # transform transposed to convert from the unit vectors for H_theta
        T_H_T = np.zeros(c.shape[:-1] + (3, 3), np.float32)
        T_H_T[...,0:3,0] = H_theta[...,0]
        T_H_T[...,0:3,1] = (R_H @ H_theta)[...,0]
        T_H_T[...,0:3,2] = [0, 0, 1]

        # R_phi = R_phi @ [1, 0, 0]
        R_phi = np.zeros(c.shape[:-1] + (3, 1), np.float32)
        R_phi[...,0,0] = np.cos(-c[...,0])
        R_phi[...,2,0] = -np.sin(-c[...,0])

        H_phi = T_H_T @ R_phi

        R_E = R
        C_1 = p_1 - R_E * H_theta

        # intersect with lens sphere
        p_0 = (self._p_0 * np.array([[1, 1, 0]], np.float32)).reshape((3, 1))
        offset = C_1 - p_0
        disc = np.sum(H_phi * offset, axis=-2) ** 2 - (np.linalg.norm(offset, axis=-2)**2 - r**2)
        d = -np.sum(H_phi * offset, axis=-2) + np.sqrt(disc)

        P = (C_1 + d.reshape((d.shape + (1,))) * H_phi) - p_0
        result = [0, 3*math.pi/2] + [1, -1] * coordinates.cart_to_polar(P.reshape(c.shape[:-1] + (3,)))

        return  result
