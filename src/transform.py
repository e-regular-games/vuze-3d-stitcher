
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def choose_closest(r, x):
    final = r[:,0]
    delta = np.absolute(r[:,0] - x)
    for i in range(1, r.shape[1]):
        pick = np.absolute(r[:,i] - x) < delta
        final[pick] = r[:,i][pick]
        delta[pick] = np.absolute(r[:,i] - x)[pick]
    return final

# where all inputs are np.array objects
# https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
def cubic_roots(a, b, c, d):
    p = c/a - b*b/(3*a*a)
    q = 2*b*b*b/(27*a*a*a) - b*c/(3*a*a) + d/a
    ds = q*q/4 + p*p*p/27

    D = np.ones(ds.shape)*b/(3*a)

    root = np.zeros((ds.shape[0],3))

    s = ds > 0
    if np.count_nonzero(s) > 0:
        root[s,:] = (np.cbrt(-q[s]/2 - np.sqrt(ds[s])) + np.cbrt(-q[s]/2 + np.sqrt(ds[s])) - D[s])[:,None] * np.ones((ds.shape[0],3))[s,:]

    m = np.logical_and(ds == 0, p != 0)
    if np.count_nonzero(m) > 0:
        root[m,0] = 3*q[m]/p[m]
        root[m,1:3] = -3*q[m]/(2*p[m])[:,None] * np.ones((ds.shape[0],2))[m,:]

    t = np.logical_and(ds < 0, p < 0)
    if np.count_nonzero(t) > 0:
        for k in range(3):
            root[t,k] = 2 * np.sqrt(-p[t]/3) * np.cos(1/3*np.arccos(3*q[t]/(2*p[t])*np.sqrt(-3/p[t])) - 2*math.pi*k/3) - D[t]

    return root

def quadratic_roots(a, b, c):
    C = b * b - 4 * a * c

    res = np.zeros((C.shape[0], 2))
    res[:,0] = (-b + np.sqrt(C)) / (2 * a)
    res[:,1] = (-b - np.sqrt(C)) / (2 * a)
    return res

def roots(c, x):
    if c.shape[1] == 4:
        rts = cubic_roots(c[:,3], c[:,2], c[:,1], c[:,0])
        return choose_closest(rts, x)
    elif c.shape[1] == 3:
        rts = quadratic_roots(c[:,2], c[:,1], c[:,0])
        return choose_closest(rts, x)
    elif c.shape[1] == 2:
        return -c[:,0] / c[:,1]

class Transform():
    """
    Determine transforms for image coordinates based on polar coordinates system
    and the desired values.
    Apply or reverse the transforms as desired based on the determined coefficients.
    """

    def __init__(self, debug):
        self.theta_coeffs_order = 3
        tc_cnt = (self.theta_coeffs_order + 1)
        self.theta_coeffs = np.zeros(tc_cnt * tc_cnt)

        self.phi_coeffs_order = 3
        pc_cnt = (self.phi_coeffs_order + 1)
        self.phi_coeffs = np.zeros(pc_cnt * pc_cnt)

        self.phi_split_coeffs_order = 3
        psc_cnt = (self.phi_split_coeffs_order + 1)
        self.phi_split_coeffs = np.zeros((psc_cnt * (psc_cnt-1), 2))

        self.phi_lr_order = 2
        plr_cnt = (self.phi_lr_order + 1)
        self.phi_lr_c = np.zeros(plr_cnt * plr_cnt)

        self._debug = debug
        if debug.enable('regression'):
            self.fig = plt.figure()

    def _apply(self, x1, x2, order, c):
        cnt = order + 1
        x = np.zeros((x1.shape[0], cnt * cnt))
        for t in range(cnt):
            for p in range(cnt):
                x[:,t*cnt + p] = np.power(x2, t) * np.power(x1, p)

        return x.dot(c)

    def _apply_zero(self, x1, x2, order, c):
        cnt = order + 1
        x = np.zeros((x1.shape[0], cnt * (cnt-1)))
        for t in range(cnt-1):
            for p in range(cnt):
                x[:,t*cnt + p] = np.power(x2, t) * np.power(x1, p)

        return x.dot(c) * x2

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
        return roots(C, x1f)

    # assumes x2 is fixed, and x1f is the value of x1 + k
    # where k is the result of applying the constants c
    # to a multivariate polynomial of x1,x2 with order.
    def _reverse_zero(self, x1f, x2, order, c):
        cnt = order + 1

        coeffs = c * np.ones((x1f.shape[0], c.shape[0]))
        coeffs[:,0] -= x1f / x2
        coeffs[:,1] += 1 / x2

        C = np.zeros((x1f.shape[0], cnt))
        for t in range(cnt-1):
            for p in range(cnt):
                C[:,p] += coeffs[:,t*cnt+p] * np.power(x2, t)

        # compute the original x1
        return roots(C, x1f)

    def _regression(self, x1, x2, order, y, subplot):
        cnt = order + 1

        if self._debug.enable('regression'):
            ax = self.fig.add_subplot(2, 4, subplot+1, projection='3d')
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

        err = y - self._apply(x1, x2, order, c)
        rev = x1 - self._reverse(self._apply(x1, x2, order, c) + x1, x2, order, c)

        if self._debug.verbose:
            print('constants:', c)
            print('error:', np.mean(err), np.std(err))
            print('reverse (expect 0,0):', np.mean(rev), np.std(rev))

        if self._debug.enable('regression'):
            ax = self.fig.add_subplot(2, 4, subplot+5, projection='3d')
            ax.plot3D(x1, x2, err, 'b+', markersize=1)
            plt.xlim([-math.pi/2, math.pi/2])
            plt.ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.1, 0.1)

        return c

    def _regression_zero(self, x1, x2, order, y, subplot):
        cnt = order + 1

        fig = None
        if self.display:
            ax = self.fig.add_subplot(2, 4, subplot+1, projection='3d')
            ax.plot3D(x1, x2, y, 'b+', markersize=1)
            plt.xlim([-math.pi/2, math.pi/2])
            plt.ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.2, 0.2)

        x = np.zeros((x1.shape[0], (cnt-1) * cnt))
        for t in range(cnt-1):
            for p in range(cnt):
                x[:,t*cnt + p] = np.power(x2, t) * np.power(x1, p)

        # QR decomposition
        Q, R = np.linalg.qr(x)
        c = np.linalg.inv(R).dot(np.transpose(Q)).dot(y / x2)

        err = y - self._apply_zero(x1, x2, order, c)
        rev = x1 - self._reverse_zero(self._apply_zero(x1, x2, order, c) + x1, x2, order, c)

        if self.verbose:
            print('constants:', c)
            print('error:', np.mean(err), np.std(err))
            print('reverse (expect 0,0):', np.mean(rev), np.std(rev))

        if self.display:
            ax = self.fig.add_subplot(2, 4, subplot+5, projection='3d')
            ax.plot3D(x1, x2, err, 'b+', markersize=1)
            plt.xlim([-math.pi/2, math.pi/2])
            plt.ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.1, 0.1)

        return c

    # matches is a np.array with phi_l, theta_l, phi_r, theta_r coordinate pairs
    # which are assumed to refer to the same points in each image.
    # use these pairs to calculate phi_lr_c coeffecients which cause
    # the phi values for l and r to meet at a mid-point
    def calculate_phi_lr_c(self, c_0, c_1):
        theta = c_0[:,1] - math.pi
        phi = c_0[:,0] - math.pi / 2
        diff = c_1[:,0] - c_0[:,0]

        self.phi_lr_c = self._regression(phi, theta, self.phi_lr_order, diff, 0)

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

        self.theta_coeffs = self._regression(theta, phi, self.theta_coeffs_order, diff, 1)

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
        self.phi_coeffs = self._regression(phi, theta, self.phi_coeffs_order, diff, 2)

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

    def calculate_phi_split_c(self, side, c_0, c_1):
        theta = c_0[:,1] - math.pi
        phi = c_0[:,0] - math.pi / 2
        diff = c_1[:,0] - c_0[:,0]
        self.phi_split_coeffs[:,side] = self._regression_zero(phi, theta, self.phi_split_coeffs_order, diff, 2 + side)

    def apply_phi_split_c(self, c):
        r = c.copy()
        theta = r[:,1] - math.pi
        phi = r[:,0] - math.pi / 2

        left = theta < 0
        right = theta > 0
        r[:,0][left] += self._apply_zero(phi[left], theta[left],
                                    self.phi_split_coeffs_order, self.phi_split_coeffs[:,0])
        r[:,0][right] += self._apply_zero(phi[right], theta[right],
                                     self.phi_split_coeffs_order, self.phi_split_coeffs[:,1])
        return r

    def reverse_phi_split_c(self, c):
        r = c.copy()
        phi = r[:,0] - math.pi / 2
        theta = r[:,1] - math.pi

        left = theta < 0
        right = theta > 0
        r[:,0][left] = self._reverse_zero(phi[left], theta[left],
                                     self.phi_split_coeffs_order,
                                     self.phi_split_coeffs[:,0]) + math.pi / 2
        r[:,0][right] = self._reverse_zero(phi[right], theta[right],
                                      self.phi_split_coeffs_order,
                                      self.phi_split_coeffs[:,1]) + math.pi / 2
        return r


    def apply(self, c):
        #r = self.apply_phi_lr_c(c)
        r = self.apply_theta_c(c)
        r = self.apply_phi_c(r)
        return r

    def reverse(self, c):
        r = self.reverse_phi_c(c)
        r = self.reverse_theta_c(r)
        #r = self.reverse_phi_lr_c(r)
        return r
