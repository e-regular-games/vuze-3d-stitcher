#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
import threading
from Equirec2Perspec import Equirectangular
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree


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

class DepthMesh():

    def __init__(self, images, debug):
        self._images = images
        self._debug = debug

        # the position of the camera assuming the ground under
        # the camera is 0,0,0. The center of the camera is at (0,0,_Z)
        self._X = 0.03
        self._Y = 0.06
        self._Z = 1.7

        self._d_close = 1.8
        self._d_far = 20

        self._features = None

    def generate(self):
        matches = self._features_determine()
        pass


    # the left and right theta values as 1d vectors
    def _radius_compute(self, left, right):
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

        p_l = np.array([[-self._X, self._Y, self._Z]])
        p_r = np.array([[self._X, self._Y, self._Z]])

        # v = p + r * m
        # find the point on v_l and v_r such that the points are closest
        # r_l and r_r are the radius along each line that results in the closest point
        # if the point is r_l = r_r = 0, ignore the point,
        m_d = np.cross(m_r, m_l)
        a_d = m_d[:,0:1]
        b_d = m_d[:,1:2]
        c_d = m_d[:,2:3]

        A = (b_l*c_r - c_l*b_r) / (c_d*b_r - b_d*c_r)
        r_l = (2 * self._X / (a_l + A*a_d - c_l*a_r/c_r - A*c_d*a_r/c_r)).reshape((n,1))
        r_r = r_l*c_l/c_r + r_l*A*c_d/c_r
        r_d = A * r_l

        v_l = p_l + r_l * m_l
        v_r = p_r + r_r * m_r

        d = v_l - v_r
        d = np.sqrt(np.sum(d*d, axis=1))
        print('r_d', np.mean(r_d), np.min(d), np.max(d))
        print('d', np.mean(d), np.min(d), np.max(d))

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

        C = np.array([0, 0, self._Z])
        r = np.sqrt(np.sum((v - C) * (v - C), axis=1))
        print('r', np.min(r), np.max(r))
        return v


    def _features_determine(self):
        points = np.zeros((0, 3), dtype=np.float32)
        for i in range(0, 2, 2):
            matches_plr = self._match_between_eyes(self._images[i], self._images[i+1], 0.75)

            p = self._radius_compute(matches_plr[:,0], matches_plr[:,1])
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
        return good_matches[:10]


    def show_polar_plot(self, polar_a, polar_b, label=None):
        plt.figure(label) if label is not None else plt.figure()
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

    def show_polar_points(self, polar, label=None):
        plt.figure(label) if label is not None else plt.figure()
        xa = polar[..., 2] * np.sin(polar[..., 0]) * np.cos(polar[..., 1])
        ya = polar[..., 2] * np.sin(polar[..., 0]) * np.sin(polar[..., 1])
        za = polar[..., 2] * np.cos(polar[..., 0])

        ax = plt.axes(projection ='3d')
        ax.plot3D(xa, ya, za, 'bo', markersize=1)

        plt.xlim([-self._d_far, self._d_far])
        plt.ylim([-self._d_far, self._d_far])
        ax.set_zlim(-self._d_far, self._d_far)
