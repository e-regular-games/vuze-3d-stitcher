#!/usr/bin/python

import coordinates
import math
import numpy as np
import cv2 as cv
import threading
from Equirec2Perspec import Equirectangular
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree


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

        self._d_eyes = 0.06
        self._d_center = 0.06

        self._features = None

    def generate(self):
        matches = self._features_determine()


        pass


    def _radius_compute(self, left, right):
        h_right = self._d_eyes / np.cos(right) / (1 - np.sin(right) * np.cos(left) / np.sin(left) / np.cos(right))
        print(left[:10], right[:10], h_right[:10])
        o = h_right * np.sin(right) + self._d_center
        a = h_right * np.cos(right) - self._d_eyes / 2
        r = np.sqrt(o * o + a * a)

        return r

    def _features_determine(self):
        all_matches_plr = np.zeros((0, 2, 2), dtype=np.float32)
        for i in range(0, len(self._images), 2):
            matches_plr = self._match_between_eyes(self._images[i], self._images[i+1], 0.75)
            matches_plr += [0, i / 2 * math.pi / 2]
            all_matches_plr = np.concatenate([all_matches_plr, matches_plr])

        all_matches_plr = all_matches_plr.reshape((all_matches_plr.shape[0], 4))
        tree = KDTree(all_matches_plr)
        overlapping = tree.query_radius(all_matches_plr, 0.0001)

        include = np.full((all_matches_plr.shape[0],), True, dtype=np.bool)
        for i, near in enumerate(overlapping):
            for o in near:
                if o != i:
                    include[o] = False

        if self._debug.verbose:
            print('overlapping features', np.count_nonzero(include), all_matches_plr.shape[0])

        all_matches_plr = all_matches_plr[include]
        all_matches_plr = all_matches_plr.reshape((all_matches_plr.shape[0], 2, 2))

        if self._debug.enable('depth_sphere'):
            self.show_polar_plot(all_matches_plr[:,0], all_matches_plr[:,1])

        r = self._radius_compute(all_matches_plr[:,0,1], all_matches_plr[:,1,1])

        left = np.concatenate([all_matches_plr[:,0], r.reshape((r.shape[0], 1))], axis=1)
        right = np.concatenate([all_matches_plr[:,1], r.reshape((r.shape[0], 1))], axis=1)

        if self._debug.enable('depth_points'):
            self.show_polar_points(left)
            self.show_polar_points(right)


    def _match_between_eyes(self, img_left, img_right, threshold):
        sift = cv.SIFT_create()
        fig = plt.figure() if self._debug.enable('depth_matches') else None
        fc = 1

        imgs = [img_left] + [img_right]
        thetas = [-45, 0, 45]
        phis = [0]
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

                num_keypoints = len(f_l.keypoints)
                kp_indices = np.zeros((num_keypoints, 2), dtype=np.int) - 1
                kp_indices[:, 0] = np.arange(0, num_keypoints)

                matches = self._determine_matches(f_r.descripts, f_l.descripts, threshold)
                for m in matches:
                    kp_indices[m.trainIdx, 1] = m.queryIdx

                kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
                if kp_indices.shape[0] == 0: continue

                if self._debug.enable('depth_matches'):
                    plot = cv.drawMatches(f_r.gray, f_r.keypoints,
                                          f_l.gray, f_l.keypoints,
                                          matches, None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    fig.add_subplot(3, 1, fc).imshow(plot)
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
        return good_matches


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

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim(-1, 1)
