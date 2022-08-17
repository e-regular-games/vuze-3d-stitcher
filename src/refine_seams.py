
import coordinates
import transform
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular

class RefineSeams():
    """
    Refine the seams across all 8 images of the Vuze Camera.
    a, b are equirectangular images with 360 degrees
    and 0deg is the center of the image.
    a is on the left, b is on the right
    the seem is assumed to exist at 45deg in a, and -45deg in b.
    """
    def __init__(self, images, debug):
        self._images = images
        self._transforms = []
        self._debug = debug

        for i, img in enumerate(images):
            t = transform.Transform(debug)
            t.label = 'lens:' + str(i+1)
            self._transforms.append(t)

    def _match_between_eyes(self, imgs_left, imgs_right, threshold):
        sift = cv.SIFT_create()
        all_polar_pts = np.zeros((0, 8))

        fig = plt.figure() if self._debug.enable('matches') else None
        fc = 1
        imgs = imgs_left + imgs_right
        thetas = [45, -45, 45, -45]
        for lat in [60, 0, -60]:
            rl = [None] * 4
            inv = [None] * 4
            kp = [None] * 4
            des = [None] * 4
            gray = [None] * 4

            for i in range(4):
                rl[i], inv[i] = Equirectangular(imgs[i]).GetPerspective(60, thetas[i], lat, 1000, 1000);
                gray[i] = cv.cvtColor(rl[i], cv.COLOR_BGR2GRAY)
                kp[i], des[i] = sift.detectAndCompute(gray[i], None)

            matches = [None]*4 # matches[0] will always be None, ie. img0 vs. img0
            kp_indices = np.zeros((len(kp[0]), 4), dtype=np.int) - 1
            kp_indices[:, 0] = np.arange(0, len(kp[0]))
            for i in range(1, 4):
                matches[i] = self._determine_matches(des[i], des[0], threshold)
                for m in matches[i]:
                    kp_indices[m.trainIdx, i] = m.queryIdx

            kp_indices = kp_indices[(kp_indices != -1).all(axis=1), :]
            if kp_indices.shape[0] == 0:
                continue

            if self._debug.enable('matches'):
                for i in range(1, 4):
                    plot = cv.drawMatches(gray[0], kp[0], gray[i], kp[i], matches[i], None,
                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    fig.add_subplot(3, 3, fc).imshow(plot)
                    fc += 1

            rl_pts = np.zeros((kp_indices.shape[0], 2 * kp_indices.shape[1]))
            for i in range(kp_indices.shape[0]):
                for j in range(kp_indices.shape[1]):
                    rl_pts[i][2*j:2*j+2] = kp[j][kp_indices[i,j]].pt

            lat_polar_pts = np.zeros(rl_pts.shape)
            for j in range(kp_indices.shape[1]):
                eq_pts = inv[j].GetEquirectangularPoints(rl_pts[:,2*j:2*j+2])
                lat_polar_pts[:,2*j:2*j+2] = coordinates.eqr_to_polar(eq_pts, imgs[j].shape)
                lat_polar_pts[:,2*j+1] -= (thetas[j] - thetas[0]) * math.pi / 180

            all_polar_pts = np.concatenate([all_polar_pts, lat_polar_pts])

        return all_polar_pts

    def _determine_matches(self, des_a, des_b, threshold):
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

    # is in a N by 1 vector.
    def _trim_outliers(self, i, d):
        med = np.median(i)
        std = np.std(i)

        inc = np.logical_and(med - d*std < i, i < med + d*std)
        return i[inc], inc

    # i and f are N by 2 matrices.
    # the difference between i, and f is computed
    # any row whose diff is outside d0 and d1 standard deviations
    # from the mean will be removed.
    # @returns the filtered i and f, and the inclusion vector aligned to the original i and f.
    def _trim_outliers_by_diff(self, i, f, d0, d1):
        diff = i - f
        mn = np.mean(diff, axis=0)
        std = np.std(diff, axis=0)

        d = [d0, d1]
        inc = np.logical_and(mn - d*std < diff, diff < mn + d*std).all(axis=1)
        return i[inc], f[inc], inc

    # compute the alignment coefficients
    def align(self, match_thres=0.75, err_thres=0.0075):
        matches = [
            self._match_between_eyes([self._images[6], self._images[0]], [self._images[7], self._images[1]], match_thres),
            self._match_between_eyes([self._images[0], self._images[2]], [self._images[1], self._images[3]], match_thres),
            self._match_between_eyes([self._images[2], self._images[4]], [self._images[3], self._images[5]], match_thres),
            self._match_between_eyes([self._images[4], self._images[6]], [self._images[5], self._images[7]], match_thres)
        ]

        for m in range(4):
            match = matches[m]
            inc = np.full((match.shape[0]), True)
            for i in range(4):
                _, inc_i = self._trim_outliers(match[:,2*i+1], 1)
                inc = np.logical_and(inc, inc_i)
            matches[m] = match[inc]

        if self._debug.verbose:
            print('matches between eyes', matches[0].shape[0], matches[1].shape[0], matches[2].shape[0], matches[3].shape[0])

        targets = []
        for i in range(4):
            target = np.zeros((matches[i].shape[0], 8))
            target[:,0] = np.sum(matches[i][:,0:8:2], axis=1) * 0.25
            target[:,1] = np.sum(matches[i][:,[1,3]], axis=1) * 0.5
            target[:,2] = np.sum(matches[i][:,0:8:2], axis=1) * 0.25
            target[:,3] = np.sum(matches[i][:,[1,3]], axis=1) * 0.5
            target[:,4] = np.sum(matches[i][:,0:8:2], axis=1) * 0.25
            target[:,5] = np.sum(matches[i][:,[5,7]], axis=1) * 0.5
            target[:,6] = np.sum(matches[i][:,0:8:2], axis=1) * 0.25
            target[:,7] = np.sum(matches[i][:,[5,7]], axis=1) * 0.5
            targets.append(target)

        left = np.array([0, math.pi/2])
        within_error = [np.zeros((matches[i].shape[0], 4)) for i in range(4)]

        fig = plt.figure() if self._debug.enable('regression') else None
        for l in range(8):
            ll = int(l / 2)
            lr = int((ll + 1) % 4)
            lc = int(4 * (l % 2) + 2)
            rc = int(4 * (l % 2))

            ls = matches[ll].shape[0]
            i = np.concatenate([matches[ll][:,lc:lc+2] - left, matches[lr][:,rc:rc+2]])
            f = np.concatenate([targets[ll][:,lc:lc+2] - left, targets[lr][:,rc:rc+2]])
            i, f, inc = self._trim_outliers_by_diff(i, f, 2, 2)

            #self.show_polar_plot(i, f)
            t = self._transforms[l]
            t.calculate_theta_c(i, f)

            im = i.copy()
            im[:,1] = f[:,1]
            t.calculate_phi_c(im, f)

            adj = t.apply(i)
            err = np.zeros((inc.shape[0]))
            err[inc] = np.sqrt(np.sum((adj-f)*(adj-f), axis=1))

            if self._debug.enable('regression'):
                fig.add_subplot(3, 3, l+1).hist(err[inc], range=(0, 0.05))

            within_error[ll][:,2*(l%2)+1] = err[:ls] < err_thres
            within_error[lr][:,2*(l%2)] = err[ls:] < err_thres


        # use within error to find a seam of points
        seams = []
        for i in range(8):
            m = int(i /2)
            l = (i-2) % 8 # image left of the seam
            r = i # image right of the seam
            c = 4 * (i % 2) # column within the matched set of 4 images.
            closest = within_error[m].all(axis=1)
            target = targets[m][:,c:c+2][closest].copy()

            target = target[target[:,0] < math.pi]
            target = target[target[:,0].argsort()]
            target = np.concatenate([np.array([[0, target[0,1]]]), target, np.array([[math.pi, target[-1,1]]])])
            _, unique = np.unique(target[:,0], return_index=True)
            target = target[unique]

            seam = np.zeros(target.shape)
            seam[:,0] = target[:,0]
            seam[:,1] = (np.cumsum(target[:,1]) / np.arange(1, target.shape[0]+1)) - math.pi

            self.show_polar_plot(target, seam + [0, math.pi], 'seam:' + str(i))
            seams.append(seam)

        return seams


    def align_verticle(self):
        img_pairs = []
        for i in range(0, len(self._images), 2):
            img_pairs.append((self._images[i], self._images[i+1]))
            img_pairs.append((self._images[i+1], self._images[i]))

        for i, pair in enumerate(img_pairs):
            matches = np.zeros((0,4))
            for a in [-90, -30, 30, 90]:
                m = self._determine_matching_points(pair[0], pair[1], a, a, 0.5)
                matches = np.concatenate([matches, m])

            middle = (matches[:,0:2] + matches[:,2:4]) / 2

            self._transforms[i].calculate_phi_lr_c(matches[:,0:2], middle)
