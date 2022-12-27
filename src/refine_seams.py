
import color_correction
import coordinates
import math
import numpy as np
import cv2 as cv
from transform import Transform
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
        self._feature_points = []
        self._debug = debug

        self.seam_points = 50
        self.seam_window = 10

        self._targets = None
        self._matches = None
        self._seams = None

        for i, img in enumerate(images):
            t = Transform(debug)
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
                if des[i] is None or des[0] is None: continue
                matches[i] = self._determine_matches(des[i], des[0], threshold)
                for m in matches[i]:
                    kp_indices[m.trainIdx, i] = m.queryIdx

            kp_indices = kp_indices[(kp_indices != -1).all(axis=1), :]
            if kp_indices.shape[0] == 0:
                continue

            if self._debug.enable('matches'):
                for i in range(1, 4):
                    plot = cv.drawMatches(gray[i], kp[i], gray[0], kp[0], matches[i], None,
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

    # is in a N by 1 vector.
    def _trim_outliers(self, i, d):
        m = np.mean(i)
        std = np.std(i)

        inc = np.logical_and(m - d*std < i, i < m + d*std)
        return i[inc], inc

    def _determine_matches_and_targets(self, match_thres):
        matches = [
            self._match_between_eyes([self._images[6], self._images[0]], [self._images[7], self._images[1]], match_thres),
            self._match_between_eyes([self._images[0], self._images[2]], [self._images[1], self._images[3]], match_thres),
            self._match_between_eyes([self._images[2], self._images[4]], [self._images[3], self._images[5]], match_thres),
            self._match_between_eyes([self._images[4], self._images[6]], [self._images[5], self._images[7]], match_thres)
        ]

        shist_pre = plt.figure() if self._debug.enable('sanitize') else None
        for m in range(4):
            match = matches[m]
            inc = np.full((match.shape[0]), True)
            for i in range(4):
                if self._debug.enable('sanitize'):
                    shist_pre.add_subplot(4, 4, 4*m+i+1).hist(match[:,2*i+1])
                _, inc_i = self._trim_outliers(match[:,2*i+1], 5)
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

        if self._matches is not None and self._targets is not None:
            for i in range(4):
                matches[i] = np.concatenate([matches[i], self._matches[i]])
                targets[i] = np.concatenate([targets[i], self._targets[i]])

        self._matches = matches
        self._targets = targets
        return matches, targets

    def _compute_transforms(self, matches, targets, err_thres):
        left = np.array([0, math.pi/2])
        within_error = [np.zeros((matches[i].shape[0], 4)) for i in range(4)]

        rhist = plt.figure() if self._debug.enable('regression_hist') else None
        shist = plt.figure() if self._debug.enable('sanitize') else None
        for l in range(8):
            ll = int(l / 2)
            lr = int((ll + 1) % 4)
            lc = int(4 * (l % 2) + 2)
            rc = int(4 * (l % 2))

            ls = matches[ll].shape[0]
            li, lf, linc = self._trim_outliers_by_diff(matches[ll][:,lc:lc+2], targets[ll][:,lc:lc+2], 2, 2)
            ri, rf, rinc = self._trim_outliers_by_diff(matches[lr][:,rc:rc+2], targets[lr][:,rc:rc+2], 2, 2)
            i = np.concatenate([li - left, ri])
            f = np.concatenate([lf - left, rf])
            inc = np.concatenate([linc, rinc])

            if self._debug.enable('sanitize'):
                diff = i - f
                shist.add_subplot(2, 8, l+1).hist(diff[:,0])
                shist.add_subplot(2, 8, 8+l+1).hist(diff[:,1])

            #self.show_polar_plot(i, f)
            t = self._transforms[l]
            t.calculate_phi_c(i, f)

            im = i.copy()
            im[:,0] = f[:,0]
            t.calculate_theta_c(im, f)

            adj = t.apply(i)
            err = np.zeros((inc.shape[0]))
            err[inc] = np.sqrt(np.sum((adj-f)*(adj-f), axis=1))

            if self._debug.enable('regression_hist'):
                rhist.add_subplot(3, 3, l+1).hist(err[inc], range=(0, 0.02))

            within_error[ll][:,2*(l%2)+1] = np.logical_and(err[:ls] < err_thres, inc[:ls])
            within_error[lr][:,2*(l%2)] = np.logical_and(err[ls:] < err_thres, inc[ls:])

        return within_error


    def _compute_seams(self, matches):
        # use within error to find a seam of points
        cphi = self.seam_points
        wphi = self.seam_window
        seams = []
        for i in range(8):
            m = int(i /2)
            l = (i-2) % 8 # image left of the seam
            r = i # image right of the seam
            c = 4 * (i % 2) # column within the matched set of 4 images.
            target = self._transforms[l].apply(matches[m][:,c:c+2])

            if target.shape[0] == 0:
                seams.append([])
                if self._debug.verbose:
                    print('invalid-seam:', i, matches[m].shape)
                continue

            target = target[target[:,0] < math.pi]
            target = target[target[:,0].argsort()]
            target = np.concatenate([np.array([[0, target[0,1]]]), target, np.array([[math.pi, target[-1,1]]])])
            _, unique = np.unique(target[:,0], return_index=True)
            target = target[unique]

            seam = np.zeros((cphi, 2))
            seam_valid = np.full((cphi), True)
            for p in range(cphi):
                lower = p*math.pi / cphi - math.pi / wphi
                upper = (p+1)*math.pi / cphi + math.pi / wphi
                in_range = np.logical_and(target[:,0] >= lower, target[:,0] <= upper)
                if np.count_nonzero(in_range) == 0:
                    seam_valid[p] = False
                else:
                    seam[p,0] = p*math.pi / cphi
                    seam[p,1] = np.mean(target[in_range,1])

            seam = seam[seam_valid,:] - [0, math.pi]
            seam = np.concatenate([seam, [[math.pi, seam[-1,1]]]])
            if self._debug.enable('seams'):
                self.show_polar_plot(target, seam + [0, math.pi], 'seam:' + str(i))

            if self._debug.verbose:
                print('seam:', i, round(np.mean(seam[:,1]), 3))

            seams.append(seam)

        self._seams = seams
        return seams

    def matches(self, match_thres=0.75, err_thres=0.0075):
        matches, targets = self._determine_matches_and_targets(match_thres)
        within_error = self._compute_transforms(matches, targets, err_thres)

        matches = []
        targets = []
        for i in range(len(self._matches)):
            closest = within_error[i].all(axis=1)
            matches.append(self._matches[i][closest])
            targets.append(self._targets[i][closest])

        return matches

    # compute the alignment coefficients
    def align(self, match_thres=0.75, err_thres=0.0075):
        matches = self.matches(match_thres, err_thres)
        seams = self._compute_seams(matches)
        return seams, matches

    def to_dict(self):
        d = {
            'seams': [],
            'transforms': [],
            'matches': [],
            'targets': []
        }

        for s in self._seams:
            d['seams'].append(s.tolist())

        for t in self._transforms:
            d['transforms'].append({
                'phiOrder': t.phi_coeffs_order,
                'thetaOrder': t.theta_coeffs_order,
                'phiCoeffs': t.phi_coeffs.tolist(),
                'thetaCoeffs': t.theta_coeffs.tolist()
            })

        for m in self._matches:
            d['matches'].append(m.tolist())

        for t in self._targets:
            d['targets'].append(t.tolist())

        return d

    def from_dict(self, d):
        self._seams = []
        for s in d['seams']:
            self._seams.append(np.array(s))

        self._transforms = []
        for s in d['transforms']:
            t = Transform(self._debug)
            t.phi_coeffs_order = s['phiOrder']
            t.theta_coeffs_order = s['thetaOrder']
            t.phi_coeffs = np.array(s['phiCoeffs'])
            t.theta_coeffs = np.array(s['thetaCoeffs'])
            self._transforms.append(t)

        self._matches = []
        for m in d['matches']:
            self._matches.append(np.array(m))

        self._targets = []
        for t in d['targets']:
            self._targets.append(np.array(t))

        return self
