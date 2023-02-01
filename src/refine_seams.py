
import color_correction
import coordinates
import math
import numpy as np
import cv2 as cv
from transform import Transform
from matplotlib import pyplot as plt
from Equirec2Perspec import Equirectangular
from scipy.spatial import KDTree
from debug_utils import show_polar_plot
from debug_utils import show_polar_points
from depth_mesh import radius_compute
from depth_mesh import switch_axis
from feature_matcher import FeatureMatcher4

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
        self.seam_window = 20
        self.calibration = None

        self._targets = None
        self._matches = None
        self._seams = None

        for i, img in enumerate(images):
            t = Transform(debug)
            t.label = 'lens:' + str(i+1)
            self._transforms.append(t)


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

    # expects existing and matches to be Nx8
    # returns an Nx1 array of elements to keep
    def _filter_existing_matches(self, existing, matches):
        keep = np.ones((existing.shape[0],), bool)
        for i in range(0, 8, 2):
            t = KDTree(existing[:,i:i+2])
            dups_idx = t.query_ball_point(matches[:,i:i+2], 0.1)
            for d in dups_idx:
                keep[d] = 0
        self._debug.log('existing kept matches:', np.count_nonzero(keep), 'of', existing.shape[0])
        return keep

    def _determine_matches_and_targets(self, match_thres):

        matches = [
            FeatureMatcher4([self._images[6], self._images[0]], \
                            [self._images[7], self._images[1]], self._debug).matches(),
            FeatureMatcher4([self._images[0], self._images[2]], \
                            [self._images[1], self._images[3]], self._debug).matches(),
            FeatureMatcher4([self._images[2], self._images[4]], \
                            [self._images[3], self._images[5]], self._debug).matches(),
            FeatureMatcher4([self._images[4], self._images[6]], \
                            [self._images[5], self._images[7]], self._debug).matches()
        ]

        for i, m in enumerate(matches):
            if m is None:
                print('empty matches', i)

        self._debug.log('matches between eyes', matches[0].shape[0], matches[1].shape[0], matches[2].shape[0], matches[3].shape[0])

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
                keep = self._filter_existing_matches(self._matches[i], matches[i])
                matches[i] = np.concatenate([matches[i], self._matches[i][keep]])
                targets[i] = np.concatenate([targets[i], self._targets[i][keep]])

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
                print('invalid-seam:', i, matches[m].shape)
                continue

            target = target[target[:,0] < math.pi]
            target = target[target[:,0].argsort()]
            target = np.concatenate([np.array([[0, target[0,1]]]), target, np.array([[math.pi, target[-1,1]]])])
            #_, unique = np.unique(target[:,0], return_index=True)
            #target = target[unique]

            density_tree = KDTree(target)
            densities = np.array(density_tree.query_ball_point(target, 0.2, return_length=True))

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
                    idx = np.argmax(densities[in_range])
                    seam[p,1] = target[in_range,1][idx]

            seam = seam[seam_valid,:] - [0, math.pi]
            seam = np.concatenate([seam, [[math.pi, seam[-1,1]]]])
            if self._debug.enable('seams'):
                show_polar_plot(target, seam + [0, math.pi], 'seam:' + str(i))

            self._debug.log('seam:', i, round(np.mean(seam[:,1]), 3))

            seams.append(seam)

        self._seams = seams
        return seams

    def depth_at_seam(self, m, i):
        ll = [0, 3*math.pi/2] + [1,-1]*m[:30,0:2]
        lr = [0, 3*math.pi/2] + [1,-1]*m[:30,2:4]
        rl = [0, 3*math.pi/2] + [1,-1]*m[:30,4:6]
        rr = [0, 3*math.pi/2] + [1,-1]*m[:30,6:8]

        pll = switch_axis(self.calibration[(2*i-2) % 8].t)
        plr = switch_axis(self.calibration[(2*i-1) % 8].t)
        prl = np.matmul(switch_axis(self.calibration[2*i].t), \
                        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
        prr = np.matmul(switch_axis(self.calibration[2*i+1].t), \
                        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))

        print(pll, plr, prl, prr)
        print(np.concatenate([ll, lr, rl, rr], axis=1))
        ld, ldd = radius_compute(ll, lr, pll, plr)
        cd, cdd = radius_compute(ll, rr, pll, prr)
        dd, ddd = radius_compute(ll, rl, pll, prl)
        ed, edd = radius_compute(lr, rr, plr, prr)
        fd, fdd = radius_compute(lr, rl, plr, prl)
        rd, rdd = radius_compute(rl, rr, prl, prr)

        r = np.concatenate([ld, rd, cd, dd, ed, fd], axis=1)
        d = np.concatenate([ldd, rdd, cdd, ddd, edd, fdd], axis=1)

        idx = np.argmin(np.abs(d), axis=1) + np.arange(r.shape[0])*r.shape[1]
        r = r.flatten()[idx].reshape(r.shape[0], 1)
        pts = np.concatenate([ll, r], axis=1)
        print(r)
        print(np.sum(d*d, axis=1))
        # show_polar_points(pts)


    def matches(self, match_thres=0.075, err_thres=0.0075):
        matches, targets = self._determine_matches_and_targets(match_thres)

        if self._debug.enable('refine-depth'):
            for i in range(4):
                self.depth_at_seam(matches[i], i)
                exit(1)

        within_error = self._compute_transforms(matches, targets, err_thres)

        matches = []
        targets = []
        for i in range(len(self._matches)):
            closest = within_error[i].all(axis=1)
            matches.append(self._matches[i][closest])
            targets.append(self._targets[i][closest])

        self._debug.log('matches post transform', matches[0].shape[0], matches[1].shape[0], matches[2].shape[0], matches[3].shape[0])

        return matches

    # compute the alignment coefficients
    def align(self, match_thres=0.075, err_thres=0.0075):
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
