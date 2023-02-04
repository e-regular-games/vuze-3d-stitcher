
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
from feature_matcher import FeatureMatcher2
from feature_matcher import FeatureMatcher4

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

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

        self._matches_side = None
        self._matches_seam = None

        for i, img in enumerate(images):
            t = Transform(debug)
            t.label = 'lens:' + str(i+1)
            self._transforms.append(t)

    # expects existing and matches to be Nx4x2
    # returns an Nx1 array of elements to keep
    def _filter_existing_matches(self, existing, matches):
        keep = np.ones((existing.shape[0],), bool)
        if matches is None:
            return keep

        for i in range(existing.shape[1]):
            t = KDTree(existing[:,i])
            dups_idx = t.query_ball_point(matches[:,i], 0.1)
            for d in dups_idx:
                keep[d] = 0
        self._debug.log('existing kept matches:', np.count_nonzero(keep), 'of', existing.shape[0])
        return keep

    def _calculate_R(self):
        p = np.zeros((len(self.calibration), 3), np.float32)
        for i in range(len(self.calibration)):
            p[i] = switch_axis(self.calibration[i].t)

        R = np.mean(np.sqrt(np.sum(p*p, axis=1)))
        print('R', R)
        return R

    def _calculate_alpha(self, R, interocular):
        return math.asin(interocular / (2 * R))


    # plr is a Nx3 vector (phi, theta, radius) of known points
    # alpha is offset angle of the eye based on the interocular distance
    def _project_to_eye_sphere(self, img, plr, p_0, alpha):
        pass

    def _determine_side_matches(self):
        matches = [
            FeatureMatcher2(self._images[0], self._images[1], self._debug).matches()
            #FeatureMatcher2(self._images[2], self._images[3], self._debug).matches(),
            #FeatureMatcher2(self._images[4], self._images[5], self._debug).matches(),
            #FeatureMatcher2(self._images[6], self._images[7], self._debug).matches()
        ]

        print('matches', matches[0].shape[0])
        r, dd = radius_compute(matches[0][:,0], matches[0][:,1], \
                               switch_axis(self.calibration[0].t), \
                               switch_axis(self.calibration[1].t))

        R = self._calculate_R()
        alpha = self._calculate_alpha(R, 0.064)
        print('alpha', alpha)

        show_polar_points(matches[0][:,0])
        dim = self._images[0].shape[0]

        m_r = np.zeros((dim*dim,), np.float32)
        m_r_mask = np.zeros((dim*dim,), np.float32)
        eqr = np.round(coordinates.polar_to_eqr(matches[0][:,0] - [0, math.pi/2], \
                                                self._images[0].shape))
        eqr_1d = eqr[:,1] * dim + (eqr[:,0])
        m_r_mask[eqr_1d.astype(int)] = 1
        m_r[eqr_1d.astype(int)] = r[:,0]
        m_r_mask = m_r_mask.reshape((dim,dim))
        m_r = m_r.reshape((dim,dim))
        print('r', np.min(m_r), np.max(m_r))

        img0 = get_middle(self._images[0])
        img0[m_r_mask == 1] = [0, 0, 255]

        layers = [5, 9, 17, 33, 65]
        for s in layers:
            next_mask = cv.filter2D(m_r_mask, -1, np.ones((s, s)))
            next_r = cv.filter2D(m_r, -1, np.ones((s, s)))
            count = next_mask.copy()
            count[count < 1] = 1
            next_r = next_r / count
            next_r[m_r_mask == 1] = m_r[m_r_mask == 1]
            m_r = next_r
            next_mask[next_mask < 0.5] = 0
            next_mask[next_mask >= 0.5] = 1
            m_r_mask = next_mask

        img = np.ones((dim, dim, 3), np.uint8)
        m_r[m_r < 0] = 0
        print('r', np.min(m_r), np.max(m_r))
        img_r = 255 * np.sqrt(m_r / np.max(m_r))
        img *= np.round(img_r).astype(np.uint8).reshape((dim,dim,1))

        plt.figure()
        plt.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(cv.cvtColor(get_middle(self._images[1]), cv.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(img)
        plt.show()
        exit(1)


        print('closest', closest_index.shape)
        plr_r = 10 * np.ones((plr.shape[0],1), np.float32)
        for i, c in enumerate(closest_index):
            if c.shape[0] == 0:
                continue
            d = plr[i] * np.ones((c.shape[0], 2), np.float32) - plr[c]
            d = np.sqrt(np.sum(d*d, axis=1))
            w = 1 - d / np.sum(d)
            plr_r[i] = np.sum(r[c] * w)

        p_0 = switch_axis(self.calibration[0].t)
        P = p_0 + coordinates.polar_to_cart(plr, plr_r)
        d = np.sqrt(np.sum(P*P, axis=1))
        a = p_0 * np.ones(P.shape, np.float32)
        b = P
        phi = np.arccos(np.matmul(a, b) / np.sqrt(np.sum(a * b, axis=1)))
        qa = 1 + np.tan(alpha) * np.tan(alpha)
        qb = 2 * R
        qc = (R*R - d*d)
        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        beta = np.arccos((R + x) / d)

        print('beta')

        plr_c = coordinates.cart_to_polar(P)
        plr_c[:,1] -= beta
        plr_c = plr_c.reshape((dim, dim, 2))

        t = KDTree(plr_c)
        closest = t.query_ball_point(plr, 0.1, return_sorted=True, workers=4)
        remapped = np.zeros((dim, dim, 3), np.uint8)
        colors0 = get_middle(self._images[0]).reshape((dim*dim, 3))
        for i, c in closest:
            if c.shape[0] == 0:
                continue
            d = plr[i] * np.ones((c.shape[0], 2), np.float32) - plr_c[c]
            d = np.sqrt(np.sum(d*d, axis=1))
            w = 1 - d / np.sum(d)
            remapped[i] = np.sum(colors0[c] * w, axis=0)

        remapped = create_from_middle(remapped.reshape((dim, dim, 3)))

        plt.figure()
        plt.imshow(self._images[0])
        plt.figure()
        plt.imshow(remapped)
        plt.show()
        exit(1)

    def _determine_seam_matches(self):

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

        if self._matches is not None:
            for i in range(4):
                keep = self._filter_existing_matches(self._matches[i], matches[i])
                matches[i] = np.concatenate([matches[i], self._matches[i][keep]])

        self._matches = matches
        return matches

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
        self._determine_side_matches()

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
