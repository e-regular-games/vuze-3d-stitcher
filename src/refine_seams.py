
import color_correction
import coordinates
import math
import numpy as np
import cv2 as cv
from transform import Transform
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from Equirec2Perspec import Equirectangular
from scipy.spatial import KDTree
from scipy.sparse.csgraph import shortest_path
from debug_utils import show_polar_plot
from debug_utils import show_polar_points
from depth_mesh import radius_compute
from depth_mesh import switch_axis
from depth_mesh import DepthMapperSlice
from depth_mesh import DepthMapper2D
from feature_matcher import FeatureMatcher2
from feature_matcher import FeatureMatcher4
from coordinates import to_1d

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

    def lens_to_head(self, img, p_0, dmap, alpha):
        dim = img.shape[0]
        R = self._calculate_R()
        plr = [0, 3*math.pi/2] + [1,-1]*coordinates.polar_points_3d((dim, dim))
        P = p_0 + coordinates.polar_to_cart(plr, dmap)
        d = np.sqrt(np.sum(P*P, axis=2))
        a = p_0 * np.ones(P.shape, np.float32)

        #phi = np.arccos(np.sum(a * P, axis=2) / np.sqrt(np.sum(a * P, axis=2)))
        qa = 1 + np.tan(alpha) * np.tan(alpha)
        qb = 2 * R
        qc = (R*R - d*d)
        x = (-qb + np.sqrt(qb*qb - 4*qa*qc)) / (2*qa)
        beta = np.arccos((R + x) / d)

        print(np.min(P[...,0]), np.max(P[...,0]), np.min(P[...,1]), np.max(P[...,1]), \
              np.min(P[...,2]), np.max(P[...,2]))

        plr_c = coordinates.cart_to_polar(P)
        plr_c[...,1] -= beta
        plr_c = plr_c.reshape((dim, dim, 2))

        print('phi', np.min(plr_c[...,0]), np.max(plr_c[...,0]))
        print('theta', np.min(plr_c[...,1]), np.max(plr_c[...,1]))

        img0 = get_middle(img)
        plr_c = [0, 3*math.pi/2] + [1,-1]*plr_c - [0,math.pi/2]
        eqr_c = coordinates.polar_to_eqr_3d(plr_c, img.shape)
        print('eqr_x', np.min(eqr_c[...,0]), np.max(eqr_c[...,0]))
        print('eqr_y', np.min(eqr_c[...,1]), np.max(eqr_c[...,1]))

        remapped = np.zeros(img0.shape, np.uint8)

        def bound(a, mn, mx):
            a[a<mn] = mn
            a[a>mx] = mx
            return a

        x_0 = bound(np.floor(eqr_c[...,0]).astype(np.int), 0, dim-1)
        x_0_w = (1 - (eqr_c[...,0] - x_0)).reshape((dim, dim, 1))
        x_1 = bound(np.floor(eqr_c[...,0] + 1).astype(np.int), 0, dim-1)
        x_1_w = (1 - (x_1 - eqr_c[...,0])).reshape((dim, dim, 1))
        y_0 = bound(np.floor(eqr_c[...,1]).astype(np.int), 0, dim-1)
        y_0_w = (1 - (eqr_c[...,1] - y_0)).reshape((dim, dim, 1))
        y_1 = bound(np.floor(eqr_c[...,1] + 1).astype(np.int), 0, dim-1)
        y_1_w = (1 - (y_1 - eqr_c[...,1])).reshape((dim, dim, 1))

        remapped[y_0,x_0] += np.floor(x_0_w * y_0_w * img0).astype(np.uint8)
        remapped[y_0,x_1] += np.floor(x_0_w * y_1_w * img0).astype(np.uint8)
        remapped[y_1,x_0] += np.floor(x_1_w * y_0_w * img0).astype(np.uint8)
        remapped[y_1,x_1] += np.floor(x_1_w * y_1_w * img0).astype(np.uint8)

        plt.figure()
        plt.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(cv.cvtColor(remapped, cv.COLOR_BGR2RGB))
        return remapped

    def _generate_path(self, matches, dmap):
        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2
        top = np.ones((1, 4, 2)) * [[[0, 3*math.pi/2]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi, 3*math.pi/2]]]
        m = np.concatenate([top, matches, bottom])
        dim = m.shape[0]
        sort_idx = np.argsort(m[:,0,0])
        m0_col = m[sort_idx,0:1]
        m0_row = m0_col.reshape((1, dim, 2))

        dtheta = m0_row[...,1] - m0_col[...,1]
        dphi = m0_row[...,0] - m0_col[...,0]

        valid = dphi > 0.0001

        def square_and_scale(cost):
            cost = cost / np.min(cost[cost>0])
            cost = cost * cost
            cost = (999 * (cost-1) / np.max(cost - 1)) + 1
            return cost

        D = 50
        phi_ff, phi_ii = np.meshgrid(m0_row[...,0], m0_row[...,0])
        theta_ff, theta_ii = np.meshgrid(m0_row[...,1], m0_row[...,1])

        rg = np.arange(D).reshape((1, 1, D, 1))
        path_at = np.zeros((dim, dim, D, 2), np.float32)
        path_at[...,0:1] = phi_ii.reshape((dim, dim, 1, 1)) \
            + rg * (phi_ff.reshape((dim, dim, 1, 1)) - phi_ii.reshape((dim, dim, 1, 1))) / (D-1)
        path_at[...,1:2] = theta_ii.reshape((dim, dim, 1, 1)) \
            + rg * (theta_ff.reshape((dim, dim, 1, 1)) - theta_ii.reshape((dim, dim, 1, 1))) / (D-1)

        path_r = dmap.eval(path_at.reshape((dim, D*dim, 2))).reshape((dim, dim, D))

        delta_position = path_at[...,1:,:] - path_at[...,:-1,:]
        delta_position = np.sqrt(np.sum(delta_position * delta_position, axis=-1))
        path_valid = np.all(delta_position > 0.00001, axis=-1)

        position_cost = square_and_scale(np.sum(delta_position, axis=-1))

        path_cart = coordinates.polar_to_cart(path_at, path_r)
        delta_cart = path_cart[...,1:,:] - path_cart[...,:-1,:]
        delta_cart = np.sqrt(np.sum(delta_cart * delta_cart, axis=-1))
        cart_cost = square_and_scale(np.sum(delta_cart, axis=-1))

        slope = np.zeros((dim, dim), np.float32)
        slope[path_valid] = np.sum(np.abs(path_r[...,1:] - path_r[...,:-1]) / delta_position, axis=-1)[path_valid]
        slope_cost = square_and_scale(slope)

        mat = 0.4*slope_cost + 0.3*cart_cost + 0.3*position_cost
        mat[np.logical_not(valid)] = 0

        if self._debug.enable('seam-path-cost'):
            f = plt.figure()
            ax = f.add_subplot(2, 2, 2)
            ax.set_title('Slope Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(slope_cost), vcenter=np.mean(slope_cost), vmax=np.max(slope_cost))
            pos = ax.imshow(slope_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 2, 3)
            ax.set_title('Position Cost')
            pos = ax.imshow(position_cost, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 2, 4)
            ax.set_title('Cartesian Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(cart_cost), vcenter=np.mean(cart_cost), vmax=np.max(cart_cost))
            pos = ax.imshow(cart_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 2, 1)
            ax.set_title('Combined Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(mat), vcenter=np.mean(mat), vmax=np.max(mat))
            pos = ax.imshow(mat, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

        rdist, rpred = shortest_path(mat, return_predecessors=True)

        top_idx = m0_col.shape[0]-1
        path = [top_idx]
        pred = rpred[0,top_idx]
        while pred >= 0:
            path.append(pred)
            pred = rpred[0, pred]

        path_points = np.flip(m[sort_idx][path], axis=0)
        path_points_r = dmap.eval(path_points[:,0]).reshape((path_points.shape[0], 1, 1))

        # the top and bottom points will have 0 radius, so replace with the next point
        path_points_r[0] = path_points_r[1]
        path_points_r[-1] = path_points_r[-2]

        path_points_r = path_points_r * np.ones(path_points.shape[0:2] + (1,))
        return np.concatenate([path_points, path_points_r], axis=-1)

        return path_points

    def _determine_side_matches(self):

        seam_imgs = self._images[-2:] + self._images
        seam_calibs = self.calibration[-2:] + self.calibration
        seam_paths = []
        dim = self._images[0].shape[0]
        depth_maps = []
        for i in range(0, 4, 2):
            slc = DepthMapperSlice(seam_imgs[i:i+4], [c.t for c in seam_calibs[i:i+4]], self._debug)
            dmap = slc.map()[0]
            seam_paths.append(self._generate_path(slc._matches, dmap))

            depth = DepthMapper2D(self._images[i], self._images[i+1], \
                                  seam_calibs[i].t, seam_calibs[i+1].t, \
                                  self._debug)
            maps = depth.map()
            depth_maps.append(maps[0])
            depth_maps.append(maps[1])

        t = Transform(self._debug) \
            .set_seams(seam_paths[0][:,0], seam_paths[1][:,0]) \
            .set_position(switch_axis(self.calibration[0].t)) \
            .set_depth(depth_maps[2])

        plr = coordinates.polar_points_3d((dim, dim))
        plr_c = t.apply(plr)
        img0 = get_middle(self._images[0])

        eqr_c = coordinates.polar_to_eqr_3d(plr_c, self._images[0].shape)
        print('eqr_x', np.min(eqr_c[...,0]), np.max(eqr_c[...,0]))
        print('eqr_y', np.min(eqr_c[...,1]), np.max(eqr_c[...,1]))

        remapped = np.zeros(img0.shape, np.uint8)

        def bound(a, mn, mx):
            a[a<mn] = mn
            a[a>mx] = mx
            return a

        x_0 = bound(np.floor(eqr_c[...,0]).astype(np.int), 0, dim-1)
        x_0_w = (1 - (eqr_c[...,0] - x_0)).reshape((dim, dim, 1))
        x_1 = bound(np.floor(eqr_c[...,0] + 1).astype(np.int), 0, dim-1)
        x_1_w = (1 - (x_1 - eqr_c[...,0])).reshape((dim, dim, 1))
        y_0 = bound(np.floor(eqr_c[...,1]).astype(np.int), 0, dim-1)
        y_0_w = (1 - (eqr_c[...,1] - y_0)).reshape((dim, dim, 1))
        y_1 = bound(np.floor(eqr_c[...,1] + 1).astype(np.int), 0, dim-1)
        y_1_w = (1 - (y_1 - eqr_c[...,1])).reshape((dim, dim, 1))

        remapped[y_0,x_0] += np.floor(x_0_w * y_0_w * img0).astype(np.uint8)
        remapped[y_0,x_1] += np.floor(x_0_w * y_1_w * img0).astype(np.uint8)
        remapped[y_1,x_0] += np.floor(x_1_w * y_0_w * img0).astype(np.uint8)
        remapped[y_1,x_1] += np.floor(x_1_w * y_1_w * img0).astype(np.uint8)

        plt.figure()
        plt.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(cv.cvtColor(remapped, cv.COLOR_BGR2RGB))

        plt.show()
        exit(1)

        depth_maps = []
        for i in range(0, 4, 2):
            depth = DepthMapper2D(self._images[i], self._images[i+1], \
                                  self.calibration[i].t, self.calibration[i+1].t, \
                                  self._debug)
            maps = depth.map()
            depth_maps.append(maps[0])
            depth_maps.append(maps[1])

        plt.show()
        exit(1)

        for m in depth_maps:
            f_scale = m < 3
            m[f_scale] = 2 * m[f_scale] / 3 + 1

        R = self._calculate_R()
        alpha = self._calculate_alpha(R, 0.064)
        print('alpha', alpha)

        a = [alpha, -alpha, alpha, -alpha]
        m = []
        for i in range(4):
            p = switch_axis(self.calibration[i].t)
            m.append(create_from_middle(self.lens_to_head(self._images[i], p, depth_maps[i], a[i])))

        matches = FeatureMatcher4([m[0], m[2]], [m[1], m[3]], self._debug).matches()
        print('adjusted', np.mean(np.std(matches, axis=-2), axis=0))

        matches = FeatureMatcher4([self._images[0], self._images[2]], \
                                  [self._images[1], self._images[3]], self._debug).matches()
        print('original', np.mean(np.std(matches, axis=-2), axis=0))

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
