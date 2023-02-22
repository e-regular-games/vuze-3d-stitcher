
import color_correction
import coordinates
import math
import numpy as np
import cv2 as cv
import debug_utils
import linear_regression
from transform import TransformDepth
from transform import Transform
from transform import Transforms
from transform import TransformLinReg
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
from depth_mesh import DepthMap
from feature_matcher import FeatureMatcher2
from feature_matcher import FeatureMatcher4
from coordinates import to_1d
from linear_regression import LinearRegression
from linear_regression import trim_outliers_by_diff

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

    def _calculate_R(self):
        p = np.zeros((len(self.calibration), 3), np.float32)
        for i in range(len(self.calibration)):
            p[i] = switch_axis(self.calibration[i].t)

        R = np.mean(np.sqrt(np.sum(p*p, axis=1)))
        print('R', R)
        return R

    def _pick_seam_angle(self, dmaps):
        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2
        top = np.ones((1, 4, 2)) * [[[0.05, 5*math.pi/4]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi - 0.05, 5*math.pi/4]]]
        return np.concatenate([top, bottom]), None

    def _generate_path(self, matches, dmaps):
        #TODO better algorithm that tries to use as many nodes as possible.
        # Maybe strongly discourage large gaps.

        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2
        top = np.ones((1, 4, 2)) * [[[0.05, 5*math.pi/4 + 0.1]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi - 0.05, 5*math.pi/4 + 0.1]]]
        return np.concatenate([top, bottom]), None

        m = np.concatenate([top, matches, bottom])
        dim = m.shape[0]
        sort_idx = np.argsort(m[:,0,0])
        m0_col = m[sort_idx,0:1]
        m0_row = m0_col.reshape((1, dim, 2))

        valid = np.ones((dim, dim, 1), bool)
        for i in range(4):
            dphi = m[sort_idx,i:i+1,0].reshape((1, dim, 1)) - m[sort_idx,i:i+1,0:1]
            valid = np.logical_and(valid, dphi > 0.01)

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

        path_r = dmaps[0].eval(path_at.reshape((dim, D*dim, 2))).reshape((dim, dim, D))

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
        mat[np.logical_not(valid.reshape((dim, dim)))] = 0

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
        path_points_r = dmaps[0].eval(path_points[:,0]).reshape((path_points.shape[0], 1, 1))

        path_points_r = path_points_r * np.ones(path_points.shape[0:2] + (1,))
        return np.concatenate([path_points, path_points_r], axis=-1), sort_idx[path[1:-1]] - 1


    def _debug_fit(self, x, y, err):
        err_0 = y - x

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection='3d')
        ax.plot3D(x[:,0], x[:,1], err_0[:,1], 'b+', markersize=0.5)
        ax.set_xlim([-math.pi/2, math.pi/2])
        ax.set_ylim([-math.pi/2, math.pi/2])
        ax.set_zlim(-0.05, 0.05)

        ax = f.add_subplot(1, 2, 2, projection='3d')
        ax.plot3D(x[:,0], x[:,1], err[:,1], 'b+', markersize=0.5)
        ax.set_xlim([-math.pi/2, math.pi/2])
        ax.set_ylim([-math.pi/2, math.pi/2])
        ax.set_zlim(-0.05, 0.05)

    def _debug_dmap(self, dmap):
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        pts = coordinates.polar_points_3d((1024, 1024))
        r = dmap.eval(pts)
        clr = colors.TwoSlopeNorm(vmin=np.min(r), vcenter=np.mean(r), vmax=np.max(r))
        pos = ax.imshow(r, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)


    def _filter_depth_map(self, dmap, seam):
        dim = 60
        c = np.zeros((dim, 2), np.float32)
        c[:,0] = np.linspace(seam[0,0], seam[-1,0], dim)
        c[:,1] = coordinates.seam_intersect(seam, c[:,0])
        return DepthMap(c, dmap.set_area(8).eval(c))

    # returns
    #   depth_maps one for each image (centered about math.pi, with data for left and right seams)
    #   matches_by_seam (aligned to the 4 seams in the image seams apply to the images as follows
    #     [left-left, left-right, right-left, right-right]
    #   seams (aligned to the 4 seams in the image)
    def _compute_by_depth(self, imgs, locations):
        imgs = imgs + imgs[:2]
        locations = locations + locations[:2]
        depth_maps = [[] for i in range(8)]
        matches_by_seam = []
        seams = []
        for i in range(0, 8, 2):
            slc = DepthMapperSlice(imgs[i:i+4], locations[i:i+4], self._debug)
            dmaps = slc.map()
            matches_by_seam.append(slc._matches)
            paths, indices = self._generate_path(slc._matches, dmaps)
            seams.append(paths)
            for j in range(4):
                depth_maps[(i+j)%8] += [dmaps[j].filter(indices)]

        depth_maps = [DepthMap.merge(d) for d in depth_maps]
        return depth_maps, matches_by_seam, seams

    def align_average(self):
        dim = self._images[0].shape[0]
        imgs = self._images + self._images[:2]
        locations = [c.t for c in self.calibration]
        locations = locations + locations[:2]

        matches_by_seam = []
        for i in range(0, 8, 2):
            slc = DepthMapperSlice(imgs[i:i+4], locations[i:i+4], self._debug)
            slc.map()
            matches_by_seam.append(slc._matches)

        initial = [[] for i in range(8)]
        target = [[] for i in range(8)]
        for i, ms in enumerate(matches_by_seam):
            adj = np.zeros(ms.shape, np.float32)
            for j in range(4):
                ii = (i*2+j) % 8
                # account for shift after the mean process of targeting
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                initial[ii].append(ms[:,j] + shift)

            for j in range(4):
                ii = (i*2+j) % 8
                ij = j % 2
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                target[ii].append(np.mean(ms[:,[ij,ij+2]], axis=1) + shift)

        initial = [np.concatenate(c) for c in initial]
        target = [np.concatenate(c) for c in target]

        offset = [math.pi/2, math.pi]
        transforms = []
        for i in range(8):
            lr = LinearRegression(np.array([2, 4]), True)
            kept = trim_outliers_by_diff(target[i], initial[i], [3, 3])
            ti = target[i][kept]
            ii = initial[i][kept]
            err = lr.regression(ti - offset, ii - offset)
            self._debug_fit(ti - offset, ii - offset, err)
            t = TransformLinReg(self._debug) \
                .set_regression(lr) \
                .set_offset(offset)
            transforms.append(t)

        seams = []
        for i in range(8):
            paths, indices = self._generate_path(None, None)
            seams.append(paths[:,0] - [0, math.pi])

        self._seams = seams
        self._transforms = transforms
        return seams

    # each input is an array of length 4.
    # the result is the seam to the right of the image with center at 0
    # the seam will be approximately at pi/4
    def _find_seam(self, dmaps, points, err):
        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2

        md = np.median(points[0][:,1])
        print(md)
        top = np.ones((1, 4, 2)) * [[[0.0, md]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi, md]]]
        return np.concatenate([top, bottom]) - [0, math.pi]

        matches = np.zeros((points[0].shape[0], 4, 2), np.float32)
        for i, p in enumerate(points):
            matches[:,i] = p.copy()

        m = np.concatenate([top, matches, bottom])
        dim = m.shape[0]
        sort_idx = np.argsort(m[:,0,0])
        m0_col = m[sort_idx,0:1]
        m0_row = m0_col.reshape((1, dim, 2))

        valid = np.ones((dim, dim, 1), bool)
        for i in range(4):
            dphi = m[sort_idx,i:i+1,0].reshape((1, dim, 1)) - m[sort_idx,i:i+1,0:1]
            valid = np.logical_and(valid, dphi > 0.01)

        def square_and_scale(cost):
            if np.count_nonzero(cost>0) == 0:
                return cost
            cost = cost / np.min(cost[cost>0])
            cost = cost * cost
            cost = (999 * (cost-1) / np.max(cost - 1)) + 1
            return cost

        def create_path(pts, D):
            dim = pts.shape[1]
            phi_ff, phi_ii = np.meshgrid(pts[...,0], pts[...,0])
            theta_ff, theta_ii = np.meshgrid(pts[...,1], pts[...,1])

            rg = np.arange(D).reshape((1, 1, D, 1))
            path_at = np.zeros((dim, dim, D, 2), np.float32)
            path_at[...,0:1] = phi_ii.reshape((dim, dim, 1, 1)) \
                + rg * (phi_ff.reshape((dim, dim, 1, 1)) - phi_ii.reshape((dim, dim, 1, 1))) / (D-1)
            path_at[...,1:2] = theta_ii.reshape((dim, dim, 1, 1)) \
                + rg * (theta_ff.reshape((dim, dim, 1, 1)) - theta_ii.reshape((dim, dim, 1, 1))) / (D-1)
            return path_at

        D = 50
        path_at = create_path(m0_row, D)
        path_r = dmaps[0].eval(path_at.reshape((dim, D*dim, 2))).reshape((dim, dim, D))

        delta_position = path_at[...,1:,:] - path_at[...,:-1,:]
        delta_position = np.sqrt(np.sum(delta_position * delta_position, axis=-1))
        path_valid = np.all(delta_position > 0.00001, axis=-1)

        error = np.zeros((dim, dim), np.float32)
        for i in range(4):
            err_map = DepthMap(points[i], err[i])
            p = create_path(m[sort_idx,i:i+1].reshape((1, dim, 2)), D)
            path_err = err_map.eval(p.reshape((dim, D*dim, 2))).reshape((dim, dim, D))
            error[path_valid] += np.sum(path_err[...,:-1] * delta_position, axis=-1)[path_valid]
        error_cost = square_and_scale(error)

        position_cost = square_and_scale(np.sum(delta_position, axis=-1))

        path_cart = coordinates.polar_to_cart(path_at, path_r)
        delta_cart = path_cart[...,1:,:] - path_cart[...,:-1,:]
        delta_cart = np.sqrt(np.sum(delta_cart * delta_cart, axis=-1))
        cart_cost = square_and_scale(np.sum(delta_cart, axis=-1))

        slope = np.zeros((dim, dim), np.float32)
        slope[path_valid] = np.sum(np.abs(path_r[...,1:] - path_r[...,:-1]) / delta_position, axis=-1)[path_valid]
        slope_cost = square_and_scale(slope)

        phi_cost = square_and_scale(m0_row[...,0] - m0_col[...,0])

        mat = 0.2*slope_cost + 0.4*error_cost + 0.1*cart_cost + 0.1*position_cost + 0.2*phi_cost
        mat[np.logical_not(valid.reshape((dim, dim)))] = 0

        if self._debug.enable('seam-path-cost'):
            f = plt.figure()
            ax = f.add_subplot(2, 3, 2)
            ax.set_title('Slope Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(slope_cost), vcenter=np.mean(slope_cost), vmax=np.max(slope_cost))
            pos = ax.imshow(slope_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 3, 3)
            ax.set_title('Position Cost')
            pos = ax.imshow(position_cost, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 3, 4)
            ax.set_title('Cartesian Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(cart_cost), vcenter=np.mean(cart_cost), vmax=np.max(cart_cost))
            pos = ax.imshow(cart_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 3, 5)
            ax.set_title('Error Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(error_cost), vcenter=np.mean(error_cost), vmax=np.max(error_cost))
            pos = ax.imshow(error_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 3, 6)
            ax.set_title('Phi Cost')
            clr = colors.TwoSlopeNorm(vmin=np.min(phi_cost), vcenter=np.mean(phi_cost), vmax=np.max(phi_cost))
            pos = ax.imshow(phi_cost, norm=clr, cmap='summer', interpolation='none')
            f.colorbar(pos, ax=ax)

            ax = f.add_subplot(2, 3, 1)
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
        path_points_r = dmaps[0].eval(path_points[:,0]).reshape((path_points.shape[0], 1, 1))

        path_points_r = path_points_r * np.ones(path_points.shape[0:2] + (1,))
        return np.concatenate([path_points, path_points_r], axis=-1) - [0, math.pi, 0]


    def align(self):
        dim = self._images[0].shape[0]
        imgs = self._images + self._images[:2]
        locations = [c.t for c in self.calibration]
        locations = locations + locations[:2]

        depth_maps_by_seam = [[] for i in range(4)]
        depth_maps = [[] for i in range(8)]
        matches_by_seam = []
        seams = []
        for i in range(0, 8, 2):
            slc = DepthMapperSlice(imgs[i:i+4], locations[i:i+4], self._debug)
            dmaps = slc.map()
            matches_by_seam.append(slc._matches)
            for j in range(4):
                depth_maps_by_seam[int(i/2)].append(dmaps[j])
                depth_maps[(i+j)%8].append(dmaps[j])

        depth_maps = [DepthMap.merge(d).set_area(4) for d in depth_maps]

        transforms = []
        R = self._calculate_R()
        for i, d in enumerate(depth_maps):
            t = TransformDepth(self._debug) \
                .override_params(R, 0.06) \
                .set_eye(i%2, switch_axis(self.calibration[i].t)) \
                .set_position(switch_axis(self.calibration[i].t)) \
                .set_depth(d)
            #self._debug_dmap(d)
            transforms.append(t)

        initial = [[] for i in range(8)]
        target = [[] for i in range(8)]
        side = [[] for i in range(8)]
        target_by_seam = [[] for i in range(4)]
        for i, ms in enumerate(matches_by_seam):
            adj = np.zeros(ms.shape, np.float32)
            for j in range(4):
                ii = (i*2+j) % 8
                # account for shift after the mean process of targeting
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                adj[:,j] = transforms[ii].apply(ms[:,j].copy() + shift) - shift
                initial[ii].append(ms[:,j].copy() + shift)

            for j in range(4):
                ii = (i*2+j) % 8
                ij = j % 2
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                print('using average instead of distance')
                t = (ms[:,ij] + ms[:,ij+2]) / 2
                target[ii].append(t + shift)
                depth_maps_by_seam[i][j] = DepthMap(t + shift, depth_maps_by_seam[i][j]._r)
                target_by_seam[i].append(t)
                side[ii].append(int(j/2) * np.ones((t.shape[0], ), np.float32))

        initial = [np.concatenate(c) for c in initial]
        target = [np.concatenate(c) for c in target]
        side = [np.concatenate(c) for c in side]

        offset = [math.pi/2, math.pi]
        transforms = []
        align_err = []
        for i in range(8):
            lr = LinearRegression(np.array([2, 4]), True)
            kept = trim_outliers_by_diff(target[i], initial[i], [5, 5])
            ti = target[i][kept]
            ii = initial[i][kept]
            #show_polar_plot(ii, ti)
            err = target[i] - initial[i]
            err[kept] = lr.regression(ti - offset, ii - offset)
            align_err.append(np.sqrt(np.sum(err*err, axis=-1)))
            if self._debug.enable('align-regression'):
                self._debug_fit(ti - offset, ii - offset, err[kept])

            t = TransformLinReg(self._debug) \
                .set_regression(lr) \
                .set_offset(offset)
            err0 = target[i] - initial[i]
            err = t.eval(target[i]) - initial[i]
            self._debug.log('initial', i, np.mean(err0, axis=0), np.std(err0, axis=0))
            self._debug.log('final', i, np.mean(err, axis=0), np.std(err, axis=0))
            transforms.append(t)

        """
        dim = self._images[0].shape[0]
        plr = coordinates.polar_points_3d((dim, dim))
        for i, img in enumerate(self._images):
            plr_lcl = transforms[i].eval(plr)
            eqr = coordinates.polar_to_eqr_3d(plr_lcl, self._images[0].shape)
            img1 = coordinates.eqr_interp_3d(eqr, img)
            plt.figure()
            plt.imshow(cv.cvtColor(get_middle(img), cv.COLOR_BGR2RGB))
            plt.figure()
            plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        """

        align_err_by_seam = [[] for i in range(4)]
        for s in range(4):
            for j in range(4):
                i = (s * 2 + j) % 8
                err = align_err[i]
                align_err_by_seam[s].append(err[side[i] == int(j/2)])

        seams = []
        for i in range(4):
            s = self._find_seam(depth_maps_by_seam[i], target_by_seam[i], align_err_by_seam[i])
            self._debug.log('seam length: ', s.shape[0])
            seams.append(s[:,0])
            seams.append(s[:,1])

        self._seams = seams
        self._transforms = transforms
        return seams



    # compute the alignment coefficients
    def align0(self):
        dim = self._images[0].shape[0]
        depth_maps, matches_by_seam, seams = \
            self._compute_by_depth(self._images, [c.t for c in self.calibration])

        transforms = []
        R = self._calculate_R()
        for i, d in enumerate(depth_maps):
            t = TransformDepth(self._debug) \
                .override_params(R, 0.06) \
                .set_eye(i%2, switch_axis(self.calibration[i].t)) \
                .set_position(switch_axis(self.calibration[i].t)) \
                .set_depth(d.set_area(10))
            #self._debug_dmap(d)
            transforms.append(t)

        shift = [0, -math.pi]
        seams_by_image = [None] * 8
        for i, s in enumerate(seams):
            for j in range(2):
                ii = (i*2+j) % 8
                l = transforms[i*2+j].apply(seams[i][:,j,0:2]) + shift
                r = transforms[(i*2+2+j)%8].apply(seams[i][:,j+2,0:2]) + shift
                seams_by_image[ii] = (l+r) / 2

        initial = [[] for i in range(8)]
        target = [[] for i in range(8)]
        for i, ms in enumerate(matches_by_seam):
            adj = np.zeros(ms.shape, np.float32)
            for j in range(4):
                ii = (i*2+j) % 8
                # account for shift after the mean process of targeting
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                adj[:,j] = transforms[ii].apply(ms[:,j] + shift) - shift
                initial[ii].append(ms[:,j] + shift)

            for j in range(4):
                ii = (i*2+j) % 8
                ij = j % 2
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                target[ii].append(np.mean(adj[:,[ij,ij+2]], axis=1) + shift)

        initial = [np.concatenate(c) for c in initial]
        target = [np.concatenate(c) for c in target]

        offset = [math.pi/2, math.pi]
        for i in range(8):
            lr = LinearRegression(np.array([2, 4]), True)
            err = lr.regression(target[i] - offset, initial[i] - offset)
            self._debug_fit(target[i] - offset, initial[i] - offset, err)
            t = TransformLinReg(self._debug) \
                .set_regression(lr) \
                .set_offset(offset)
            transforms[i] = t

        self._seams = seams_by_image
        self._transforms = transforms
        return seams_by_image

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
