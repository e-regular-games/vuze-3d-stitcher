
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

    def _generate_path(self, matches, dmap):
        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2
        top = np.ones((1, 4, 2)) * [[[0, 5*math.pi/4]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi, 5*math.pi/4]]]
        #return np.concatenate([top, bottom]), None

        m = np.concatenate([top, matches, bottom])
        dim = m.shape[0]
        sort_idx = np.argsort(m[:,0,0])
        m0_col = m[sort_idx,0:1]
        m0_row = m0_col.reshape((1, dim, 2))

        valid = np.ones((dim, dim, 1), bool)
        for i in range(4):
            dphi = m[sort_idx,i:i+1,0].reshape((1, dim, 1)) - m[sort_idx,i:i+1,0:1]
            valid = np.logical_and(valid, dphi > 0.0001)

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
        path_points_r = dmap.eval(path_points[:,0]).reshape((path_points.shape[0], 1, 1))

        path_points_r = path_points_r * np.ones(path_points.shape[0:2] + (1,))
        return np.concatenate([path_points, path_points_r], axis=-1), sort_idx[path[1:-1]] - 1


    def matches(self, match_thres=0.075, err_thres=0.0075):
        pass


    def _align(self):
        seam_imgs = self._images[-2:] + self._images
        seams = []
        initial_c = [[] for i in range(8)]
        target_c = [[] for i in range(8)]
        for i in range(0, 8, 2):
            matches = FeatureMatcher4(seam_imgs[i:i+4:2], seam_imgs[i+1:i+5:2], self._debug) \
                .matches()
            print('matches', matches.shape[0])
            paths, indices = self._generate_path(None, None)
            seams.append(paths[:,0,0:2] - [0, math.pi])
            seams.append(paths[:,1,0:2] - [0, math.pi])
            self._debug.log('seam points(', i, '):', paths.shape[0])

            for j in range(4):
                shift = math.floor(j/2) * np.array([0, -math.pi/2], np.float32)
                initial_c[(i-2 + j) % 8] += [matches[:,j] + shift]
                m = matches[:,j].copy()
                m[:,0] = np.mean(matches[:,[j, (j+2)%4],0], axis=-1)
                m[:,1] = np.mean(matches[:,[j, (j+2)%4],1], axis=-1)
                target_c[(i-2 + j) % 8] += [m + shift]

        transforms = []
        offset = [math.pi/2, math.pi]
        initial_c = [np.concatenate(c) for c in initial_c]
        target_c = [np.concatenate(c) for c in target_c]
        for i in range(8):
            lr = LinearRegression(np.array([4, 4]), True)
            err1 = (target_c[i] - initial_c[i])
            _, _, inc = linear_regression \
                .trim_outliers_by_diff(target_c[i], initial_c[i], 1, 1)
            self._debug.log('initial error', np.round(np.mean(err1[inc], axis=0), 6), \
                            np.round(np.std(err1[inc], axis=0), 6))
            err = lr.regression(target_c[i][inc] - offset, initial_c[i][inc] - offset)
            self._debug.log('regression error', np.round(np.mean(err, axis=0), 6), \
                            np.round(np.std(err, axis=0), 6))

            f = plt.figure()
            ax = f.add_subplot(1, 1, 1, projection='3d')
            ax.plot3D(target_c[i][inc,0] - math.pi/2, target_c[i][inc,1] - math.pi, \
                      err[:,1], 'b+', markersize=0.5)
            ax.set_xlim([-math.pi/2, math.pi/2])
            ax.set_ylim([-math.pi/2, math.pi/2])
            ax.set_zlim(-0.1, 0.1)

            t = TransformLinReg(self._debug) \
                .set_regression(lr) \
                .set_offset(offset)
            transforms.append(t)

        self._seams = seams
        self._transforms = transforms
        return seams

    # compute the alignment coefficients
    def align(self):
        seam_imgs = self._images[-2:] + self._images
        dim = self._images[0].shape[0]
        seam_calibs = self.calibration[-2:] + self.calibration
        seams = []
        depth_maps = [[] for i in range(8)]
        seam_depth_maps = [[] for i in range(8)]
        for i in range(0, 8, 2):
            slc = DepthMapperSlice(seam_imgs[i:i+4], [c.t for c in seam_calibs[i:i+4]], self._debug)
            dmaps = slc.map()
            paths, indices = self._generate_path(slc._matches, dmaps[0])
            seams.append(paths[:,0,0:2] - [0, math.pi])
            seams.append(paths[:,1,0:2] - [0, math.pi])
            for j in range(4):
                depth_maps[(i-2 + j) % 8] += [dmaps[j].filter(indices)]

            self._debug.log('seam points(', i, '):', paths.shape[0])

        transforms = []
        R = self._calculate_R()
        self._debug.log('R', R)
        for i, d in enumerate(depth_maps):
            dmap = DepthMap.merge(depth_maps[i])
            t = TransformDepth(self._debug) \
                .override_params(R, 0.06) \
                .set_eye(i%2, switch_axis(self.calibration[i].t)) \
                .set_position(switch_axis(self.calibration[i].t)) \
                .set_depth(dmap)
            if self._debug.enable('depth-map-seams'):
                f = plt.figure()
                img0 = get_middle(self._images[i])
                f.add_subplot(1, 2, 1).imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))

                ax = f.add_subplot(1, 2, 2)
                pts = coordinates.polar_points_3d((1024, 1024))
                r = dmap.eval(pts)
                clr = colors.TwoSlopeNorm(vmin=np.min(r), vcenter=np.mean(r), vmax=np.max(r))
                pos = ax.imshow(r, norm=clr, cmap='summer', interpolation='none')
                f.colorbar(pos, ax=ax)

            transforms.append(t)
        self._seams = seams

        imgs_head = []
        plr = coordinates.polar_points_3d((dim, dim))
        for i, img in enumerate(self._images):
            plr_c = transforms[i].eval(plr)
            eqr_c = coordinates.polar_to_eqr_3d(plr_c, (dim, 2*dim))
            img_c = coordinates.eqr_interp_3d(eqr_c, img)
            imgs_head.append(create_from_middle(img_c))

        seam_imgs = imgs_head[-2:] + imgs_head
        initial_c = [[] for i in range(8)]
        target_c = [[] for i in range(8)]
        for i in range(0, 8, 2):
            matches = FeatureMatcher4(seam_imgs[i:i+4:2], seam_imgs[i+1:i+5:2], self._debug) \
                .matches()
            for j in range(4):
                shift = math.floor(j/2) * np.array([0, -math.pi/2], np.float32)
                initial_c[(i-2 + j) % 8] += [matches[:,j] + shift]
                m = matches[:,j].copy()
                m[:,0] = np.mean(matches[:,[j, (j+2)%4],0], axis=-1)
                m[:,1] = np.mean(matches[:,[j, (j+2)%4],1], axis=-1)
                target_c[(i-2 + j) % 8] += [m + shift]

        initial_c = [np.concatenate(c) for c in initial_c]
        target_c = [np.concatenate(c) for c in target_c]
        offset = [math.pi/2, math.pi]
        for i in range(8):
            lr = LinearRegression(np.array([4, 4]), True)
            err = lr.regression(target_c[i] - offset, initial_c[i] - offset)
            self._debug.log('regression stderr', np.round(np.std(err, axis=0), 6))
            t = TransformLinReg(self._debug) \
                .set_regression(lr) \
                .set_offset(offset)
            transforms[i] = Transforms([t, transforms[i]], self._debug)

        plt.show()

        #TODO have to recompute seams??

        self._transforms = transforms
        return seams

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
