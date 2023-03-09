
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
from depth_mesh import DepthMapEllipsoid
from depth_mesh import DepthMapCloud
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
        self._debug = debug

        self.calibration = None

        self._seams = None
        self._transforms = []

        self._matches_seam = None

        for i, img in enumerate(images):
            t = Transform(debug)
            t.label = 'lens:' + str(i+1)
            self._transforms.append(t)

        self.world_radius = np.array([[40, 40, 10]], np.float32)
        self.camera_height = 1.8
        self.interocular = 0.06

    def data(self):
        return self._seams, self._transforms

    def _calculate_R(self):
        p = np.zeros((len(self.calibration), 3), np.float32)
        for i in range(len(self.calibration)):
            p[i] = switch_axis(self.calibration[i].t)

        R = np.mean(np.sqrt(np.sum(p*p, axis=1)))
        return R

    def _debug_fit(self, x, y, err):
        err_0 = y - x

        self._debug.log('initial', np.mean(err_0, axis=0), np.std(err_0, axis=0))
        self._debug.log('final', np.mean(err, axis=0), np.std(err, axis=0))

        if not self._debug.enable('align-regression'):
            return

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection='3d')
        ax.plot3D(x[:,0], x[:,1], err_0[:,1], 'b+', markersize=0.5)
        ax.set_xlim([-math.pi/2, math.pi/2])
        ax.set_ylim([-math.pi/2, math.pi/2])
        ax.set_zlim(-0.1, 0.1)

        ax = f.add_subplot(1, 2, 2, projection='3d')
        ax.plot3D(x[:,0], x[:,1], err[:,1], 'b+', markersize=0.5)
        ax.set_xlim([-math.pi/2, math.pi/2])
        ax.set_ylim([-math.pi/2, math.pi/2])
        ax.set_zlim(-0.1, 0.1)

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

    def align(self):
        dim = self._images[0].shape[0]
        imgs = self._images + self._images[:2]
        locations = [switch_axis(c.t) for c in self.calibration]
        locations = locations + locations[:2]

        R = self._calculate_R()
        z = np.array([[0, 0, self.world_radius[0,2] - self.camera_height]], np.float32)
        depth_maps = [DepthMapEllipsoid(self.world_radius, locations[i] + z) for i in range(8)]

        transforms = []
        for i in range(8):
            t = TransformDepth(self._debug) \
                .set_interocular(R, self.interocular) \
                .set_eye(i % 2) \
                .set_position(locations[i]) \
                .set_depth(depth_maps[i])
            if self._debug.verbose:
                t.validate()
            transforms.append(t)

        matches_by_seam = []
        seams = []
        for i in range(0, 8, 2):
            matches = FeatureMatcher4(imgs[i:i+4:2], imgs[i+1:i+4:2], self._debug).matches()
            if matches is not None:
                matches = matches[:,[0,2,1,3]]
            if self._matches_seam is not None:
                keep = self._filter_existing_matches(self._matches_seam[int(i/2)], matches)
                combined = np.concatenate([self._matches_seam[int(i/2)][keep], matches])
                matches_by_seam.append(combined)
            else:
                matches_by_seam.append(matches)

        self._matches_seam = matches_by_seam

        initial = [[] for i in range(8)]
        target = [[] for i in range(8)]
        side = [[] for i in range(8)]
        target_by_seam = [[] for i in range(4)]
        depth_maps_by_seam = [[] for i in range(4)]
        for i, ms in enumerate(matches_by_seam):
            adj = np.zeros(ms.shape, np.float32)
            for j in range(4):
                ii = (i*2+j) % 8
                ij = j % 2
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)
                adj[:,j] = transforms[ii].forward(ms[:,j] + shift) - shift
                initial[ii].append(adj[:,j] + shift)

            for j in range(4):
                ii = (i*2+j) % 8
                ij = j % 2
                shift = int(j/2)*np.array([0, -math.pi/2], np.float32)

                t = (adj[:,ij] + adj[:,ij+2]) / 2
                target[ii].append(t + shift)
                depth_maps_by_seam[i].append(depth_maps[ii])
                target_by_seam[i].append(t)
                side[ii].append(int(j/2) * np.ones((t.shape[0], ), np.float32))

        initial = [np.concatenate(c) for c in initial]
        target = [np.concatenate(c) for c in target]
        side = [np.concatenate(c) for c in side]

        offset = [math.pi/2, math.pi]
        align_err = []
        for i in range(8):
            lrr = LinearRegression(np.array([3, 4]), True)
            kept = trim_outliers_by_diff(target[i], initial[i], [3, 3])
            ti = target[i][kept]
            ii = initial[i][kept]
            err = target[i] - initial[i]
            err[kept] = lrr.regression(ti - offset, ii - offset)
            align_err.append(np.sqrt(np.sum(err*err, axis=-1)))

            lrf = LinearRegression(np.array([3, 4]), True)
            lrf.regression(ii - offset, ti - offset)

            self._debug_fit(ti - offset, ii - offset, err[kept])

            tlr = TransformLinReg(self._debug) \
                .set_regression(lrf, lrr) \
                .set_offset(offset)
            transforms[i] = Transforms([transforms[i], tlr])

            if self._debug.verbose:
                transforms[i].validate()
            if self._debug.enable('image-transforms'):
                transforms[i].show(get_middle(self._images[i]))

        align_err_by_seam = [[] for i in range(4)]
        for s in range(4):
            for j in range(4):
                i = (s * 2 + j) % 8
                err = align_err[i]
                align_err_by_seam[s].append(err[side[i] == int(j/2)])

        seams = []
        for i in range(4):
            s = ChooseSeam(self._debug) \
                .depth_maps(depth_maps_by_seam[i]) \
                .matches(target_by_seam[i]) \
                .error(align_err_by_seam[i]) \
                .find_seam()
            self._debug.log('seam length: ', s.shape[0])
            seams.append(s[:,0])
            seams.append(s[:,1])

        self._seams = seams
        self._transforms = transforms
        return seams

    def to_dict(self):
        d = {
            'seams': [],
            'transforms': [],
            'matchesBySeam': []
        }

        for s in self._seams:
            d['seams'].append(s.tolist())
        for t in self._transforms:
            d['transforms'].append(t.to_dict())
        for m in self._matches_seam:
            d['matchesBySeam'].append(m.tolist())

        return d

    def from_dict(self, d):
        self._seams = []
        for s in d['seams']:
            self._seams.append(np.array(s))

        self._transforms = []
        for s in d['transforms']:
            t = Transform.from_dict(s, self._debug)
            self._transforms.append(t)

        self._matches_seam = []
        for m in d['matchesBySeam']:
            self._matches_seam.append(np.array(m))

        return self

# each input is an array of length 4.
# the result is the seam to the right of the image with center at 0
# the seam will be approximately at pi/4
class ChooseSeam():
    def __init__(self, debug):
        self._debug = debug
        self._dmaps = None
        self._points = None
        self._err = None

        self._D = 50
        self._valid = None
        self._path_valid = None
        self._sort_index = None
        self._delta_position = None

        self._error_cost = None
        self._position_cost = None
        self._cart_cost = None
        self._slope_cost = None
        self._phi_cost = None
        self._mat = None


    def depth_maps(self, dmaps):
        self._dmaps = dmaps
        return self

    # list of np arrays, one for each images in the seam
    def matches(self, matches):
        self._points = matches
        return self

    def error(self, err):
        self._err = err
        return self

    def _create_error_cost(self, m):
        dim = m.shape[0]
        error = np.zeros((dim, dim), np.float32)
        for i in range(4):
            err_map = DepthMapCloud(self._points[i], self._err[i])
            p = self._create_path(m[self._sort_idx,i:i+1].reshape((1, dim, 2)))
            path_err = err_map.eval(p.reshape((dim, self._D * dim, 2))).reshape((dim, dim, self._D))
            error[self._path_valid] += \
                np.sum(path_err[...,:-1] * self._delta_position, axis=-1)[self._path_valid]
        self._error_cost = self._square_and_scale(error)

    def _plot(self):
        f = plt.figure()
        ax = f.add_subplot(2, 3, 2)
        ax.set_title('Slope Cost')
        #clr = colors.TwoSlopeNorm(vmin=np.min(self._slope_cost), \
                                  #vcenter=np.mean(self._slope_cost), \
                                  #vmax=np.max(self._slope_cost))
        pos = ax.imshow(self._slope_cost, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

        ax = f.add_subplot(2, 3, 3)
        ax.set_title('Theta Cost')
        clr = colors.TwoSlopeNorm(vmin=np.min(self._theta_cost), \
                                  vcenter=np.mean(self._theta_cost), \
                                  vmax=np.max(self._theta_cost))
        pos = ax.imshow(self._theta_cost, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

        ax = f.add_subplot(2, 3, 4)
        ax.set_title('Cartesian Cost')
        clr = colors.TwoSlopeNorm(vmin=np.min(self._cart_cost), \
                                  vcenter=np.mean(self._cart_cost), \
                                  vmax=np.max(self._cart_cost))
        pos = ax.imshow(self._cart_cost, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

        ax = f.add_subplot(2, 3, 5)
        ax.set_title('Error Cost')
        clr = colors.TwoSlopeNorm(vmin=np.min(self._error_cost),
                                  vcenter=np.mean(self._error_cost), \
                                  vmax=np.max(self._error_cost))
        pos = ax.imshow(self._error_cost, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

        ax = f.add_subplot(2, 3, 6)
        ax.set_title('Phi Cost')
        clr = colors.TwoSlopeNorm(vmin=np.min(self._phi_cost), \
                                  vcenter=np.mean(self._phi_cost), \
                                  vmax=np.max(self._phi_cost))
        pos = ax.imshow(self._phi_cost, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

        ax = f.add_subplot(2, 3, 1)
        ax.set_title('Combined Cost')
        clr = colors.TwoSlopeNorm(vmin=np.min(self._mat), \
                                  vcenter=np.mean(self._mat), \
                                  vmax=np.max(self._mat))
        pos = ax.imshow(self._mat, norm=clr, cmap='summer', interpolation='none')
        f.colorbar(pos, ax=ax)

    def _create_valid(self, m):
        dim = m.shape[0]
        self._valid = np.ones((dim, dim, 1), bool)
        for i in range(4):
            dphi = m[self._sort_idx,i:i+1,0].reshape((1, dim, 1)) - m[self._sort_idx,i:i+1,0:1]
            self._valid = np.logical_and(self._valid, dphi > 0.01)

    def find_seam(self):
        # note: phi=0 is the top of the image, phi=pi is the bottom
        # theta is around 3pi/2

        md = np.median(self._points[0][:,1])
        top = np.ones((1, 4, 2)) * [[[0.0, md]]]
        bottom = np.ones((1, 4, 2)) * [[[math.pi, md]]]

        matches = np.zeros((self._points[0].shape[0], 4, 2), np.float32)
        for i, p in enumerate(self._points):
            matches[:,i] = p.copy()

        m = np.concatenate([top, matches, bottom])
        self._sort_idx = np.argsort(m[:,0,0])
        dim = m.shape[0]
        m0_col = m[self._sort_idx,0:1]
        m0_row = m0_col.reshape((1, dim, 2))

        self._create_valid(m)

        path_at = self._create_path(m0_row)
        path_r = self._dmaps[0].eval(path_at.reshape((dim, self._D*dim, 2))).reshape((dim, dim, self._D))

        delta_position = path_at[...,1:,:] - path_at[...,:-1,:]
        self._delta_position = np.sqrt(np.sum(delta_position * delta_position, axis=-1))
        self._path_valid = np.all(self._delta_position > 0.00001, axis=-1)

        self._create_error_cost(m)

        path_cart = coordinates.polar_to_cart(path_at, path_r)
        delta_cart = path_cart[...,1:,:] - path_cart[...,:-1,:]
        delta_cart = np.sqrt(np.sum(delta_cart * delta_cart, axis=-1))
        self._cart_cost = self._square_and_scale(np.sum(delta_cart, axis=-1))

        slope = np.zeros((dim, dim), np.float32)
        slope[self._path_valid] = np.sum(np.abs(path_r[...,1:] - path_r[...,:-1])[self._path_valid] / self._delta_position[self._path_valid], axis=-1)
        self._slope_cost = self._square_and_scale(slope)

        self._phi_cost = self._square_and_scale(m0_row[...,0] - m0_col[...,0])
        self._theta_cost = self._square_and_scale(m0_row[...,1] - m0_col[...,1])

        self._mat = 0.2 * self._slope_cost \
            + 0.4 * self._error_cost \
            + 0.1 * self._cart_cost \
            + 0.1 * self._phi_cost \
            + 0.2 * self._theta_cost
        self._mat[np.logical_not(self._valid.reshape((dim, dim)))] = 0

        if self._debug.enable('seam-path-cost'):
            self._plot()

        rdist, rpred = shortest_path(self._mat, return_predecessors=True)

        top_idx = m0_col.shape[0]-1
        path = [top_idx]
        pred = rpred[0,top_idx]
        while pred >= 0:
            path.append(pred)
            pred = rpred[0, pred]

        path_points = np.flip(m[self._sort_idx][path], axis=0)
        return path_points - [0, math.pi]


    def _square_and_scale(self, cost):
        if np.count_nonzero(cost>0) == 0:
            return cost
        cost = cost / np.min(cost[cost>0])
        cost = cost * cost
        cost = (999 * (cost-1) / np.max(cost - 1)) + 1
        return cost

    def _create_path(self, pts):
        dim = pts.shape[1]
        phi_ff, phi_ii = np.meshgrid(pts[...,0], pts[...,0])
        theta_ff, theta_ii = np.meshgrid(pts[...,1], pts[...,1])

        rg = np.arange(self._D).reshape((1, 1, self._D, 1))
        path_at = np.zeros((dim, dim, self._D, 2), np.float32)
        path_at[...,0:1] = phi_ii.reshape((dim, dim, 1, 1)) \
            + rg * (phi_ff.reshape((dim, dim, 1, 1)) - phi_ii.reshape((dim, dim, 1, 1))) / (self._D-1)
        path_at[...,1:2] = theta_ii.reshape((dim, dim, 1, 1)) \
            + rg * (theta_ff.reshape((dim, dim, 1, 1)) - theta_ii.reshape((dim, dim, 1, 1))) / (self._D-1)
        return path_at
