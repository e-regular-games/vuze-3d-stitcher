
import coordinates
import math
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from Equirec2Perspec import Equirectangular
from linear_regression import trim_outliers_by_diff
from linear_regression import trim_outliers
from matplotlib import pyplot as plt
import threading

# A collection of feature matching behaviors.
# FeatureMatcher is the base class which is not intended for use.
# Instead use FeatureMatcher2 for matching between the left and right eyes
# or FeatureMatcher4 for matching between the left and right eyes on the left
# and right side of a seam.

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

class PolarKeypoints():
    def __init__(self):
        self.keypoints = []
        self.descriptors = np.zeros((0, 128), np.float32)
        self.polar = np.zeros((0, 2), np.float32)
        self.rotation = np.zeros((0,1), np.float32)

    def empty(self):
        return len(self.keypoints) == 0

    def refine(self, flt):
        r = PolarKeypoints()
        r.keypoints = [i for (i, v) in zip(self.keypoints, flt.tolist()) if v]
        r.descriptors = self.descriptors[flt]
        r.polar = self.polar[flt]
        r.rotation = self.rotation[flt]
        return r

class FilterMatches(threading.Thread):
    def __init__(self):
        super().__init__()
        self._result = None
        pass

    def keypoints(self, a, b):
        self._kp_a = a
        self._kp_b = b
        return self

    def constants(self, dist, thres):
        self._dist = dist
        self._thres = thres
        return self

    def matches(self, matches, i):
        self._matches = matches
        self._idx = i
        return self

    # set or get the result
    def result(self, result=None):
        if result is not None:
            self._result = result

        return self._result


    def run(self):
        kp_a = self._kp_a
        kp_b = self._kp_b

        def compute_range(ai, bi):
            dp = np.abs(kp_a.polar[ai] - kp_b.polar[bi]) / [math.pi, 2*math.pi]
            da = (kp_a.rotation[ai] - kp_b.rotation[bi]) % (math.pi/2)
            if da > math.pi/4:
                da -= math.pi/2
            return np.sqrt(np.sum(dp**2)), da

        for i in self._idx:
            ma = self._matches[i]
            p = None
            if len(ma) >= 2 and ma[0].distance < self._thres * ma[1].distance:
                d, r = compute_range(ma[0].queryIdx, ma[0].trainIdx)
                p = [ma[0].queryIdx, ma[0].trainIdx, d, ma[0].distance, r]

            for m in ma:
                if m.distance * self._thres >  ma[0].distance:
                    break
                d, r = compute_range(m.queryIdx, m.trainIdx)
                if d < self._dist and abs(r) < self._dist and (p is None or d < p[2]):
                    p = [m.queryIdx, m.trainIdx, d, m.distance, r]

            if p is not None:
                self._result[p[0]] = p


class FeatureMatcher(threading.Thread):
    def __init__(self, debug):
        super().__init__()
        self.rectilinear_resolution = 1000 # pixels
        self.rectilinear_fov = 62 # degrees
        self._debug = debug
        self._parallel = 8
        self.result = None

    def _dedup(self, kp):
        keep = np.ones((len(kp.keypoints),), bool)
        t = KDTree(kp.polar)
        dups_idx = t.query_ball_point(kp.polar, 0.0001, workers=8)
        for i, dups in enumerate(dups_idx):
            if not keep[i]:
                continue
            for d in dups:
                if d == i:
                    continue
                keep[d] = False

        self._debug.log('dedup', np.count_nonzero(keep), '/', keep.shape[0])
        return kp.refine(keep)

    def _reduce_points(self, sub_pts, desired):
        pairings = []
        for a in range(sub_pts.shape[1]):
            for b in range(a+1, sub_pts.shape[1]):
                pairings.append((a, b))

        theta_diff = np.zeros((len(sub_pts), len(pairings)), np.float32)
        for i, p in enumerate(pairings):
            theta_diff[:,i] = sub_pts[:,p[1],1] - sub_pts[:,p[0],1]
        std_theta_diff = np.std(theta_diff, axis=-1)
        sort_idx = np.argsort(std_theta_diff)
        k0 = np.zeros((len(sub_pts),), bool)
        k1 = np.zeros((len(sub_pts),), bool)
        k1[:int(desired+1)] = True
        k0[sort_idx] = k1
        return k0

    def _generate_polar_grid(self, sections):
        h = math.pi / sections
        gridh = sections
        gridw = int((math.pi / 3) / h)
        w = (math.pi/3) / gridw
        r = math.sqrt((h/2)**2 + (w/2)**2)

        theta0 = (math.pi + math.pi/4) - math.pi/6
        theta1 = (math.pi + math.pi/4) + math.pi/6
        phi_center = np.linspace(h/2, math.pi - h/2, gridh)
        theta_center = np.linspace(theta0 + w/2, theta1 - w/2, gridw)
        grid = np.zeros((gridh, gridw, 2), np.float32)
        grid[...,0], grid[...,1] = np.meshgrid(phi_center, theta_center, indexing='ij')
        grid = grid.reshape((gridh * gridw, 2))

        return grid, r

    def _reduce_density(self, pts, sections, limits):
        grid, r = self._generate_polar_grid(sections)

        t = KDTree(pts.reshape((pts.shape[0] * pts.shape[1], 2)))
        idx = t.query_ball_point(grid, r, workers=8)

        insides = []
        inside_cnts = []
        for i in idx:
            if len(i) == 0:
                continue
            inside = np.zeros((pts.shape[0] * pts.shape[1],), bool)
            inside[np.array(i)] = True
            inside = inside.reshape((pts.shape[0], pts.shape[1]))
            inside = inside.all(axis=-1)
            cnt = np.count_nonzero(inside)
            if cnt > 0:
                insides.append(inside)
                inside_cnts.append(cnt)

        inside_cnts = np.array(inside_cnts, np.int)
        density = np.median(inside_cnts)
        density = max(min(density, limits[1]), limits[0])

        keep = np.zeros((len(pts),), bool)
        for cnt, inside in zip(inside_cnts, insides):
            if cnt < density:
                keep[inside] = True
                continue
            keep[inside] = self._reduce_points(pts[inside], density)
        return pts[keep]


    # angles in degrees as a list of (phi, theta) tuples.
    def _create_polar_keypoints(self, img, angles):
        self._debug.perf('feature-matcher-keypoints')

        sift = cv.SIFT_create()
        pkp = PolarKeypoints()
        r = self.rectilinear_resolution
        fov = self.rectilinear_fov

        for a in angles:
            rl, inv = Equirectangular(img).GetPerspective(fov, a[1], a[0], r, r);
            gray = cv.cvtColor(rl, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            if len(kp) == 0:
                continue
            pkp.keypoints += kp
            pkp.descriptors = np.concatenate([pkp.descriptors, des])

            pts = np.zeros((len(kp), 2), np.float32)
            rots = np.zeros((len(kp), 1), np.float32)
            for ki, k in enumerate(kp):
                pts[ki] = k.pt
                rots[ki] = k.angle
            rots *= math.pi / 180
            pkp.rotation = np.concatenate([pkp.rotation, rots])
            pts = inv.GetEquirectangularPoints(pts)
            pts = coordinates.eqr_to_polar(pts, img.shape)
            pkp.polar = np.concatenate([pkp.polar, pts])

        pkp = self._dedup(pkp)

        if self._debug.enable('features-keypoints'):
            eqr = np.round(coordinates.polar_to_eqr(pkp.polar, img.shape))
            ax = plt.figure().add_subplot(1, 1, 1)
            ax.imshow(cv.cvtColor(get_middle(img), cv.COLOR_BGR2RGB))
            ax.plot(eqr[:,0] - img.shape[0]/2, eqr[:,1], 'ro', markersize=0.5)

        self._debug.perf('feature-matcher-keypoints')
        return pkp

    def _determine_matches(self, kp_a, kp_b, dist=0.125, thres=0.75):
        self._debug.perf('feature-matcher-matches-bf')
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=6)
        self._debug.perf('feature-matcher-matches-bf')

        good_matches = np.zeros((len(kp_a.keypoints), 5), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))

        self._debug.perf('feature-matcher-matches-filter')
        threads = []
        for p in range(self._parallel):
            t = FilterMatches() \
                .keypoints(kp_a, kp_b) \
                .constants(dist, thres) \
                .matches(matches, np.arange(p, len(matches), self._parallel))
            t.result(good_matches)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        self._debug.perf('feature-matcher-matches-filter')

        self._debug.perf('feature-matcher-matches-dedup')
        vals, cnt = np.unique(good_matches[:,1], return_counts=True)
        dups = cnt > 1
        for v in vals[dups][1:]: # the first entry is always -1
            possible = (good_matches[:,1] == v)
            flt = good_matches[possible]
            best = None
            for i in range(flt.shape[0]):
                if best is None or \
                   (thres * flt[i,3] <  best[3] and best[2] > flt[i,2]) or \
                   flt[i,3] < thres * best[3]:
                    best = flt[i]

            good_matches[possible,1] = -1
            good_matches[int(best[0]),1] = v

        self._debug.perf('feature-matcher-matches-dedup')
        return good_matches[:,:2]

class FeatureMatcher2(FeatureMatcher):

    def __init__(self, img_left, img_right, debug):
        super().__init__(debug)
        self._img_left = img_left
        self._img_right = img_right
        self._threshold = 0.1
        self._filter = 1

    def run(self):
        imgs = [self._img_left, self._img_right]
        angles = [(60, -45), (0, -45), (-60, -45), \
                  (60, 0), (0, 0), (-60, 0), \
                  (60, 45), (0, 45), (-60, 45)]

        kp = [None] * 2
        for i, img in enumerate(imgs):
            kp[i] = self._create_polar_keypoints(imgs[i], angles)
            if kp[i].empty():
                return None

        kp_indices = np.zeros((len(kp[0].keypoints), 2), dtype=np.int) - 1
        kp_indices[:, 0] = np.arange(0, len(kp[0].keypoints))
        m = self._determine_matches(kp[0], kp[1])
        kp_indices[:, 1] = m[:,1]

        kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
        if kp_indices.shape[0] == 0:
            return None

        pts = np.zeros((kp_indices.shape[0], kp_indices.shape[1], 2))
        for j in range(kp_indices.shape[1]):
            pts[:,j] = kp[j].polar[kp_indices[:,j].astype(np.int)]

        inc = trim_outliers_by_diff(pts[:,0], pts[:,1], 2)
        pts = pts[inc]

        self.result = pts


# Used for seam alignment. Searches the right side of the left images
# and the left side of the right images for matching feature points.
class FeatureMatcher4(FeatureMatcher):

    # images for the left and right eyes.
    def __init__(self, imgs_left, imgs_right, debug):
        super().__init__(debug)
        self._imgs_left = imgs_left
        self._imgs_right = imgs_right

    # returns aligned to [ left eye images, right eye images]
    def run(self):
        imgs = self._imgs_left + self._imgs_right
        thetas = [45, -45, 45, -45]

        kp = [None] * 4
        for i in range(4):
            angles = [(60, thetas[i]), (0, thetas[i]), (-60, thetas[i])]
            kp[i] = self._create_polar_keypoints(imgs[i], angles)
            if kp[i].empty():
                return None
            kp[i].polar[:,1] += (i%2)*math.pi/2

        kp_indices = np.zeros((len(kp[0].keypoints), 4), dtype=np.int) - 1
        kp_indices[:, 0] = np.arange(0, len(kp[0].keypoints))
        flt = np.ones(len(kp[0].keypoints), bool)
        kp_base = kp[0]
        for i in range(1, 4):
            m = self._determine_matches(kp_base, kp[i])
            kp_indices[flt, i] = m[:,1]
            flt[flt] = m[:,1] >= 0
            kp_base = kp_base.refine(m[:,1] >= 0)

        kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
        if kp_indices.shape[0] == 0:
            return None

        pts = np.zeros((kp_indices.shape[0], kp_indices.shape[1], 2))
        for j in range(kp_indices.shape[1]):
            pts[:,j] = kp[j].polar[kp_indices[:,j]]

        inc = np.ones((pts.shape[0],), bool)
        inc = np.logical_and(inc, trim_outliers_by_diff(pts[:,0], pts[:,1], 2))
        inc = np.logical_and(inc, trim_outliers_by_diff(pts[:,2], pts[:,3], 2))
        pts = pts[inc]

        pts = self._reduce_density(pts, 12, (10, 20))

        self._debug.log('FeatureMatcher4: matches:', len(pts))

        if self._debug.enable('features-matches'):
            f = plt.figure();
            for i in range(4):
                ax = f.add_subplot(2, 2, i+1)
                img = imgs[i].copy()
                p = pts[:,i].copy()
                p[:,1] -= (i%2)*math.pi/2
                p = np.round(coordinates.polar_to_eqr(p, img.shape)).astype(np.int)
                ax.imshow(cv.cvtColor(get_middle(img), cv.COLOR_BGR2RGB))
                ax.plot(p[:,0] - img.shape[0]/2, p[:,1], 'ro', markersize=0.5)

        self.result = pts
