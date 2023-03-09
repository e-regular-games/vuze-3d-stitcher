
import coordinates
import math
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from Equirec2Perspec import Equirectangular
from linear_regression import trim_outliers_by_diff
from linear_regression import trim_outliers
from matplotlib import pyplot as plt

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

class FeatureMatcher():
    def __init__(self, debug):
        self.rectilinear_resolution = 1000 # pixels
        self.rectilinear_fov = 62 # degrees
        self._debug = debug

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

        rkp = PolarKeypoints()
        rkp.keypoints = [i for (i, v) in zip(kp.keypoints, keep.tolist()) if v]
        rkp.descriptors = kp.descriptors[keep]
        rkp.polar = kp.polar[keep]
        rkp.rotation = kp.rotation[keep]
        return rkp


    # angles in degrees as a list of (phi, theta) tuples.
    def _create_polar_keypoints(self, img, angles):
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
            img_f = img.copy()
            eqr = np.round(coordinates.polar_to_eqr(pkp.polar, img.shape)).astype(np.int)
            img_f[eqr[:,1], eqr[:,0]] = [0, 0, 255]
            plt.figure().add_subplot(1, 1, 1).imshow(cv.cvtColor(img_f, cv.COLOR_BGR2RGB))

        return pkp

    def _determine_matches_0(self, kp_a, kp_b, thres=0.75):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=2)

        sin_a = np.sin(kp_a.rotation)
        cos_a = np.cos(kp_a.rotation)
        sin_b = np.sin(kp_b.rotation)
        cos_b = np.cos(kp_b.rotation)

        good_matches = np.zeros((len(kp_a.keypoints), 4), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))

        def compute_range(ai, bi):
            dp = np.abs(kp_a.polar[ai] - kp_b.polar[bi]) / [math.pi, 2*math.pi]
            da = [(sin_a[ai] - sin_b[bi]), (cos_a[ai] - cos_b[bi])]
            d = np.array([dp[0], dp[1], da[0], da[1]], np.float32)
            return np.sqrt(np.sum(d**2))

        for ma in matches:
            p = None
            if len(ma) >= 2 and ma[0].distance < thres * ma[1].distance:
                diff = compute_range(ma[0].queryIdx, ma[0].trainIdx)
                r = ma[0].queryIdx
                good_matches[r, 1] = ma[0].trainIdx
                good_matches[r, 2] = diff
                good_matches[r, 3] = ma[0].distance

        vals, cnt = np.unique(good_matches[:,1], return_counts=True)
        dups = cnt > 1
        self._debug.log('duplicates', np.count_nonzero(dups) - 1)
        for v in vals[dups][1:]: # the first entry is always -1
            possible = (good_matches[:,1] == v)
            flt = good_matches[possible]
            best = None
            for i in range(flt.shape[0]):
                if best is None or \
                   (best[3] < thres * flt[i,3] and best[2] > flt[i,2]) or \
                   flt[i,3] < thres * best[3]:
                    best = flt[i]

            good_matches[possible,1] = -1
            good_matches[int(best[0]),1] = v

        #good_matches[good_matches[:,2]>0.2,1] = -1

        return good_matches[:,:2]

    def _determine_matches(self, kp_a, kp_b, dist=0.125, thres=0.75):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=6)

        sin_a = np.sin(kp_a.rotation)
        cos_a = np.cos(kp_a.rotation)
        sin_b = np.sin(kp_b.rotation)
        cos_b = np.cos(kp_b.rotation)
        phi_ratio = np.abs(kp_a.polar[:,0] - math.pi/2) / (math.pi/2)

        good_matches = np.zeros((len(kp_a.keypoints), 4), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))

        def compute_range(ai, bi):
            dp = np.abs(kp_a.polar[ai] - kp_b.polar[bi]) / [math.pi, 2*math.pi]
            da = [(sin_a[ai] - sin_b[bi]), (cos_a[ai] - cos_b[bi])]
            d = np.array([dp[0], dp[1], da[0], da[1]], np.float32)
            return np.sqrt(np.sum(d**2))

        for ma in matches:
            p = None
            if len(ma) >= 2 and ma[0].distance < thres * ma[1].distance:
                d = compute_range(ma[0].queryIdx, ma[0].trainIdx)
                p = [ma[0].queryIdx, ma[0].trainIdx, d, ma[0].distance]

            for m in ma:
                if m.distance * thres >  ma[0].distance:
                    break
                d = compute_range(m.queryIdx, m.trainIdx)
                if d < dist and (p is None or d < p[1]):
                    p = [m.queryIdx, m.trainIdx, d, m.distance]

            if p is not None:
                good_matches[p[0]] = p

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

        return good_matches[:,:2]

class FeatureMatcher2(FeatureMatcher):

    def __init__(self, img_left, img_right, debug):
        super().__init__(debug)
        self._img_left = img_left
        self._img_right = img_right
        self._threshold = 0.1
        self._filter = 1

    def matches(self):
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

        return pts


# Used for seam alignment. Searches the right side of the left images
# and the left side of the right images for matching feature points.
class FeatureMatcher4(FeatureMatcher):

    # images for the left and right eyes.
    def __init__(self, imgs_left, imgs_right, debug):
        super().__init__(debug)
        self._imgs_left = imgs_left
        self._imgs_right = imgs_right

    # returns aligned to [ left eye images, right eye images]
    def matches(self):
        imgs = self._imgs_left + self._imgs_right
        thetas = [40, -40, 40, -40]

        kp = [None] * 4
        for i in range(4):
            angles = [(60, thetas[i]), (0, thetas[i]), (-60, thetas[i])]
            kp[i] = self._create_polar_keypoints(imgs[i], angles)
            if kp[i].empty():
                return None
            kp[i].polar[:,1] += (i%2)*math.pi/2

        kp_indices = np.zeros((len(kp[0].keypoints), 4), dtype=np.int) - 1
        kp_indices[:, 0] = np.arange(0, len(kp[0].keypoints))
        for i in range(1, 4):
            m = self._determine_matches(kp[0], kp[i])
            kp_indices[:, i] = m[:,1]

        kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
        if kp_indices.shape[0] == 0:
            return None

        pts = np.zeros((kp_indices.shape[0], kp_indices.shape[1], 2))
        for j in range(kp_indices.shape[1]):
            pts[:,j] = kp[j].polar[kp_indices[:,j]]

        inc = np.ones((pts.shape[0],), bool)
        inc = np.logical_and(inc, trim_outliers_by_diff(pts[:,0], pts[:,1], 1))
        inc = np.logical_and(inc, trim_outliers_by_diff(pts[:,2], pts[:,3], 1))
        #for i in range(4):
        #    inc = np.logical_and(inc, trim_outliers(pts[:,i,1], 3))
        pts = pts[inc]

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
                ax.plot(p[:,0] - img.shape[0]/2, p[:,1], 'ro', markersize=1)

        return pts
