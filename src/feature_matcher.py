
import coordinates
import math
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from Equirec2Perspec import Equirectangular
from linear_regression import trim_outliers_by_diff
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
        dups_idx = t.query_ball_point(kp.polar, 0.001, workers=8)
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

    def _determine_matches(self, kp_a, kp_b, threshold):
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
            p_diff = np.sqrt(np.sum(np.array(dp) * np.array(dp)))
            a_diff = math.sqrt((sin_a[ai] - sin_b[bi]) * (sin_a[ai] - sin_b[bi]) + \
                               (cos_a[ai] - cos_b[bi]) * (cos_a[ai] - cos_b[bi]))


            diff = np.array([p_diff, a_diff])
            diff = np.sqrt(np.sum(diff * diff))
            return diff

        for ma in matches:
            p = None
            if len(ma) >= 2 and ma[0].distance < 0.75 * ma[1].distance:
                diff = compute_range(ma[0].queryIdx, ma[0].trainIdx)
                r = ma[0].queryIdx
                good_matches[r, 1] = ma[0].trainIdx
                good_matches[r, 2] = diff
                good_matches[r, 3] = ma[0].distance

        plt.figure()
        plt.hist(good_matches[good_matches[:,2] > 0,2], bins=20)

        vals, cnt = np.unique(good_matches[:,1], return_counts=True)
        dups = cnt > 1
        print('duplicates', np.count_nonzero(dups) - 1)
        for v in vals[dups][1:]: # the first entry is always -1
            possible = (good_matches[:,1] == v)
            flt = good_matches[possible]
            best = None
            for i in range(flt.shape[0]):
                if best is None or \
                   (0.9 * best[3] < flt[i,3] and best[2] > flt[i,2]) or \
                   flt[i,3] < 0.75 * best[3]:
                    best = flt[i]

            good_matches[possible,1] = -1
            good_matches[int(best[0]),1] = v

        return good_matches[:,:2]

    def _determine_matches_range(self, kp_a, kp_b, threshold):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=4)

        sin_a = np.sin(kp_a.rotation)
        cos_a = np.cos(kp_a.rotation)
        sin_b = np.sin(kp_b.rotation)
        cos_b = np.cos(kp_b.rotation)
        phi_ratio = np.abs(kp_a.polar[:,0] - math.pi/2) / (math.pi/2)

        good_matches = np.zeros((len(kp_a.keypoints), 4), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))

        def compute_range(ai, bi):
            pr = phi_ratio[ai]
            dp = np.abs(kp_a.polar[ai] - kp_b.polar[bi]) / [math.pi, 2*math.pi]
            dp_t = [0.8*threshold, threshold]
            in_range = np.count_nonzero(dp < dp_t) == 2
            if not in_range:
                return False, 0

            a_diff = math.sqrt((sin_a[ai] - sin_b[bi]) * (sin_a[ai] - sin_b[bi]) + \
                               (cos_a[ai] - cos_b[bi]) * (cos_a[ai] - cos_b[bi]))

            # have the acceptable rotation scale with phi
            if a_diff < threshold + 0.5 * threshold * pr:
                p_diff = np.sqrt(np.sum(np.array(dp) * np.array(dp)))
                diff = np.array([p_diff, a_diff])
                diff = np.sqrt(np.sum(diff * diff))
                return True, diff

            return False, 0


        out_of_range = 0
        for ma in matches:
            p = None
            if len(ma) >= 2 and ma[0].distance < 0.75 * ma[1].distance:
                valid, diff = compute_range(ma[0].queryIdx, ma[0].trainIdx)
                if not valid:
                    out_of_range += 1
                p = (ma[0], diff)

            for m in ma:
                valid, diff = compute_range(m.queryIdx, m.trainIdx)
                if valid and (p is None or p[1] > diff):
                    p = (m, diff)

            if p is not None:
                r = p[0].queryIdx
                if good_matches[r, 1] == -1 or \
                   (0.9 * p[0].distance < good_matches[r, 3] and good_matches[r, 2] > p[1]) or \
                   p[0].distance < 0.75 * good_matches[r, 3]:
                    good_matches[r, 1] = p[0].trainIdx
                    good_matches[r, 2] = p[1]
                    good_matches[r, 3] = p[0].distance


        print('out of range:', out_of_range)
        vals, cnt = np.unique(good_matches[:,1], return_counts=True)
        dups = cnt > 1
        print('duplicates', np.count_nonzero(dups) - 1)
        for v in vals[dups][1:]: # the first entry is always -1
            possible = (good_matches[:,1] == v)
            flt = good_matches[possible]
            best = None
            for i in range(flt.shape[0]):
                if best is None or \
                   (0.9 * best[3] < flt[i,3] and best[2] > flt[i,2]) or \
                   flt[i,3] < 0.75 * best[3]:
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
        thetas = [45, -45]
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
        m = self._determine_matches(kp[0], kp[1], self._threshold)
        kp_indices[:, 1] = m[:,1]

        kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
        if kp_indices.shape[0] == 0:
            return None

        pts = np.zeros((kp_indices.shape[0], kp_indices.shape[1], 2))
        for j in range(kp_indices.shape[1]):
            pts[:,j] = kp[j].polar[kp_indices[:,j].astype(np.int)]

        inc = self._refine_matches(pts[:,0], pts[:,1], self._filter, self._filter)
        return pts[inc]


# Used for seam alignment. Searches the right side of the left images
# and the left side of the right images for matching feature points.
class FeatureMatcher4(FeatureMatcher):

    # images for the left and right eyes.
    def __init__(self, imgs_left, imgs_right, debug):
        super().__init__(debug)
        self._imgs_left = imgs_left
        self._imgs_right = imgs_right
        self._threshold = 0.15
        self._filter = 2.0

    # returns aligned to [ left eye images, right eye images]
    def matches(self):
        imgs = self._imgs_left + self._imgs_right
        thetas = [40, -40, 40, -40]

        kp = [None] * 4
        for i in range(4):
            angles = [(50, thetas[i]), (0, thetas[i]), (-50, thetas[i])]
            kp[i] = self._create_polar_keypoints(imgs[i], angles)
            if kp[i].empty():
                return None
            kp[i].polar[:,1] += (i%2)*math.pi/2

        kp_indices = np.zeros((len(kp[0].keypoints), 4), dtype=np.int) - 1
        kp_indices[:, 0] = np.arange(0, len(kp[0].keypoints))
        for i in range(1, 4):
            m = self._determine_matches(kp[0], kp[i], self._threshold)
            kp_indices[:, i] = m[:,1]

        kp_indices = kp_indices[(kp_indices != -1).all(axis=1)]
        if kp_indices.shape[0] == 0:
            return None

        pts = np.zeros((kp_indices.shape[0], kp_indices.shape[1], 2))
        for j in range(kp_indices.shape[1]):
            pts[:,j] = kp[j].polar[kp_indices[:,j]]

        #f = [self._filter, self._filter]
        #inc0 = trim_outliers_by_diff(pts[:,0], pts[:,2], self._filter)
        #inc1 = trim_outliers_by_diff(pts[:,1], pts[:,3], self._filter)
        #pts = pts[np.logical_and(inc0, inc1)]

        self._debug.log('FeatureMatcher4: matches:', len(pts))

        if self._debug.enable('features-matches'):
            f = plt.figure();
            for i in range(4):
                ax = f.add_subplot(2, 2, i+1)
                img = imgs[i].copy()
                p = pts[:,i].copy()
                p[:,1] -= (thetas[0] - thetas[i]) * math.pi / 180
                p = np.round(coordinates.polar_to_eqr(p, img.shape)).astype(np.int)
                img[p[:,1], p[:,0]] = [0, 0, 255]
                img[p[:,1]-1, p[:,0]] = [0, 0, 255]
                img[p[:,1]+1, p[:,0]] = [0, 0, 255]
                img[p[:,1], p[:,0]+1] = [0, 0, 255]
                img[p[:,1], p[:,0]-1] = [0, 0, 255]
                img[p[:,1]-1, p[:,0]-1] = [0, 0, 255]
                img[p[:,1]+1, p[:,0]+1] = [0, 0, 255]
                img[p[:,1]-1, p[:,0]+1] = [0, 0, 255]
                img[p[:,1]+1, p[:,0]-1] = [0, 0, 255]
                ax.imshow(cv.cvtColor(get_middle(img), cv.COLOR_BGR2RGB))

        return pts
