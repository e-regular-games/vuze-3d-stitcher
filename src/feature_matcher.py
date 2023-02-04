
import coordinates
import math
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from Equirec2Perspec import Equirectangular

# A collection of feature matching behaviors.
# FeatureMatcher is the base class which is not intended for use.
# Instead use FeatureMatcher2 for matching between the left and right eyes
# or FeatureMatcher4 for matching between the left and right eyes on the left
# and right side of a seam.

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
        self.rectilinear_resolution = 800 # pixels
        self.rectilinear_fov = 62 # degrees
        self._debug = debug

    def _dedup(self, kp):
        keep = np.ones((len(kp.keypoints),), bool)
        t = KDTree(kp.polar)
        dups_idx = t.query_ball_point(kp.polar, 0.0001, workers=8)
        for i, dups in enumerate(dups_idx):
            if not keep[i]: continue
            for d in dups:
                if d == i: continue
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

        return self._dedup(pkp)

    def _determine_matches(self, kp_a, kp_b, threshold):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=8)

        sin_a = np.sin(kp_a.rotation)
        cos_a = np.cos(kp_a.rotation)
        sin_b = np.sin(kp_b.rotation)
        cos_b = np.cos(kp_b.rotation)

        good_matches = np.zeros((len(kp_a.keypoints), 3), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))
        for ma in matches:
            p = None
            cnt = 0
            for m in ma:
                ai = m.queryIdx
                bi = m.trainIdx
                dp = kp_a.polar[ai] - kp_b.polar[bi]
                in_range = np.count_nonzero(np.abs(dp) < [threshold, 2*threshold]) == 2
                if not in_range:
                    continue

                a_diff = math.sqrt((sin_a[ai] - sin_b[bi]) * (sin_a[ai] - sin_b[bi]) + \
                                   (cos_a[ai] - cos_b[bi]) * (cos_a[ai] - cos_b[bi]))

                phi_ratio = math.fabs(kp_a.polar[ai][0] - math.pi/2) / (math.pi/2)
                # have the acceptable rotation scale with phi
                if a_diff < threshold + threshold * phi_ratio:
                    cnt += 1
                    p_diff = np.sqrt(np.sum(np.array(dp) * np.array(dp)))
                    diff = np.array([p_diff, a_diff])
                    p = (m, np.sqrt(np.sum(diff * diff)))
            if cnt == 1:
                r = p[0].queryIdx
                if good_matches[r, 2] > p[1] or good_matches[r, 2] == -1:
                    good_matches[r, 1] = p[0].trainIdx
                    good_matches[r, 2] = p[1]

        return good_matches[:,:2]


class FeatureMatcher2(FeatureMatcher):

    def __init__(self, img_left, img_right, debug):
        super().__init__(debug)
        self._img_left = img_left
        self._img_right = img_right
        self._threshold = 0.05

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
        return pts


# Used for seam alignment. Searches the right side of the left images
# and the left side of the right images for matching feature points.
class FeatureMatcher4(FeatureMatcher):
    def __init__(self, imgs_left, imgs_right, debug):
        super().__init__(debug)
        self._imgs_left = imgs_left
        self._imgs_right = imgs_right
        self._threshold = 0.1

    def matches(self):
        imgs = self._imgs_left + self._imgs_right
        thetas = [45, -45, 45, -45]

        kp = [None] * 4
        for i in range(4):
            angles = [(60, thetas[i]), (0, thetas[i]), (-60, thetas[i])]
            kp[i] = self._create_polar_keypoints(imgs[i], angles)
            if kp[i].empty():
                return None
            kp[i].polar[:,1] += (thetas[0] - thetas[i]) * math.pi / 180

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
            pts[:,j] = kp[j].polar[kp_indices[:,j].astype(np.int)]
        return pts
