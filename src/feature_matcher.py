
import coordinates
import math
import numpy as np
import cv2 as cv
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
    def __init__(self):
        self.rectilinear_resolution = 800 # pixels
        self.rectilinear_fov = 62 # degrees

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

        return pkp

    def _determine_matches(self, kp_a, kp_b, threshold):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(kp_a.descriptors, kp_b.descriptors, k=8)

        good_matches = np.zeros((len(kp_a.keypoints), 3), np.float32) - 1
        good_matches[:,0] = np.arange(0, len(kp_a.keypoints))
        for ma in matches:
            p = None
            cnt = 0
            for m in ma:
                dp = kp_a.polar[m.queryIdx] - kp_b.polar[m.trainIdx]
                in_range = np.count_nonzero(np.abs(dp) < [threshold, 2*threshold]) == 2
                if not in_range:
                    continue

                sin_a_a = math.sin(kp_a.rotation[m.queryIdx])
                cos_a_a = math.cos(kp_a.rotation[m.queryIdx])
                sin_a_b = math.sin(kp_b.rotation[m.trainIdx])
                cos_a_b = math.cos(kp_b.rotation[m.trainIdx])
                a_diff = math.sqrt((sin_a_a - sin_a_b) * (sin_a_a - sin_a_b) + \
                                   (cos_a_a - cos_a_b) * (cos_a_a - cos_a_b))

                # have the acceptable rotation scale with phi
                if a_diff < threshold + threshold * math.fabs(kp_a.polar[m.queryIdx][0] - math.pi / 2) / (math.pi / 2):
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


class FeatureMatcher2():
    def __init__(self):
        pass


class FeatureMatcher4(FeatureMatcher):
    def __init__(self, imgs_left, imgs_right, debug):
        super().__init__()
        self._imgs_left = imgs_left
        self._imgs_right = imgs_right
        self._threshold = 0.075
        self._debug = debug

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
        return pts.reshape((kp_indices.shape[0], 2 * kp_indices.shape[1]))
