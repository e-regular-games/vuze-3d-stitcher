# Operates on 180deg FOV equirectangular images only!

import coordinates
import math
import numpy as np
import cv2 as cv

class FisheyeImage():
    def __init__(self, resolution, debug):

        self._eq_shape = None
        self._pts = None

        if resolution is not None:
            self._eq_shape = (resolution, resolution*2)
            self._pts = coordinates.equirect_points(resolution)

        self._debug = debug

        self._img = None
        self._calib = None


    # calib is the 6 parameters that define the ellipse which encompasses
    # the points within the image that are within the fisheye lense.
    def clone_with_image(self, img, calib):
        c = FisheyeImage(None, self._debug)
        c._pts = self._pts
        c._eq_shape = self._eq_shape

        c._img = img
        c._calib = calib
        return c

    def to_equirect(self):
        if self._calib.vuze_config:
            return self.to_equirect_full()
        else:
            return self.to_equirect_ellipse()

    def to_equirect_ellipse(self):
        polar = coordinates.eqr_to_polar(self._pts, self._eq_shape)
        n = polar.shape[0]
        polar[:,0] = (polar[:,0] * -1) % math.pi
        polar[:,1] = (-1 * polar[:,1] - math.pi / 2) % (2 * math.pi)
        cart = coordinates.polar_to_cart(polar, 1)
        polar = None

        aperture = self._calib.aperture / 180 * math.pi
        r = 2 * np.arctan2(np.sqrt(cart[:,0]*cart[:,0] + cart[:,2]*cart[:,2]), cart[:,1]) / aperture
        a = np.arctan2(cart[:,2], cart[:,0])

        radius = max(self._calib.ellipse[2], self._calib.ellipse[3])
        center = (self._calib.ellipse[0], self._calib.ellipse[1])
        f = np.zeros((n, 2))
        f[:,0] = r * np.cos(a)
        f[:,1] = r * np.sin(a)

        # create the unit vectors that align to the axes of the ellipse
        ellipse_unit = np.array([
            [math.cos(self._calib.ellipse[5]), -math.sin(self._calib.ellipse[5])],
            [math.sin(self._calib.ellipse[5]), math.cos(self._calib.ellipse[5])]
        ])

        # convert the points f into the unit vectors v1, and v2.
        # then scale along the minor axis to get the correct point from the sensor
        c = np.transpose(np.matmul(ellipse_unit, np.transpose(f)))
        c[:,1] *= self._calib.ellipse[3] / self._calib.ellipse[2]

        # convert from the unit vectors v1 and v2, back to the unit vectors along x and y
        p = np.transpose(np.matmul(np.transpose(ellipse_unit), np.transpose(c)))
        x = p[:,0].reshape(self._eq_shape).astype(np.float32) * radius + center[0]
        y = p[:,1].reshape(self._eq_shape).astype(np.float32) * radius + center[1]

        return cv.remap(self._img, x, y, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0).astype(np.uint8)

    def to_equirect_full(self):
        polar = coordinates.eqr_to_polar(self._pts, self._eq_shape)

        n = polar.shape[0]
        polar[:,0] = (polar[:,0]) % math.pi
        polar[:,1] = (polar[:,1] - math.pi / 2) % (2 * math.pi)
        # adjust from the scripts perception of the world axis, y-front to the vuze camera
        # perception which is y-up.
        cart = np.matmul(coordinates.polar_to_cart(polar, 1), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
        polar = None

        camera = cv.fisheye.projectPoints(cart.reshape(self._eq_shape[0], self._eq_shape[1], 3), \
                                          cv.Rodrigues(self._calib.R)[0], \
                                          np.zeros((3)), #self._calib.t, \
                                          self._calib.camera_matrix, \
                                          self._calib.k_coeffs)[0].astype(np.float32)
        cart = None

        # convert camera to be centered at 0
        center = np.array([self._calib.camera_matrix[0,2], self._calib.camera_matrix[1,2]])
        f = camera.reshape((n, 2)) - center
        camera = None

        # create the unit vectors that align to the axes of the ellipse
        #ellipse_unit = np.array([
        #    [math.cos(self._calib.ellipse[5]), -math.sin(self._calib.ellipse[5])],
        #    [math.sin(self._calib.ellipse[5]), math.cos(self._calib.ellipse[5])]
        #])

        # convert the points f into the unit vectors v1, and v2.
        # then scale along the minor axis to get the correct point from the sensor
        #f = np.transpose(np.matmul(ellipse_unit, np.transpose(f)))
        #f[:,1] *= self._calib.ellipse[3] / self._calib.ellipse[2]

        # convert from the unit vectors v1 and v2, back to the unit vectors along x and y
        #f = np.transpose(np.matmul(np.transpose(ellipse_unit), np.transpose(f)))
        flipped = np.zeros((n), dtype=np.bool)
        flipped[int(0.6 * n):] = f[int(0.6 * n):,1] < 0
        flipped[:int(0.4 * n)] = f[:int(0.4 * n),1] > 0

        x = f[:,0].reshape(self._eq_shape).astype(np.float32) + center[0]
        y = f[:,1].reshape(self._eq_shape).astype(np.float32) + center[1]

        eqr = cv.remap(self._img, x, y, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0).astype(np.uint8)
        eqr[flipped.reshape(self._eq_shape)] = 0

        if self._calib.depth:
            eqr = self._calib.depth.apply(eqr)

        return eqr
