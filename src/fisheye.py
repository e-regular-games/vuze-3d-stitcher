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
        self._radius = 0
        self._center = None
        self._aperture = math.pi


    def clone_with_image(self, img, center, radius, aperture):
        c = FisheyeImage(None, self._debug)
        c._pts = self._pts
        c._eq_shape = self._eq_shape

        c._img = img
        c._center = center
        c._radius = radius
        c._aperture = aperture / 180 * math.pi
        return c

    def to_equirect(self):
        polar = coordinates.eqr_to_polar(self._pts, self._eq_shape)
        polar[:,0] = (polar[:,0] * -1) % math.pi
        polar[:,1] = (-1 * polar[:,1] - math.pi / 2) % (2 * math.pi)
        cart = coordinates.polar_to_cart(polar, 1)
        polar = None

        r = 2 * np.arctan2(np.sqrt(cart[:,0]*cart[:,0] + cart[:,2]*cart[:,2]), cart[:,1]) / self._aperture
        a = np.arctan2(cart[:,2], cart[:,0])
        x = (r * np.cos(a) * self._radius + self._center[0]).reshape(self._eq_shape).astype(np.float32)
        y = (r * np.sin(a) * self._radius + self._center[1]).reshape(self._eq_shape).astype(np.float32)

        return cv.remap(self._img, x, y, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0).astype(np.uint8)
