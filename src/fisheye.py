# Operates on 180deg FOV equirectangular images only!

import coordinates
import math
import numpy as np
import cv2 as cv

class FisheyeImage():
    def __init__(self, debug):
        self._img = None
        self._pts = None
        self._eq_shape = None
        self._radius = 0
        self._center = None
        self._aperture = math.pi
        self._restrict_to_image_angle = False
        self._debug = debug

    def set_image(self, img, center, radius, aperture):
        self._img = img
        self._center = center
        self._radius = radius
        self._aperture = aperture / 180 * math.pi

    def set_output_resolution(self, resolution):
        self._eq_shape = (resolution, resolution*2)
        self._pts = coordinates.equirect_points(resolution)

    def restrict_to_image_angle(self, b):
        self._restrict_to_image_angle = b

    def _restrict_equirect(self, eqr):
        # find the left/right side as a normalized radius,
        l = (self._center[0] + (self._radius - self._center[0]) * 0.75) / self._radius
        r = (self._img.shape[1] - self._center[0])
        r = (r + (self._radius - r) * 0.75) / self._radius

        pl = [math.sin(-l * self._aperture / 2), math.cos(-l * self._aperture / 2), 0]
        pr = [math.sin(r * self._aperture / 2), math.cos(r *self._aperture / 2), 0]

        lx = int(eqr.shape[1] / 2 * (1.0 - math.atan2(pl[1], pl[0]) / math.pi + 0.5))
        rx = int(eqr.shape[1] / 2 * (1.0 - math.atan2(pr[1], pr[0]) / math.pi + 0.5))

        if self._debug.verbose: print('fisheye limits:' , lx, rx)
        result = eqr.copy()
        result[:,:lx,:] = 0
        result[:,rx:,:] = 0
        return result

    def to_equirect(self):
        polar = coordinates.eqr_to_polar(self._pts, self._eq_shape)
        polar[:,0] = (polar[:,0] * -1) % math.pi
        polar[:,1] = (-1 * polar[:,1] - math.pi / 2) % (2 * math.pi)
        cart = coordinates.polar_to_cart(polar, 1)

        r = 2 * np.arctan2(np.sqrt(cart[:,0]*cart[:,0] + cart[:,2]*cart[:,2]), cart[:,1]) / self._aperture
        a = np.arctan2(cart[:,2], cart[:,0])
        x = (r * np.cos(a) * self._radius + self._center[0]).reshape(self._eq_shape).astype(np.float32)
        y = (r * np.sin(a) * self._radius + self._center[1]).reshape(self._eq_shape).astype(np.float32)
        eqr = cv.remap(self._img, x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

        if not self._restrict_to_image_angle:
            return eqr

        return self._restrict_equirect(eqr)
