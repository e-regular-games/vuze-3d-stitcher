"""
A collection of tools for converting between image coordinate systems.
S. Ryan Edgar
August 1st, 2022
"""

import math
import numpy as np
import cv2 as cv

def to_1d(a):
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])

def to_2d(a):
    return a.reshape((a.shape[0],) + (1,) + a.shape[1:])

# angle between 2 vectors
def angle(v1, v2):
    d = np.sqrt(np.sum(v1*v1, axis=-1)) * np.sqrt(np.sum(v2*v2, axis=-1))
    return np.arccos(np.sum(v1 * v2, axis=-1) / d).reshape(v1.shape[:-1] + (1,))

# @param c a matrix with N rows and 2 columns, (x, y)
# @param shape a tuple with at least 2 values, (height, width)
# @returns a matrix with N rows and 2 columns, (phi, theta)
def eqr_to_polar(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    phi = (c[:, 1:2] / h) * math.pi;
    theta = c[:, 0:1] / w * shape[1] / shape[0] * math.pi;
    return np.concatenate([phi, theta], axis=-1)

# @param c a 3D matrix with N rows and M columns of (x, y) pairs
# @returns a 3D matrix with N rows and M columns of (phi, theta) pairs
def eqr_to_polar_3d(c):
    w = c.shape[1] - 1
    h = c.shape[0] - 1
    phi = (c[:,:,1:2] / h) * math.pi;
    theta = c[:,:,0:1] / w * c.shape[1] / c.shape[0] * math.pi;
    return np.concatenate([phi, theta], axis=-1)

# @param c a matrix with N rows and 2 columns, (phi, theta)
# @param r the radius to use during the conversion to cartesian.
# @returns a matrix with N rows and 3 columns, (x, y, z)
def polar_to_cart(c, r):
    res = np.zeros(c.shape[:-1] + (3,))
    res[...,0] = r * np.sin(c[...,0]) * np.cos(c[...,1])
    res[...,1] = r * np.sin(c[...,0]) * np.sin(c[...,1])
    res[...,2] = r * np.cos(c[...,0])
    return res

# @param c a matrix with N rows and 2 columns, (phi, theta)
# @param shape a tuple with at least 2 values, (height, width)
# @returns a matrix with N rows and 2 columns, (x, y)
def polar_to_eqr(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    r = c.copy()
    r[..., 0] = w * c[..., 1] / math.pi * shape[0] / shape[1]
    r[..., 1] = h * (c[..., 0] / math.pi)
    return r

# @param c a 3d matrix with N rows and M columns of (phi, theta) pairs
# @returns a 3d matrix with N rows and M columns of (x, y) pairs
def polar_to_eqr_3d(c, shape=None):
    shape = c.shape if shape is None else shape
    w = shape[1] - 1
    h = shape[0] - 1
    res = np.zeros(c.shape, np.float32)
    res[..., 0:1] = w * c[..., 1:2] / math.pi * shape[0] / shape[1]
    res[..., 1:2] = h * c[..., 0:1] / math.pi
    return res

# @param c a matrix with N rows and 3 columns, (x, y, z)
# @returns a matrix with N rows and 2 columns, (phi, theta)
def cart_to_polar(c):
    r = np.zeros(c.shape[:-1] + (2,))
    r[...,1] = np.arctan2(c[...,1], c[...,0])
    r[...,0] = np.arctan2(np.sqrt(c[...,0]*c[...,0] + c[...,1]*c[...,1]), c[...,2])
    return r

# dims is a tuple (height, width),
# the PI radians longitude will be in the center of width
# height is assumed to be PI radians
def polar_points_3d(dims):
    w = dims[1]
    h = dims[0]

    phi = np.linspace(0, math.pi, h)
    theta = np.linspace(math.pi * (1 - w/h/2), math.pi * (1 + w/h/2), w + 1)[:-1]

    res = np.zeros((h, w, 2), np.float32)
    res[...,1], res[...,0] = np.meshgrid(theta, phi)
    return res


# @returns a matrix of N rows by 2 columns, (x, y) where each row
#   represents the pixel coordinate within an equirectangular image with
#   verticle resolution
def equirect_points(resolution):
    width = resolution * 2
    height = resolution
    shape = (width*height, 2)
    eq = np.zeros(shape, np.float32)

    for y in range(height):
        for x in range(width):
            i = y * width + x
            eq[i, 0] = x
            eq[i, 1] = y

    return eq

# determine the theta value for each phi at which the seam
# is intersected.
def seam_intersect(seam, phi):

    # Ensure the seam is monotonically increasing along phi
    s_phi = seam[:, 0]
    inc = np.ones((len(s_phi),), np.bool)
    v = s_phi[0]
    for i, p in enumerate(s_phi[1:]):
        inc[i+1] = p > v
        if p > v:
            v = p
    seam = seam[inc]

    s_phi = seam[:, 0]
    s_theta = seam[:, 1]
    slope = (s_theta[1:] - s_theta[:-1]) / (s_phi[1:] - s_phi[:-1])
    offset = s_theta[:-1] - slope * s_phi[:-1]

    phi = phi.reshape((phi.shape[0], 1))
    theta = np.zeros((phi.shape[0]), np.float32)
    phi_n = phi.shape[0]
    slope_n = slope.shape[0]

    f_mat = np.ones((phi_n, slope_n + 1), np.float32) * seam[:, 0]
    in_range = np.logical_and(phi < f_mat[:,1:], phi >= f_mat[:,:-1])
    f_slope = (np.ones((phi_n, slope_n)) * slope)[in_range]
    f_offset = (np.ones((phi_n, offset.shape[0])) * offset)[in_range]

    in_range = np.any(in_range, axis=1)
    theta[in_range] = phi[in_range,0] * f_slope + f_offset
    return theta

def eqr_interp(eqr, img, method=cv.INTER_CUBIC):
    l = eqr.shape[0]
    s = math.floor(math.sqrt(l) + 1)
    padding = np.zeros(s*s - l, dtype=np.float32)

    x = np.concatenate([eqr[:, 0], padding]).reshape(s, s).astype(np.float32)
    y = np.concatenate([eqr[:, 1], padding]).reshape(s, s).astype(np.float32)

    pixels = cv.remap(img, x, y, method, borderMode=cv.BORDER_WRAP)
    if len(img.shape) > 2:
        return pixels.reshape(s * s, img.shape[-1])[:l]
    return pixels.reshape(s * s)[:l]

def eqr_interp_3d(eqr, img, method=cv.INTER_CUBIC):
    return cv.remap(img, \
                    eqr[...,:,0].astype(np.float32), \
                    eqr[...,:,1].astype(np.float32), \
                    method)
