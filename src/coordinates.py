"""
A collection of tools for converting between image coordinate systems.
S. Ryan Edgar
August 1st, 2022
"""

import math
import numpy as np
import cv2 as cv

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
    res = np.zeros((c.shape[0], 3))
    res[:,0] = r * np.sin(c[:,0]) * np.cos(c[:,1])
    res[:,1] = r * np.sin(c[:,0]) * np.sin(c[:,1])
    res[:,2] = r * np.cos(c[:,0])
    return res

# @param c a matrix with N rows and 2 columns, (phi, theta)
# @param shape a tuple with at least 2 values, (height, width)
# @returns a matrix with N rows and 2 columns, (x, y)
def polar_to_eqr(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    r = c.copy()
    r[:, 0] = w * c[:, 1] / math.pi * shape[0] / shape[1]
    r[:, 1] = h * (c[:, 0] / math.pi)
    return r

# @param c a 3d matrix with N rows and M columns of (phi, theta) pairs
# @returns a 3d matrix with N rows and M columns of (x, y) pairs
def polar_to_eqr_3d(c):
    w = c.shape[1] - 1
    h = c.shape[0] - 1
    x = w * c[:,:,1:2] / math.pi * c.shape[0] / c.shape[1]
    y = h * (c[:,:,0:1] / math.pi)
    return np.concatenate([x, y], axis=-1)

# @param c a matrix with N rows and 3 columns, (x, y, z)
# @returns a matrix with N rows and 2 columns, (phi, theta)
def cart_to_polar(c):
    r = np.zeros((c.shape[0], 2))
    r[:,1] = np.arctan2(c[:,1], c[:,0])
    r[:,0] = np.arctan2(np.sqrt(c[:,0]*c[:,0] + c[:,1]*c[:,1]), c[:,2])
    return phi, theta

# @returns a matrix of N rows by 2 columns, (x, y) where each row
#   represents the pixel coordinate within an equirectangular image with
#   verticle resolution
def equirect_points(resolution):
    width = resolution * 2
    height = resolution
    shape = (width*height, 2)
    eq = np.zeros(shape)

    for y in range(height):
        for x in range(width):
            i = y * width + x
            eq[i, 0] = x
            eq[i, 1] = y

    return eq

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
    return cv.remap(img, eqr[...,:,0], eqr[...,:,1], method, borderMode=cv.BORDER_WRAP)
