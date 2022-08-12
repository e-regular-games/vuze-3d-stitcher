#!/usr/bin/python

# Operates on 180deg FOV equirectangular images only!
import copy
import getopt
import sys
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# scale phi and theta adjust around the center of the original image.
class Transform():
    def __init__(self):
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.rotate_spin = 0.0
        self.rotate_theta = 0.0
        self.rotate_phi = 0.0
        self.scale_phi = 1.0
        self.scale_theta = 1.0
        self.radius_subject = 2.0

    def read_from_string(self, s):
        opt = s.strip().split(',')
        if len(opt) != 2:
            return

        if opt[0] == 'x':
            self.offset_x = float(opt[1])
        elif opt[0] == 'y':
            self.offset_y = float(opt[1])
        elif opt[0] == 'z':
            self.offset_z = float(opt[1])
        elif opt[0] == 'spin':
            self.rotate_spin = float(opt[1]) / 180.0 * math.pi
        elif opt[0] == 'theta':
            self.rotate_theta = float(opt[1]) / 180.0 * math.pi
        elif opt[0] == 'phi':
            self.rotate_phi = float(opt[1]) / 180.0 * math.pi
        elif opt[0] == 'scale_phi':
            self.scale_phi = float(opt[1])
        elif opt[0] == 'scale_theta':
            self.scale_theta = float(opt[1])

    def adjust_cart(self, c):
        # adjust to local image coordinates based on the center of the image
        #dx = self.offset_x * math.cos(self.rotate_theta) - self.offset_y * math.sin(self.rotate_theta) + self.offset_z * math.sin(self.rotate_phi) * math.cos(self.rotate_theta)
        #dy = self.offset_y * math.cos(self.rotate_theta) - self.offset_x * math.sin(self.rotate_theta) + self.offset_z * math.sin(self.rotate_phi) * math.sin(self.rotate_theta)
        #dz = self.offset_z * math.cos(self.rotate_phi)
        #cf = (c[0] + dx, c[1] + dy, c[2] + dz)
        cf = c
        if self.rotate_spin != 0:
            r = np.sqrt(cf[1] * cf[1] + cf[2] * cf[2])
            a = np.arctan2(cf[2], cf[1]) + self.rotate_spin
            cf = (cf[0], r * np.cos(a), r * np.sin(a))
        return cf

    def adjust_polar(self, c):
        phi = c[0] % math.pi
        phi = phi + self.rotate_phi
        phi = (phi - math.pi / 2) / self.scale_phi + math.pi / 2
        theta = c[1] % (2 * math.pi)
        theta = theta + self.rotate_theta
        theta = (theta - math.pi) / self.scale_theta + math.pi
        return (phi, theta)

class ProgramOptions:

    def __init__(self):
        self.verbose = False
        self.config = ""

        options, arguments = getopt.getopt(
            sys.argv[1:],
            'vc:',
            ["config=", "verbose"])

        for o, a in options:
            if o in ("-c", "--config"):
                self.config = a
            elif o in ("-v", "--verbose"):
                self.verbose = True

    def valid(self):
        return len(self.config) > 0


class Config:

    def __init__(self, file_path):
        self.input = ""
        self.output = ""
        self.resolution_v = 1080
        self.transforms = [Transform() for _ in range(8)]
        self.splices = []
        self.offset_left = 0
        self.offset_right = 0

        f = open(file_path, 'r')
        for l in f.readlines():
            cmd = l.strip().split(',')
            if cmd[0] == 'in' and len(cmd) == 2:
                self.input = cmd[1]
            elif cmd[0] == 'out' and len(cmd) == 2:
                self.output = cmd[1]
            elif cmd[0] == 'transform' and len(cmd) == 4:
                self.transforms[int(cmd[1]) - 1].read_from_string(','.join(cmd[2:]))
            elif cmd[0] == 'resolution' and len(cmd) == 2:
                self.resolution_v = int(cmd[1])
            elif cmd[0] == 'splices':
                self.splices.append(float(cmd[-1]) / 180.0 * math.pi - 2 * math.pi)
                for s in cmd[1:]:
                    self.splices.append(float(s) / 180.0 * math.pi)
            elif cmd[0] == 'offset_right':
                self.offset_right = float(cmd[1]) / 180.0 * math.pi
            elif cmd[0] == 'offset_left':
                self.offset_left = float(cmd[1]) / 180.0 * math.pi


    def valid(self):
        return self.input != '' and self.output != ''

# assumes the height is 180deg of image and
def eqr_to_polar(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    phi = (c[1] / h) * math.pi;
    theta = c[0] / w * shape[1] / shape[0] * math.pi;
    return phi, theta

def polar_to_eqr(c, shape):
    w = shape[1] - 1
    h = shape[0] - 1
    y = h * (c[0] / math.pi)
    x = w * c[1] / math.pi * shape[0] / shape[1]
    return x, y

def polar_to_cart(c, r):
    x = r * np.sin(c[0]) * np.cos(c[1])
    y = r * np.sin(c[0]) * np.sin(c[1])
    z = r * np.cos(c[0])
    return x, y, z

def cart_to_polar(c):
    theta = np.arctan2(c[1], c[0])
    phi = np.arctan2(np.sqrt(c[0]*c[0] + c[1]*c[1]), c[2])
    return phi, theta

def get_pixel(im, shape, c):
    w = shape[1] - 1
    h = shape[0] - 1
    x = c[0].astype(np.int)
    y = c[1].astype(np.int)
    x[x < 0] += w
    x[x > w] -= w
    y[y < 0] += h
    y[y > h] -= h
    return im[y * shape[1] + x]

def interp(im, c):
    weights = []
    weights.append((np.floor(c[0]), np.floor(c[1])))
    weights.append((np.floor(c[0]), np.floor(c[1] + 1)))
    weights.append((np.floor(c[0] + 1), np.floor(c[1])))
    weights.append((np.floor(c[0] + 1), np.floor(c[1] + 1)))

    r = np.zeros((c[0].size, im.shape[2]), dtype=np.uint8)
    im_1d = im.reshape((im.shape[0] * im.shape[1], im.shape[2]))
    for w in weights:
        dx = 1.0 - np.absolute(c[0] - w[0])
        dy = 1.0 - np.absolute(c[1] - w[1])
        px = get_pixel(im_1d, im.shape, w)
        for ch in range(im.shape[2]):
            r[:,ch] += np.floor(dx * dy * px[:,ch]).astype(np.uint8)
    return r

class ImageRemap():
    def __init__(self, img, t):
        self.image = img
        self.transform = t

    def interp(self, polar_points):
        # polar_points a list of numpy.array objects
        # the first of which is phi coordinates, the 2nd is theta coordinates
        # returns the image after applying the appropriate transforms.
        p_f = polar_points
        c_f = polar_to_cart(p_f, self.transform.radius_subject)
        c_0 = self.transform.adjust_cart(c_f)
        p_0 = self.transform.adjust_polar(cart_to_polar(c_0))
        eq_0 = polar_to_eqr(p_0, self.image.shape)
        return interp(self.image, eq_0)

options = ProgramOptions()
if not options.valid():
    exit(1)

config = Config(options.config)
if not config.valid():
    exit(1)

print('loading images')
images = []
for l in range(1, 9):
    img = cv.imread(config.input + '_' + str(l) + '_eq360.JPG')
    img_remap = ImageRemap(img, config.transforms[l - 1])
    images.append(img_remap)

print('computing pixel locations')
output_shape = (config.resolution_v, 2 * config.resolution_v, 3)
px_count = output_shape[1] * output_shape[0]

eq_f = (np.zeros(px_count, dtype=np.float), np.zeros(px_count, dtype=np.float))

x_range = range(output_shape[1])
y_range = range(output_shape[0])
for y in y_range:
    for x in x_range:
        i = y * len(x_range) + x
        eq_f[0][i] = x
        eq_f[1][i] = y

left_sphere = images[0:7:2]
right_sphere = images[1:8:2]
p_f = eqr_to_polar(eq_f, output_shape)
result_left = np.zeros((px_count, 3), dtype=np.uint8)
result_right = np.zeros((px_count, 3), dtype=np.uint8)

def filter_to_splice(polar, s0, s1, center):
    indices = np.arange(polar[0].size)
    phi = polar[0]
    theta = polar[1].copy()

    cap_idx = None
    if s1 > 2 * math.pi and s0 < 2 * math.pi:
        s1 -= 2 * math.pi
        cap_idx = np.logical_or(theta < s1, theta >= s0)
        theta[theta < s1] += (2 * math.pi - center)
        theta[theta >= s0] += (s0 - center)
    elif s0 < 0 and s1 > 0:
        s0 += 2 * math.pi
        cap_idx = np.logical_or(theta > s0, theta <=  s1)
        theta[theta > s0] -= (2 * math.pi + center)
        theta[theta <= s1] -= center
    else:
        cap_idx = np.logical_and(theta > s0, theta <= s1)
        theta -= center

    return (phi[cap_idx], theta[cap_idx] + math.pi), indices[cap_idx]

for s in range(4):
    print('calculating segment ' + str(s + 1))
    center = s * math.pi / 2
    lpolar, lindices = filter_to_splice(p_f, config.splices[s] + config.offset_left, config.splices[s+1] + config.offset_left, center)
    result_left[lindices] = left_sphere[s].interp(lpolar)
    rpolar, rindices = filter_to_splice(p_f, config.splices[s] + config.offset_right, config.splices[s+1] + config.offset_right, center)
    result_right[rindices] = right_sphere[s].interp(rpolar)


#cv.imwrite(config.output + '_left.JPG', result_left.reshape(output_shape), [int(cv.IMWRITE_JPEG_QUALITY), 100])
#cv.imwrite(config.output + '_right.JPG', result_right.reshape(output_shape), [int(cv.IMWRITE_JPEG_QUALITY), 100])

combined = np.concatenate([result_left.reshape(output_shape), result_right.reshape(output_shape)])
cv.imwrite(config.output + '.JPG', combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])
