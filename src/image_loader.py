#!/usr/bin/python

import fisheye
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure
import threading
from depth_mesh import DepthCalibration

def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

def plot_lenses(images, title):
    f, axs = plt.subplots(2, 4, sharex=True, sharey=True)
    f.canvas.manager.set_window_title(title)
    for i, img in enumerate(images):
        axs[int(i/4), i%4].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axs[int(i/4), i%4].axes.xaxis.set_ticklabels([])
        axs[int(i/4), i%4].axes.yaxis.set_ticklabels([])
    return axs

class CalibrationParams():
    def __init__(self, debug):
        self.aperture = 180 #degrees
        self._default_size = (1088, 1600)

        self.recalc_ellipse = False
        self.ellipse = None # (x0, y0, a, b, eccentricity, rotation)
        self.lens_pixels = None

        self.depth = None

        # params read from vuze camera yaml file.
        self.vuze_config = False
        self.camera_matrix = None
        self.t = None
        self.R = None
        self.k_coeffs = None
        self.fov = None
        self.radius = None

        self._debug = debug

    def to_dict(self):
        d = {'aperture': self.aperture}
        if self.ellipse is not None:
            d['ellipse'] = self.ellipse

        if self.vuze_config:
            d['cameraMatrix'] = self.camera_matrix.tolist()
            d['t'] = self.t.tolist()
            d['R'] = self.R.tolist()
            d['kCoeffs'] = self.k_coeffs.tolist()
            d['fov'] = self.fov
            d['radius'] = self.radius
            d['vuzeConfig'] = True

        if self.depth:
            d['depth'] = self.depth.to_dict()

        return d

    def from_dict(self, d):
        self.aperture = d['aperture']
        if 'ellipse' in d:
            self.ellipse = tuple(d['ellipse'])

        if 'vuzeConfig' in d and d['vuzeConfig']:
            self.camera_matrix = np.array(d['cameraMatrix'])
            self.t = np.array(d['t'])
            self.R = np.array(d['R'])
            self.k_coeffs = np.array(d['kCoeffs'])
            self.fov = d['fov']
            self.radius = d['radius']
            self.vuze_config = True

        if 'depth' in d:
            self.depth = DepthCalibration().from_dict(d['depth'])

        return self

    def from_yaml(self, y, i):
        a = -math.pi / 2 * int(i / 2)
        Rr = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])

        self.vuze_config = True
        self.camera_matrix = y.getNode('K').mat()
        self.t = np.matmul(Rr, y.getNode('CamCenter').mat() / 1000)
        self.k_coeffs = np.array([y.getNode('DistortionCoeffs_' + str(i)).real() for i in range(2, 10, 2)])
        self.R = np.matmul(Rr, y.getNode('R').mat())
        self.fov = y.getNode('Fov_deg').real()
        self.radius = y.getNode('ImageCircleRadius').real()

        if self.ellipse:
            self._debug.log('overwritting lens center with ellipse configuration', \
                            '(', self.camera_matrix[0,2], self.camera_matrix[1,2], ')', \
                            '(', self.ellipse[0], self.ellipse[1], ')')
            self.camera_matrix[0,2] = self.ellipse[0]
            self.camera_matrix[1,2] = self.ellipse[1]

        return self

    def _get_ellipse_pts(self, params, npts=100, tmin=0, tmax=2*np.pi):
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.

        """

        x0, y0, ap, bp, e, phi = params
        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return x, y

    def _fit_ellipse(self, x, y):
        """
        https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
        """

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel()

    def _cart_to_pol(self, coeffs):
        """
        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.
        """

        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')

        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi

        return x0, y0, ap, bp, e, phi

    def empty(self):
        return self.ellipse is None and not self.vuze_config

    def plot(self, ax):
        if self.lens_pixels is not None:
            ax.imshow(cv.cvtColor(self.lens_pixels, cv.COLOR_BGR2RGB))
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])

        if self.ellipse is not None:
            x, y = self._get_ellipse_pts(self.ellipse)
            ax.plot(x, y)
            ax.set_xlim(0, self._default_size[0])
            ax.set_ylim(0, self._default_size[1])

    # an already correctly oriented image
    def from_image(self, img):
        self._default_size = (img.shape[1], img.shape[0])

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        thres = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv.THRESH_BINARY, 11, 2)
        thres = cv.GaussianBlur(thres, (23,23), sigmaX=0, sigmaY=0)
        thres = cv.threshold(thres, 170, 255, cv.THRESH_BINARY)[1]

        # create an (img.shape[:-1]) of short int aligned
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0]).reshape(img.shape[0],1) * np.ones(img.shape[:-1])

        y[thres == 255] = np.Inf
        top = np.min(y, axis=0)
        y[thres == 255] = np.NINF
        bottom = np.max(y, axis=0)

        points = np.zeros((2 * img.shape[1], 2))
        points[:img.shape[1],0] = x
        points[:img.shape[1],1] = top
        points[img.shape[1]:,0] = x
        points[img.shape[1]:,1] = bottom

        coeffs = self._fit_ellipse(points[:,0], points[:,1])

        self.ellipse = self._cart_to_pol(coeffs)
        self.lens_pixels = thres

        # set the center of the lens into the camera_matrix
        if self.camera_matrix is not None:
            self._debug.log('overwritting lens center with ellipse configuration', \
                            '(', self.camera_matrix[0,2], self.camera_matrix[1,2], ')', \
                            '(', self.ellipse[0], self.ellipse[1], ')')
            self.camera_matrix[0,2] = self.ellipse[0]
            self.camera_matrix[1,2] = self.ellipse[1]

class LoadImage(threading.Thread):
    def __init__(self, fish, calib, f):
        threading.Thread.__init__(self)
        self._fish = fish
        self._f = f

        self.result = None
        self.calib = calib

    def run(self):
        img = cv.imread(self._f + '.JPG')
        img = np.rot90(img)
        if self.calib.recalc_ellipse:
            self.calib.from_image(img)
            self.calib.recalc_ellipse = False
        fish = self._fish.clone_with_image(img, self.calib)
        self.result = fish.to_equirect()
        print('.', end='', flush=True)

class ImageLoader:
    def __init__(self, config, debug):
        self._config = config
        self._debug = debug
        self._fish = fisheye.FisheyeImage(config.resolution, debug)
        self._calib = [CalibrationParams(debug) for i in range(8)]

    def get_calibration(self):
        return self._calib

    def load(self, calib=None):
        if calib is not None and len(calib) == 8:
            self._calib = calib

        print('loading images')
        images = self._load_images(self._config.input)
        if self._debug.enable('fisheye'): plot_lenses(images, 'Equirectangular')
        if self._debug.enable('fisheye-fio'):
            for i, img in enumerate(images):
                cv.imwrite('equirect_' + str(i) + '.JPG', img)

        if len(self._config.exposure_fuse) > 0:
            images = self._fuse_exposures(images)

        if len(self._config.super_res) > 0:
            images = self._super_resolution(images, self._config.super_res)

        if len(self._config.super_res_buckets) > 0:
            images = self._super_resolution_buckets()

        if self._config.exposure_match != 0:
            images = self._match_exposure(images)

        if len(self._config.denoise) == 3:
            images = self._denoise(images)

        for i, img in enumerate(images):
            images[i] = create_from_middle(img)

        return images

    def _load_images(self, f, parallel=2):
        threads = []
        images = []

        self._debug.log_pause()
        for l in range(1, 9):
            if len(threads) >= parallel:
                threads[0].join()
                images.append(threads[0].result)
                threads = threads[1:]

            t = LoadImage(self._fish, self._calib[l-1], f + '_' + str(l))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
            images.append(t.result)

        print('')
        self._debug.log_resume()

        if self._debug.enable('calib'):
            f, axs = plt.subplots(2, 4, sharex=True, sharey=True)

            for i, c in enumerate(self._calib):
                c.plot(axs[int(i/4), i%4])

        if self._debug.enable('calib-fio'):
            for i, c in enumerate(self._calib):
                if c.lens_pixels is not None:
                    cv.imwrite('lens_alignment_in_lens_' + str(i) + '.png', c.lens_pixels)

        return images

    def _match_exposure(self, images):
        print('matching exposure')
        # match histograms using the reference image.
        ref = images[self._config.exposure_match]
        for i in range(len(images)):
            print('.', end='', flush=True)
            if i != (self._config.exposure_match - 1):
                images[i] = exposure.match_histograms(images[i], ref, channel_axis=2)
        if self._debug.enable('exposure'): plot_lenses(images, 'Exposures Matched')
        print('')

        return images

    def _denoise(self, images):
        print('denoising images')
        for i, img in enumerate(images):
            images[i] = cv.fastNlMeansDenoisingColored(img, None, \
                                                       self._config.denoise[0], \
                                                       self._config.denoise[0], \
                                                       self._config.denoise[1], \
                                                       self._config.denoise[2])
            print('.', end='', flush=True)
        print('')
        if self._debug.enable('denoise'): plot_lenses(images, 'Denoise')
        return images

    def _fuse_exposures(self, images):
        print('fusing exposures')
        for l in range(1, 9):
            images_exp = [images[l-1]]
            for exp in self._config.exposure_fuse:
                img = cv.imread(exp + '_' + str(l) + '.JPG')
                img = np.rot90(img)
                fish_ = self._fish.clone_with_image(img, self._calib[l-1])
                images_exp.append(fish_.to_equirect())

            alignMTB = cv.createAlignMTB()
            alignMTB.process(images_exp, images_exp)
            mergeMertens = cv.createMergeMertens()
            images[l-1][...,:3] = \
                np.clip(mergeMertens.process([i[...,:3] for i in images_exp]) * 255, 0, 255) \
                  .round().astype(np.uint8)
            for i in images_exp:
                images[l-1][...,3:4] *= i[...,3:4]
            print('.', end='', flush=True)
        if self._debug.enable('exposure'): plot_lenses(images, 'Exposures Fused')
        print('')
        return images

    def _super_resolution(self, images, names):
        per_lens = []
        n = len(names) + 1
        for i, img in enumerate(images):
            per_lens.append(np.zeros((n,) + img.shape, np.uint8))
            per_lens[i][0] = img
        images = None

        for f, name in enumerate(names):
            print(str(f+2) + '/' + str(n) + ': ', end='', flush=True)
            for i, img in enumerate(self._load_images(name)):
                per_lens[i][f+1] = img

        print('super resolution merge')
        images = []
        for l, imgs in enumerate(per_lens):
            alignMTB = cv.createAlignMTB()
            imgs_list = []
            for i in range(imgs.shape[0]):
                imgs_list.append(imgs[i])
            alignMTB.process(imgs_list, imgs_list)
            for i in range(imgs.shape[0]):
                imgs[i] = imgs_list[i]

            imgs_list = None
            per_lens[l] = None

            if 'outlier_limit' in self._config.super_res_config:
                limit = float(self._config.super_res_config['outlier_limit'])
                avg = np.median(imgs, axis=0)[...,:3]
                std = np.std(imgs, axis=0)[...,:3]
                for i in range(n):
                    outliers = np.any(np.logical_or(imgs[i,...,:3] > avg + limit * std, \
                                                    imgs[i,...,:3] < avg - limit * std), axis=-1)
                    imgs[i,outliers,:3] = avg[outliers]
                    outliers = None

                avg = None
                std = None

            result = np.mean(imgs, axis=0).round().astype(np.uint8)

            if 'sharpen' in self._config.super_res_config \
               and self._config.super_res_config['sharpen'] == 'true':
                sharpen = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                result = cv.filter2D(src=result, ddepth=-1, kernel=sharpen)

            images.append(result)
            print('.', end='', flush=True)
        print('')

        return images

    def _super_resolution_buckets(self):
        highres = {}
        for b, names in self._config.super_res_buckets.items():
            images = self._load_images(names[0])
            images = self._super_resolution(images, names[1:])
            highres[b] = images

        print('super resolution bucket merge')
        images = []
        for l in range(0, 8):
            lens = []
            for b, lenses in highres.items():
                lens.append(lenses[l])

            alignMTB = cv.createAlignMTB()
            alignMTB.process(lens, lens)
            mergeMertens = cv.createMergeMertens()
            lens = np.clip(mergeMertens.process(lens) * 255, 0, 255).round().astype(np.uint8)
            images.append(lens)
            print('.', end='', flush=True)

        print('')
        return images
