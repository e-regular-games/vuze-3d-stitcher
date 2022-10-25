#!/usr/bin/python

import fisheye
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure
import threading

def get_middle(img):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    return img[:,middle]

def set_middle(img, value):
    width = img.shape[1]
    middle = range(int(width/4), int(width*3/4))
    img[:,middle] = value

def create_from_middle(middle):
    w = middle.shape[1] * 2
    r = np.zeros((middle.shape[0], w, middle.shape[2]), np.uint8)
    r[:,int(w/4):int(3*w/4)] = middle
    return r

def plot_lenses(images, title):
    f, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    f.canvas.manager.set_window_title(title)
    for i, img in enumerate(images):
        axs[int(i/3), i%3].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axs[int(i/3), i%3].axes.xaxis.set_ticklabels([])
        axs[int(i/3), i%3].axes.yaxis.set_ticklabels([])

class LoadImage(threading.Thread):
    def __init__(self, fish, config, f, l):
        threading.Thread.__init__(self)
        self._fish = fish
        self._config = config
        self._f = f
        self._l = l

        self.result = None

    def run(self):
        img = cv.imread(self._f + '_' + str(self._l) + '.JPG')
        img = np.rot90(img)
        fish = self._fish.clone_with_image(img, \
                                           self._config.lens_centers[self._l-1], \
                                           self._config.radius, \
                                           self._config.aperture)
        self.result = get_middle(fish.to_equirect())
        print('.', end='', flush=True)

class ImageLoader:
    def __init__(self, config, debug):
        self._config = config
        self._debug = debug
        self._fish = fisheye.FisheyeImage(config.resolution, debug)

    def load(self):
        print('loading images')
        images = self._load_images(self._config.input)
        if self._debug.enable('fisheye'): plot_lenses(images, 'Equirectangular')

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

    def _load_images(self, f, parallel=4):
        threads = []
        images = []

        for l in range(1, 9):
            if len(threads) >= parallel:
                threads[0].join()
                images.append(threads[0].result)
                threads = threads[1:]

            t = LoadImage(self._fish, self._config, f, l)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
            images.append(t.result)

        print('')
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
                fish_ = self._fish.clone_with_image(img, \
                                                    self._config.lens_centers[l-1], \
                                                    self._config.radius, \
                                                    self._config.aperture)
                images_exp.append(get_middle(fish_.to_equirect()))

            alignMTB = cv.createAlignMTB()
            alignMTB.process(images_exp, images_exp)
            mergeMertens = cv.createMergeMertens()
            images[l-1] = np.clip(mergeMertens.process(images_exp) * 255, 0, 255).round().astype(np.uint8)
            print('.', end='', flush=True)
        if self._debug.enable('exposure'): plot_lenses(images, 'Exposures Fused')
        print('')
        return images

    def _super_resolution(self, images, names):
        per_lens = []
        n = len(names)
        for i, img in enumerate(images):
            per_lens.append(np.zeros((n,) + img.shape, np.uint8))
            per_lens[i][0] = img
        images = None

        for f, name in enumerate(names):
            for i, img in enumerate(self._load_images(name)):
                per_lens[i][f] = img

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
                avg = np.median(imgs, axis=0)
                std = np.std(imgs, axis=0)
                for i in range(n):
                    outliers = np.any(np.logical_or(imgs[i] > avg + limit * std, \
                                                    imgs[i] < avg - limit * std), axis=-1)
                    imgs[i, outliers] = avg[outliers]
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
