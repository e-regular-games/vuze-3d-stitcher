#!/usr/bin/python
import base64
import math
import numpy as np
import cv2 as cv
import os
from exiftool import ExifToolHelper
from exiftool import ExifTool
from Equirec2Perspec import Equirectangular

class FormatVR():

    # @param left numpy matrix of raw image pixel data
    # @param right numpy matrix of raw image pixel data
    def __init__(self, left, right):
        self._left = left
        self._right = right

        self._exif = ExifToolHelper()

        # The <?xml must be the first line and characters of the string, a new line cannot exist
        # after the third question mark and the <?xml
        self._gpano = \
"""<?xml version="1.0" encoding="UTF-8"?>
<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Image::ExifTool 12.16">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description xmlns:GImage="http://ns.google.com/photos/1.0/image/" rdf:about="">
         <GImage:Data>__BASE64DATA__</GImage:Data>
         <GImage:Mime>image/jpeg</GImage:Mime>
      </rdf:Description>
      <rdf:Description xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" rdf:about="">
         <GPano:CroppedAreaImageHeightPixels>__CROP_HEIGHT__</GPano:CroppedAreaImageHeightPixels>
         <GPano:InitialViewHeadingDegrees>__HEADING__</GPano:InitialViewHeadingDegrees>
         <GPano:InitialHorizontalFOVDegrees>150</GPano:InitialHorizontalFOVDegrees>
         <GPano:CroppedAreaImageWidthPixels>__CROP_WIDTH__</GPano:CroppedAreaImageWidthPixels>
         <GPano:CroppedAreaLeftPixels>__CROP_LEFT__</GPano:CroppedAreaLeftPixels>
         <GPano:CroppedAreaTopPixels>__CROP_TOP__</GPano:CroppedAreaTopPixels>
         <GPano:FullPanoHeightPixels>__HEIGHT__</GPano:FullPanoHeightPixels>
         <GPano:FullPanoWidthPixels>__WIDTH__</GPano:FullPanoWidthPixels>
         <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""

        self._meta = {
            'File:ImageWidth': 3840,
            'File:ImageHeight': 3840,
            'EXIF:ExifImageWidth': 3840,
            'EXIF:ExifImageHeight': 3840,
            'XMP:UsePanoramaViewer': True,
            'XMP:CaptureSoftware': '',
            'XMP:StitchingSoftware': 'Vuze Merge by SRE',
            'XMP:ProjectionType': 'equirectangular',
            'XMP:PoseHeadingDegrees': 0,
            'XMP:InitialViewHeadingDegrees': 0,
            'XMP:InitialViewPitchDegrees': 0,
            'XMP:InitialViewRollDegrees': 0,
            'XMP:InitialHorizontalFOVDegrees': 100.0,
            'XMP:CroppedAreaLeftPixels': 0,
            'XMP:CroppedAreaTopPixels': 0,
            'XMP:CroppedAreaImageWidthPixels': 3840,
            'XMP:CroppedAreaImageHeightPixels': 3840,
            'XMP:FullPanoWidthPixels': 3840,
            'XMP:FullPanoHeightPixels': 3840,
            'XMP:SourcePhotosCount': 4,
            'XMP:ExposureLockUsed': False,
            'Composite:ImageSize': '3840 3840',
            'Composite:Megapixels': 14.7456,
        }

    def _set_dimensions(self, width, height):
        m = {
            'File:ImageWidth': width,
            'File:ImageHeight': height,
            'EXIF:ExifImageWidth': width,
            'EXIF:ExifImageHeight': height,
            'XMP:CroppedAreaImageWidthPixels': width,
            'XMP:CroppedAreaImageHeightPixels': height,
            'XMP:FullPanoWidthPixels': width,
            'XMP:FullPanoHeightPixels': height,
            'Composite:ImageSize': str(width) + ' ' + str(height),
            'Composite:Megapixels': round(width * height / 1000000, 1)
        }

        for k, v in m.items():
            self._meta[k] = v

    def set_date(self, d):
        self._meta['EXIF:DateTimeOriginal'] = d

    def set_gps(self, gps):
        self._meta['Composite:GPSLatitude'] = gps['latitude']
        self._meta['Composite:GPSLongitude'] = gps['longitude']
        self._meta['Composite:GPSPosition'] = str(gps['latitude']) + ' ' + str(gps['longitude'])
        self._meta['EXIF:GPSLatitudeRef'] = 'N' if gps['latitude'] > 0 else 'S'
        self._meta['EXIF:GPSLatitude'] = abs(gps['latitude'])
        self._meta['EXIF:GPSLongitudeRef'] = 'W' if gps['longitude'] < 0 else 'E'
        self._meta['EXIF:GPSLongitude'] = abs(gps['longitude'])

    def write_stereo(self, file_left, file_right):
        self._set_dimensions(self._left.shape[1], self._left.shape[0])
        cv.imwrite(file_left, self._left, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(file_right, self._right, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        self._exif.set_tags([file_left, file_right], tags=self._meta, params=['-P', '-overwrite_original'])

    def write_over_under(self, file_name):
        self._set_dimensions(self._left.shape[1], 2 * self._left.shape[0])
        combined = np.concatenate([self._left, self._right])
        cv.imwrite(file_name, combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        self._exif.set_tags(file_name, tags=self._meta, params=['-P', '-overwrite_original'])

    def write_anaglyph(self, file_name, fov, phi, theta, xres, yres):
        # https://3dtv.at//knowhow/anaglyphcomparison_en.aspx
        # Using the optimized anaglyph equation from the above link.

        left, _ = Equirectangular(self._left).GetPerspective(fov, theta+5, phi, yres, xres)
        right, _ = Equirectangular(self._right).GetPerspective(fov, theta-5, phi, yres, xres)

        shape_1d = (left.shape[0] * left.shape[1], 3)
        lm = np.array([[0, 0, 0], [0, 0, 0], [0.3, 0.7, 0]])
        rm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

        combined = np.matmul(lm, left.reshape(shape_1d).transpose()) \
                     .transpose().reshape(left.shape)
        combined += np.matmul(rm, right.reshape(shape_1d).transpose()) \
                      .transpose().reshape(right.shape)

        cv.imwrite(file_name, combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        self._set_dimensions(xres, yres)
        self._meta['XMP:InitialHorizontalFOVDegrees'] = fov
        self._exif.set_tags(file_name, tags=self._meta, params=['-P', '-overwrite_original'])

    def write_over_under_cropped(self, file_name, fov, phi, theta, xres, yres):
        left, _ = Equirectangular(self._left).GetPerspective(fov, theta + 2.5, phi, yres, xres)
        right, _ = Equirectangular(self._right).GetPerspective(fov, theta - 2.5, phi, yres, xres)

        combined = np.concatenate([left, right])
        cv.imwrite(file_name, combined, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        self._set_dimensions(xres, yres)
        self._meta['XMP:InitialHorizontalFOVDegrees'] = fov
        self._exif.set_tags(file_name, tags=self._meta, params=['-P', '-overwrite_original'])

    def write_stereo_cropped(self, file_left, file_right, fov, phi, theta, xres, yres):
        left, _ = Equirectangular(self._left).GetPerspective(fov, theta + 2.5, phi, yres, xres)
        right, _ = Equirectangular(self._right).GetPerspective(fov, theta - 2.5, phi, yres, xres)

        self._set_dimensions(xres, yres)
        cv.imwrite(file_left, left, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite(file_right, right, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        self._exif.set_tags([file_left, file_right], tags=self._meta, params=['-P', '-overwrite_original'])

    def write_cardboard_cropped(self, file_name, phi, theta, vfov, hfov):
        h = cb = self._left.shape[0]
        w = cr = self._left.shape[1]
        ct = cl = 0

        if vfov != 180:
            ct = int((-1 * phi - vfov / 2 + 90) / 180 * h)
            cb = int((-1 * phi + vfov / 2 + 90) / 180 * h)

        if hfov != 360:
            cl = int((theta - hfov / 2 + 180) / 360 * w) % w
            cr = int((theta + hfov / 2 + 180) / 360 * w) % w

        self._set_dimensions(abs(cr-cl), cb-ct)
        if cl < cr:
            left = self._left[ct:cb, cl:cr]
            right = self._right[ct:cb, cl:cr]
        else:
            left = np.concatenate([self._left[ct:cb, cl:], self._left[ct:cb, :cr]], axis=1)
            right = np.concatenate([self._right[ct:cb, cl:], self._right[ct:cb, :cr]], axis=1)

        cv.imwrite(file_name, left, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        cv.imwrite('tmp_right.JPG', right, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        self._exif.set_tags([file_name, 'tmp_right.JPG'], tags=self._meta, params=['-P', '-overwrite_original'])

        right_encode = None
        with open('tmp_right.JPG', mode='rb') as file:
            right_encode = base64.b64encode(file.read()).decode('ascii')

        gpano = self._gpano \
                    .replace('__HEADING__', str(theta)) \
                    .replace('__WIDTH__', str(self._left.shape[1])) \
                    .replace('__CROP_WIDTH__', str(abs(cr-cl))) \
                    .replace('__CROP_LEFT__', str(cl)) \
                    .replace('__HEIGHT__', str(self._left.shape[0])) \
                    .replace('__CROP_HEIGHT__', str(cb-ct)) \
                    .replace('__CROP_TOP__', str(ct)) \
                    .replace('__BASE64DATA__', str(right_encode))

        with open('tmp_gpano.xmp', 'w') as f:
            f.write(gpano)

        try:
            self._exif.execute('-overwrite_original', '-tagsfromfile', 'tmp_gpano.xmp', '"-xmp:all<all"', file_name)
        except Exception as e:
            pass

        os.remove('tmp_right.JPG')
        os.remove('tmp_gpano.xmp')

    def write_cardboard(self, file_name):
        self.write_cardboard_cropped(file_name, 0, 0, 165, 360)
