#!/usr/bin/python
import base64
import math
import numpy as np
import cv2 as cv
import os
from exiftool import ExifToolHelper
from exiftool import ExifTool

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
         <GPano:CroppedAreaImageHeightPixels>__HEIGHTEYE__</GPano:CroppedAreaImageHeightPixels>
         <GPano:InitialViewHeadingDegrees>180</GPano:InitialViewHeadingDegrees>
         <GPano:InitialHorizontalFOVDegrees>150</GPano:InitialHorizontalFOVDegrees>
         <GPano:CroppedAreaImageWidthPixels>__WIDTH__</GPano:CroppedAreaImageWidthPixels>
         <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
         <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
         <GPano:FullPanoHeightPixels>__HEIGHTEYE__</GPano:FullPanoHeightPixels>
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
            'XMP:InitialViewPitchDegrees': 1,
            'XMP:InitialViewRollDegrees': -1,
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

    def write_cardboard(self, file_name):
        self.write_stereo(file_name, 'tmp_right.JPG')

        right_encode = None
        with open('tmp_right.JPG', mode='rb') as file:
            right_encode = base64.b64encode(file.read()).decode('ascii')

        gpano = self._gpano \
                    .replace('__WIDTH__', str(self._left.shape[1])) \
                    .replace('__HEIGHTEYE__', str(self._left.shape[0])) \
                    .replace('__BASE64DATA__', str(right_encode))

        with open('tmp_gpano.xmp', 'w') as f:
            f.write(gpano)

        try:
            self._exif.execute('-overwrite_original', '-tagsfromfile', 'tmp_gpano.xmp', '"-xmp:all<all"', file_name)
        except Exception as e:
            pass

        os.remove('tmp_right.JPG')
        os.remove('tmp_gpano.xmp')
