# vuze-3d-stitcher
A python based script for combining 8 separate photos taken by the Vuze 4K 3D 360 camera into a stereo 360 image. The goal is to provide a unix based alternative to the Vuze VR Studio which is only available on Windows and MAC. In the process, it may be possible to improve upon the overall stitch and output image quality.

# Website
[E-Regular Games](https://www.e-regular-games.com)

# Repository Structure
## Notes
This directory contains documentation on useful processes and tests. It also documents attempts which failed, to avoid repeating them in the future. This is an on-going development log.

[Development Log](./notes/README.md)

## Src
The working source code. The current implementation is in python and requires several added libraries such as Numpy, OpenCV and MatPlotLib.

[Main Function](./src/vuze_merge.py)

## Test
A collection of test images, configuration files, and result images.

[Test Images](./test/README.md)

# References

[Jumping Jack Flash - Foto 3d a 180° o 360° (VR180 o VR360)](https://jumpjack.wordpress.com/2021/07/03/foto-3d-a-180-o-360-vr180-o-vr360/) - A thorough explanation of how to create an image compatible with Google Photos that will render in 360° Stereoscopic.
