## Developer Log
*Entries at the top are more recent.*

### Exposure Fusion
Expanding the feature set of the Vuze VR Studio to include exposure bracketing and HDR image creation would be helpful. The exposure fusion script attempts to take multiple images from a single lens and fuse the exposures into a new single image for that lens. This new image can be used in Vuze VR Studio or provided to a yet to be created splice script.

The first attempt at exposure fusion relied on Luminance HDR. The application has a batch interface which allows for creating HDR images and saving them as HDR images. The application also allows for creating a LDR image from the HDR image. Due to the test images have over-exposed areas Luminance HDR and the LDR conversion alogirthms create images with grey blotches in the over-exposed area. To mitigate this, the bash script for exposure fusion added support for negating the image and then fusing exposures. A clipping feature was also added. Finally a reasonable result was obtained. This result was given to the python splice script to produce the output below.

<img src="../test/HET_1014_exposure_luminance.JPG" alt="Luminance HDR fused exposure before stitching." width="540px" />

Given the complexity of Luminance HDR stitching and its in ability to provide a reasonable image given real world input a simpler solution is needed. The `enfuse` application was incorporated into the script as an option. Enfuse does not require the HDR configuration or LDR conversion. It does not require the exposure value for each image either.

<img src="../test/HET_1014_exposure_enfuse.JPG" alt="Enfuse fused exposure before stitching." width="540px" />

Cleary the winner here is Enfuse. The simple configuration and color accurate result speak for themselves.

#### References
[Luminance HDR](https://luminancehdr.readthedocs.io/en/latest/)
[pfsclamp](https://resources.mpi-inf.mpg.de/pfstools/man1/pfsclamp.1.html)
[Enfuse](http://enblend.sourceforge.net/index.htm)
[Mertens-Kautz-Van Reeth](https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf)

### Python Splice

Script: [splice_360.py](./splice_360.py)

Usage (from ./test):
`../notes/splice_360.sh -d -c config_bash_splice_360.dat`
`../notes/splice_360.py -c config_python_splice_360.dat`

This script allows for scaling, rotating, and shifting each lens individually. It attempts to correct the issue of missing pieces which was found in the Bash script. The python script requires the equirectangular 180 degree images created by the previous Bash script. OpenCV and Numpy were used for loading images and manipulating image data in an efficient manner. To ensure efficiency and reduce the number of intermediate images, the coordinates are computed without resolving to pixel colors until the final step.

The equirectangular image coordinates for each pixel in the final image are created. These coordinates are translated to polar, and then filtered to the respective lens the polar coordinate should reference. The polar coordinate is then converted to an equirectangular coordinate relative to that lens. The final step is to determine the color of that coordinate using a linear interpolation of the 4 surrounding pixels.

The results were reasonable, but required complete manual configuration of the transformation parameters. This is infeasible as there are 8 lenses and each lens could have different parameters. Adjust parameters for one lens could impact the parameters for other lenses. It is a problem which will need a more automated solution.

The example configuration file [config_python_splice_360.dat](../test/config_python_splice_360.dat) and output is provided in the [/test](../test) directory.

<img src="../test/HET_0014_python.JPG" alt="Python splice result." width="540px" />

#### References
[OpenCV Getting Started](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html)

### Bash Splice

Script: [splice_360.sh](./splice_360.sh)

Usage (from ./test): `../notes/splice_360.sh -c config_bash_splice_360.dat`

The goal is to join 4 left eye images into a single 360 panorama, and the 4 right eye images into a second 360 panorama. The first step was determining the center of the fisheye and the radius for each lens. The images were then cropped and padded such that the center of the fisheye was the center of the iamge. Ffmpeg was used to convert from fisheye to cubemap with faces ordered consistently such that the front of the camera was always the same side of the cubemap. The cubemap for each image was split apart and re-assembled such that a single cubemap including pieces of the original 4 was created. This was done for each eye and then the cubemaps were converted to equirectangular images.

The result was very poorly stitched 360 panorama for each eye. The seam locations being at the same angle from center in each image causes 8 seams to appear when viewing in stereo mode. Parts of the image were missing as horizontal scaling causes each image to occupy more than 90 degrees of the view. Ideally, there should be 4 seams and no missing parts of the image.

The example configuration file [config_bash_splice_360.dat](../test/config_bash_splice_360.dat) and output is provided in the [/test](../test/) directory.

<img src="../test/HET_0014_bash.JPG" alt="Bash splice result." width="540px" />

#### References
[FFMPEG v360 Filter](https://ffmpeg.org/ffmpeg-filters.html#v360)
[ImageMagick Convert](https://imagemagick.org/script/convert.php)

### Make VR
Google Photos can render 360 images. It shows them in flat vr or as a single eye 2d moveable frame. It also appears to have the ability to render 360 3d images, but does not use the standard side-by-side or over under stereo format.

It turns out Google Photos uses image metadata to embed the right eye image in the left eye file. It uses XMP tags for dimensions, initial position, the image type, and the actual right eye image. While the standard appears straightforward there are some caveats to actually inserting the image. Allow exiftool to insert the XMP tags as it pleases leads to a file which Google Photos does not understand. Special care must be taken to preserve a format compatible with Google Photos.

To preserve the proper format a template xmp file is used. This template is modified with the current photo data and the exiftool inserts the template into the left eye image. The result is a single image which can be displayed both as a normal image and as 3d 360 VR.

#### References
[Google VR Metadata Tags](https://developers.google.com/vr/reference/cardboard-camera-vr-photo-format)
[Jumping Jack Flash - Foto 3d a 180° o 360° (VR180 o VR360)](https://jumpjack.wordpress.com/2021/07/03/foto-3d-a-180-o-360-vr180-o-vr360/) - A thorough explanation of how to create an image compatible with Google Photos that will render in 360° Stereoscopic.
[JPEG ExifTool](https://exiftool.org/)