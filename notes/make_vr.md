## Make VR
Google Photos can render 360 images. It shows them in flat vr or as a single eye 2d moveable frame. It also appears to have the ability to render 360 3d images, but does not use the standard side-by-side or over under stereo format.

It turns out Google Photos uses image metadata to embed the right eye image in the left eye file. It uses XMP tags for dimensions, initial position, the image type, and the actual right eye image. While the standard appears straightforward there are some caveats to actually inserting the image. Allow exiftool to insert the XMP tags as it pleases leads to a file which Google Photos does not understand. Special care must be taken to preserve a format compatible with Google Photos.

To preserve the proper format a template xmp file is used. This template is modified with the current photo data and the exiftool inserts the template into the left eye image. The result is a single image which can be displayed both as a normal image and as 3d 360 VR.

### References

[Google VR Metadata Tags](https://developers.google.com/vr/reference/cardboard-camera-vr-photo-format)

[Jumping Jack Flash - Foto 3d a 180° o 360° (VR180 o VR360)](https://jumpjack.wordpress.com/2021/07/03/foto-3d-a-180-o-360-vr180-o-vr360/) - A thorough explanation of how to create an image compatible with Google Photos that will render in 360° Stereoscopic.

[JPEG ExifTool](https://exiftool.org/)