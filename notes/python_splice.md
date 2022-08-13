## Python Splice

Script: [splice_360.py](./splice_360.py)

Usage (from ./test):
```
../notes/splice_360.sh -d -c config_bash_splice_360.dat
../notes/splice_360.py -c config_python_splice_360.dat
```

This script allows for scaling, rotating, and shifting each lens individually. It attempts to correct the issue of missing pieces which was found in the Bash script. The python script requires the equirectangular 180 degree images created by the previous Bash script. OpenCV and Numpy were used for loading images and manipulating image data in an efficient manner. To ensure efficiency and reduce the number of intermediate images, the coordinates are computed without resolving to pixel colors until the final step.

The equirectangular image coordinates for each pixel in the final image are created. These coordinates are translated to polar, and then filtered to the respective lens the polar coordinate should reference. The polar coordinate is then converted to an equirectangular coordinate relative to that lens. The final step is to determine the color of that coordinate using a linear interpolation of the 4 surrounding pixels.

The results were reasonable, but required complete manual configuration of the transformation parameters. This is infeasible as there are 8 lenses and each lens could have different parameters. Adjust parameters for one lens could impact the parameters for other lenses. It is a problem which will need a more automated solution.

The example configuration file [config_python_splice_360.dat](../test/config_python_splice_360.dat) and output is provided in the [/test](../test) directory.

<img src="../test/HET_0014_python.JPG" alt="Python splice result." width="540px" />

### References
[OpenCV Getting Started](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html)
