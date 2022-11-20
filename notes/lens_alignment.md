## Lens Alignment

*Date: November 17, 2022*

Script: [vuze_merge.py](../src/vuze_merge.py)

Usage:
```
../src/vuze_merge.py -d "calib" -v -w coeffs_v3.dat
```

### Objective

Ensure the images captured by each image sensor are properly centered, rotated, and de-skewed. The lens and the image sensor may not be in the same plane or perfectly aligned causing issues during more complex calculations.

### Method

When attempting to determine the depth of objects within left and right eye images, the results were wrong. The reason behind the incorrect calculations was determined to be poor alignment of the images from each lens. The biggest source of error was using the incorrect lens center when adjusting for the offset between the center of the lens and the center of the image sensor.

| Correcting Lens and Sensor Position, Alignment, and Skew |
| :----: |
| <img src="lens_alignment_calibration.png" alt="Lens and sensor mis-aligned and then aligned." width="500px" /> |

The lens is assumed to be a perfect circle. When the circle is viewed from the sensor it may appear shifted and skewed. The lens may appear elliptical to the sensor.

First the boundary between the lens and the surrounding area is established. An outdoor image of woods was used to provided a large number of edges. Assuming the scene captured by the lens is all edges the surrounding area outside the lens should have no edges.

To determine whether a given pixel in the image was viewing the lens or not the following steps were used.

1. Convert the image to gray scale.
1. Median Blur with 5 pixels.
1. Adaptive gaussian threshold.
1. Gaussian blur over an area of 31x31 pixels.
1. threshold of 180/255.

The result is an image similar to the images below.

| Pixels within each Lens |
| :----: |
| <img src="lens_alignment_in_lens_3.png" alt="Lens is shifted up." width="200px" /> <img src="lens_alignment_in_lens_4.png" alt="Lens is shifted down." width="200px" /> <img src="lens_alignment_in_lens_5.png" alt="Lens is shifted right." width="200px" /> |

Within each column the top-most and bottom most white pixels were determined. These values are the edge of the lens. The lens is assumed to be elliptical when viewed from the sensor and using the points along the ellipse the constants for the ellipse can be determined. The python code from [SciPython - Christian](https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/) was used to determine the least squares fit of the ellipse and the parameters $x_0, y_0, a_p, b_p, e, and \phi$. This method uses the distance from the conic as represented by $F$.

$$F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f$

The coefficients $a, b, c, d, e, f$ are then converted to ellipse parameters $x_0, y_0, a, b, \rho$. In this notation, $a$ is the radius along the semi-major axis and $b$ is the radius along the semi-minor axis. The rotation of the semi-major axis about the center is $\rho$. Using HET_0014 as the image with a lot of edges, the following ellipse parameters are computed. All coordinate values are in pixels from the top left corner.

| Lens | $x_0$ | $y_0$ | $a$ | $b$ | $\rho$ |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | 561.22 | 767.73 | 763.56 | 722.83 |  1.44 |

These coefficients are inline with the observations made from the images above. The semi-major axis of the ellipse appears along the verticle axis. This means the rotation $\rho$ should be approximately $\pi/2$. The center of each lens appears to be near the center of the sensor and the radius of the lens appears to be approximately half the height of the sensor, 800px.

Reversing the misalignment of the sensor and the lens can be performed using the following manipulations.

$$\vec{v_a} = \begin{pmatrix} a\cos\rho \\\\ a\sin\rho \end{pmatrix}$$

$$\vec{v_b} = \begin{pmatrix} b\sin\rho \\\\ b\cos\rho \end{pmatrix}$$

These new unit vectors are used during the fisheye to equirectangular conversion. For every pixel within the desired equirectangular output image, the coordinate location is converted to polar, then to cartesian (radius=1), then converted to the pixel location on a perfectly aligned fisheye lens with radius=1 and centered at the origin. The coordiante within the perfect fisheye is translated to the unit vector space defined by the axises of the calibration ellipse and then adjusted to map into the sensor coordinates.

Given the point $\vec{f}$ in the perfect fisheye lens the following is used to map to the actual pixel value within the sensor, $p_s$. The vectors $u_x$ and $u_y$ are the unit vectors in the x-axis and y-axis respectively.

$$\vec{f_s} = \begin{pmatrix} \vec{f} \cdot \vec{v_a} \\\\ \frac{b}{a}(\vec{f} \cdot \vec{v_b}) \end{pmatrix}$$

$$\vec{p_s} = \begin{pmatrix} \vec{f_s} \cdot \vec{u_x} \\\\ \vec{f_s} \cdot \vec{u_y} \end{pmatrix}$$
