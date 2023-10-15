# Patchify-images

## Overview

This Python script reads a grayscale image from a file, identifies the four non-overlapping 5x5 patches with the highest average brightness, calculates the area of the quadrilateral formed by their centers, draws the quadrilateral in red on the image, and saves the result in PNG format. It uses the OpenCV library for image handling.

## Features

- Reads a grayscale image from a file.
- Identifies the four non-overlapping 5x5 patches with the highest average brightness.
- Calculates the area of the quadrilateral formed by the centers of these patches.
- Draws the quadrilateral in red on the image.
- Saves the resulting image in PNG format.

## Usage
```shell
git clone https://github.com/sabrinekr/Patchify-images.git
```

1. Place your grayscale image in the repository folder or specify its path in the Python script.

2. Run the Python script:
```shell
python image_processing.py path-to-the-input-image
```
3. The script will process the image, draw the quadrilateral, and save the result as "updated_image.png".

## Testing
The script includes test cases to ensure correct functionality. You can run the tests using the following command:
```shell
pytest test_patching_utils.py
```
