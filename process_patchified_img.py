"""Python file processing the patchified image."""
import sys

import cv2
import numpy as np
from patchify import patchify

from patching_utils import *

input_image_path = sys.argv[1]

# Load the grascale image
grayscale_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(input_image_path)
PATCH_SIZE = 5

# Patchify the image
patches = patchify(grayscale_image, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)


# Calculate the brightness.
brightness = get_patches_brightness(patches=patches)
NUM_TOP_PATCHES = 4

top_brightness_indices = get_top_bright_indices(brightness, NUM_TOP_PATCHES)
top_bright_centers = get_top_bright_centers(top_brightness_indices, PATCH_SIZE)
top_bright_centers = sorted(top_bright_centers, key=lambda x: (x[0], x[1]))
vertices = np.array(top_bright_centers)
quadrilateral_area = calculate_quadrilateral_area(vertices)
RED = (0, 0, 255)
cv2.line(img, top_bright_centers[0], top_bright_centers[1], RED, 2)
cv2.line(img, top_bright_centers[0], top_bright_centers[2], RED, 2)
cv2.line(img, top_bright_centers[1], top_bright_centers[3], RED, 2)
cv2.line(img, top_bright_centers[2], top_bright_centers[3], RED, 2)

FILENAME = "updated_image.png"

# Saving the image
cv2.imwrite(FILENAME, img)
