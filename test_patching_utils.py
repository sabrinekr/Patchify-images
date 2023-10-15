"""Test image patching utils."""
import numpy as np
from patchify import patchify
import pytest
from patching_utils import *


@pytest.mark.parametrize(
    ("patches", "expected_brightness"),
    [(np.array([
    [
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    ]
]), [[3.0]]),
    (np.array([[[[1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5]],

        [[1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5]]],


       [[[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]]]), [[3.0, 3.0], [1.0, 0.0]])]
)
def test_get_patches_brightness(
    patches: np.ndarray, expected_brightness
) -> None:
    """Check get the patches brightness.

    Args:
        patches: array of the patches of shape
            (image_width/ patch_shape, image_heigh/ patch_shape,
                patch_shape, patch_shape).
        expected: expected brightness.
    """
    brightness = get_patches_brightness(patches)
    assert brightness.shape == patches.shape[:2]
    assert (brightness == expected_brightness).all()

@pytest.mark.parametrize(
    ("brightness", "num_top_patches", "expected_top_bright_indices"),
    [(np.array([[3.0, 3.0], [1.0, 0.0]]), 2, (np.array([0, 0]), np.array([0, 1])))])
def test_get_top_bright_indices(
    brightness: np.ndarray, num_top_patches, expected_top_bright_indices
) -> None:
    """Check the functionality of the get top bright indices.

    Args:
        brightness: the brightness of the patches of
            shape (image_width/ patch_shape, image_heigh/ patch_shape).
        num_top_patches: the number of top bright patches to return.
        expected_top_bright_indices: expected top bright indices.
    """
    top_bright_indices = get_top_bright_indices(brightness, num_top_patches)
    assert (top_bright_indices[0] == expected_top_bright_indices[0]).all()
    assert (top_bright_indices[1] == expected_top_bright_indices[1]).all()

@pytest.mark.parametrize(
    ("image", "patch_size", "num_top_patches", "expected_top_bright_centers"),
    [
    (np.array([
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8],
    [9, 10, 20, 12, 13, 4, 5, 16, 7, 8],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8],
]), 5, 2, (20, 16)),
    (np.array([
    [1, 2, 3, 4, 5, 200, 200, 200, 200, 200, 1, 8, 7],
    [1, 2, 3, 4, 5, 200, 200, 200, 200, 200, 1, 8, 7],
    [1, 2, 3, 4, 5, 200, 200, 200, 200, 200, 1, 8, 7],
    [1, 2, 3, 4, 5, 200, 200, 200, 200, 200, 1, 8, 7],
    [1, 2, 3, 4, 5, 200, 200, 200, 200, 200, 1, 8, 7],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8, 1, 12, 13],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8, 1, 12, 13],
    [9, 10, 20, 12, 13, 4, 5, 16, 7, 8, 1, 12, 13],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8, 1, 12, 13],
    [9, 10, 11, 12, 13, 4, 5, 6, 7, 8, 1, 12, 13],
]), 5, 2, (200, 20))
]
)
def test_get_image_brightest_centers(
    image: np.ndarray, patch_size, num_top_patches, expected_top_bright_centers
) -> None:
    """Check the functionality of getting the images bright centers.

    Args:
        image: array presenting the image.
        patch_size: the size of the patch.
        num_top_patches: the top bright patches to get in the output.
        expected_top_bright_centers: the expected top bright centers to get in the output.
    """
    patches = patchify(image, (patch_size,patch_size), step= patch_size) 
    brightness = get_patches_brightness(patches=patches)
    top_brightness_indices = get_top_bright_indices(brightness, num_top_patches)
    top_bright_centers = get_top_bright_centers(top_brightness_indices, patch_size)
    assert image[top_bright_centers[0][0], top_bright_centers[0][1]] == expected_top_bright_centers[0]
    assert image[top_bright_centers[1][0], top_bright_centers[1][1]] == expected_top_bright_centers[1]