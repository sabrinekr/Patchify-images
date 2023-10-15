"""Python file containing the utilities for the patching scrpit."""

import numpy as np


def calculate_quadrilateral_area(vertices):
    """
    Calculate the area of a quadrilateral defined by its vertices using 
        the shoelace formula.

    Args:
        vertices (list of tuples): List of (x, y) coordinates for the four vertices
            of the quadrilateral.

    Returns:
        float: The area of the quadrilateral in square pixels.
    """
    vertices_num = len(vertices)
    sum1 = 0
    sum2 = 0

    for i in range(0, vertices_num - 1):
        sum1 = sum1 + vertices[i][0] * vertices[i + 1][1]
        sum2 = sum2 + vertices[i][1] * vertices[i + 1][0]

    # Add xn.y1
    sum1 = sum1 + vertices[vertices_num - 1][0] * vertices[0][1]
    # Add x1.yn
    sum2 = sum2 + vertices[0][0] * vertices[vertices_num - 1][1]

    area = abs(sum1 - sum2) / 2
    return area


def get_patches_brightness(patches: np.ndarray) -> np.ndarray:
    """Calculate the brightness of the patches.

    Args:
        patches: array of the patches of shape
            (image_width/ patch_shape, image_heigh/ patch_shape, 
                patch_shape, patch_shape).

    Returns:
        The brightness of the patches of
            shape (image_width/ patch_shape, image_heigh/ patch_shape).

    """
    return np.mean(patches, axis=(2, 3))


def get_top_bright_indices(
    brightness: np.ndarray, num_top_patches: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of the top "num_top_patches" bright patches.

    Args:
        brightness: the brightness of the patches of
            shape (image_width/ patch_shape, image_heigh/ patch_shape).
        num_top_patches: the number of top bright patches to return.

    Returns:
        top_bright_indices: tuple of the top bright indices.

    """
    top_bright_indices = np.argpartition(-brightness, num_top_patches, axis=None)[
        :num_top_patches
    ]
    top_bright_indices = np.unravel_index(top_bright_indices, brightness.shape)
    return top_bright_indices


def get_top_bright_centers(top_bright_indices, patch_size):
    """Get the centers of the top bright patches in the original image.

    Args:
        top_bright_indices: tuple of the top bright indices.
        patch_size: the size of the patch.

    Returns:
        top_bright_centers: list of the centers of the top bright patches.

    """
    top_brightness_patches_indices = [
        (i * patch_size, j * patch_size)
        for i, j in zip(top_bright_indices[0], top_bright_indices[1])
    ]
    top_bright_centers = [
        (patch[0] + 2, patch[1] + 2) for patch in top_brightness_patches_indices
    ]
    return top_bright_centers
