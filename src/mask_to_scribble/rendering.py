"""Stroke rendering and thickness utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_stroke_radius(
    area: int,
    *,
    min_radius: int,
    radius_scale: float,
    max_radius: int,
) -> int:
    """Compute stroke radius based on mask area.

    Args:
        area: Mask area in pixels
        min_radius: Minimum radius
        radius_scale: Scale factor relative to sqrt(area)
        max_radius: Maximum radius

    Returns:
        Stroke radius in pixels

    """
    r = round(np.sqrt(float(area)) * float(radius_scale))
    r = max(r, int(min_radius))
    return min(r, int(max_radius))


def dilate_with_radius(img_255: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """Dilate binary image with circular kernel.

    Args:
        img_255: Binary image (0/255)
        radius: Dilation radius

    Returns:
        Dilated image

    """
    if radius <= 0:
        return img_255
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(img_255, kernel, iterations=1)


def generate_soft_labels(scribble_255: NDArray[np.uint8], sigma: float) -> NDArray[np.float32]:
    """Generate soft labels from binary scribble using Gaussian falloff.

    Args:
        scribble_255: Binary scribble (0/255)
        sigma: Gaussian sigma for falloff

    Returns:
        Soft label map [0, 1]

    """
    dist = cv2.distanceTransform(255 - scribble_255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(
        np.float32,
    )
    return np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)
