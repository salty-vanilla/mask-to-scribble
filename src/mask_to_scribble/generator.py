"""Core scribble generation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.morphology import skeletonize

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ScribbleConfig:
    """Configuration for scribble generation.

    Attributes:
        contour_ratio: Fraction of contour to use (0.2-0.4 recommended)
        min_thickness: Minimum stroke thickness in pixels
        thickness_scale: Scale factor for thickness based on defect area
        use_soft_labels: Whether to generate soft labels with distance decay
        soft_label_sigma: Sigma for Gaussian distance decay in soft labels

    """

    contour_ratio: float = 0.3
    min_thickness: int = 2
    thickness_scale: float = 0.01
    use_soft_labels: bool = False
    soft_label_sigma: float = 2.0


class ScribbleGenerator:
    """Generates deterministic human-like scribble annotations from binary masks.

    The generator creates scribbles with:
    - One skeleton stroke (main branch)
    - One contour stroke (partial boundary segment)
    - Optional connecting stroke between skeleton and contour

    Scribbles are reproducible: same mask always produces same scribble.
    """

    def __init__(self, config: ScribbleConfig | None = None) -> None:
        """Initialize scribble generator.

        Args:
            config: Configuration for scribble generation. Uses defaults if None.

        """
        self.config = config or ScribbleConfig()

    def from_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8 | np.float32]:
        """Generate scribble annotation from binary mask.

        Args:
            mask: Binary mask (H, W) with defect region as 255, background as 0

        Returns:
            Scribble annotation with same shape as mask.
            - If use_soft_labels=False: uint8 array with strokes as 255
            - If use_soft_labels=True: float32 array with values in [0, 1]

        """
        if mask.dtype != np.uint8:
            msg = f"Mask must be uint8, got {mask.dtype}"
            raise TypeError(msg)

        if mask.max() == 0:
            # Empty mask returns empty scribble
            if self.config.use_soft_labels:
                return np.zeros_like(mask, dtype=np.float32)
            return np.zeros_like(mask, dtype=np.uint8)

        # Extract skeleton stroke
        skeleton_stroke = self._extract_skeleton_stroke(mask)

        # Extract contour stroke
        contour_stroke = self._extract_contour_stroke(mask)

        # Combine strokes
        scribble = cv2.bitwise_or(skeleton_stroke, contour_stroke)

        # Calculate thickness based on defect area
        thickness = self._calculate_thickness(mask)

        # Apply thickness
        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            scribble = cv2.dilate(scribble, kernel, iterations=1)

        # Apply soft labels if requested
        if self.config.use_soft_labels:
            return self._apply_soft_labels(scribble)

        return scribble

    def _extract_skeleton_stroke(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract main skeleton branch from mask.

        Args:
            mask: Binary mask

        Returns:
            Binary image with skeleton stroke

        """
        # Skeletonize the mask (convert to binary first)
        binary_mask = mask > 0
        skeleton = skeletonize(binary_mask)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        # Find connected components in skeleton
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            skeleton_uint8, connectivity=8,
        )

        if num_labels <= 1:
            # No skeleton found
            return np.zeros_like(mask)

        # Find largest connected component (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1

        # Keep only the largest branch
        skeleton_stroke = np.zeros_like(mask)
        skeleton_stroke[labels == largest_label] = 255

        return skeleton_stroke

    def _extract_contour_stroke(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract partial contour segment from mask.

        Uses only a fraction (contour_ratio) of the boundary to avoid
        full boundary tracing.

        Args:
            mask: Binary mask

        Returns:
            Binary image with contour stroke

        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.zeros_like(mask)

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate segment length
        contour_length = len(contour)
        segment_length = int(contour_length * self.config.contour_ratio)

        if segment_length < 2:
            return np.zeros_like(mask)

        # Take a segment starting from a deterministic position
        # Use the topmost point for reproducibility
        contour_points = contour.reshape(-1, 2)
        start_idx = np.argmin(contour_points[:, 1])  # Topmost point (min y)

        # Extract segment
        indices = np.arange(start_idx, start_idx + segment_length) % contour_length
        segment = contour[indices]

        # Draw segment on blank image
        contour_stroke = np.zeros_like(mask)
        cv2.drawContours(contour_stroke, [segment], -1, 255, 1)

        return contour_stroke

    def _calculate_thickness(self, mask: NDArray[np.uint8]) -> int:
        """Calculate stroke thickness based on defect area.

        Args:
            mask: Binary mask

        Returns:
            Thickness in pixels

        """
        defect_area = np.sum(mask > 0)
        thickness = int(np.sqrt(defect_area) * self.config.thickness_scale)
        return max(thickness, self.config.min_thickness)

    def _apply_soft_labels(self, scribble: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Apply soft labels with distance-based decay.

        Args:
            scribble: Binary scribble (0 or 255)

        Returns:
            Soft labels in range [0, 1] with Gaussian decay from strokes

        """
        # Compute distance transform from scribble strokes
        distance = cv2.distanceTransform(255 - scribble, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Apply Gaussian decay
        soft_labels = np.exp(-(distance**2) / (2 * self.config.soft_label_sigma**2))

        return soft_labels.astype(np.float32)


def generate_scribble(
    mask: NDArray[np.uint8],
    contour_ratio: float = 0.3,
    min_thickness: int = 2,
    thickness_scale: float = 0.01,
    use_soft_labels: bool = False,
    soft_label_sigma: float = 2.0,
) -> NDArray[np.uint8 | np.float32]:
    """Generate scribble annotation from binary mask (functional API).

    Args:
        mask: Binary mask (H, W) with defect region as 255, background as 0
        contour_ratio: Fraction of contour to use (0.2-0.4 recommended)
        min_thickness: Minimum stroke thickness in pixels
        thickness_scale: Scale factor for thickness based on defect area
        use_soft_labels: Whether to generate soft labels with distance decay
        soft_label_sigma: Sigma for Gaussian distance decay in soft labels

    Returns:
        Scribble annotation with same shape as mask.
        - If use_soft_labels=False: uint8 array with strokes as 255
        - If use_soft_labels=True: float32 array with values in [0, 1]

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[30:70, 30:70] = 255
        >>> scribble = generate_scribble(mask)
        >>> scribble.shape
        (100, 100)

    """
    config = ScribbleConfig(
        contour_ratio=contour_ratio,
        min_thickness=min_thickness,
        thickness_scale=thickness_scale,
        use_soft_labels=use_soft_labels,
        soft_label_sigma=soft_label_sigma,
    )
    generator = ScribbleGenerator(config)
    return generator.from_mask(mask)
