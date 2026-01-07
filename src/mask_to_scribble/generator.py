"""Deterministic multi-scribble generator from binary masks.

Core idea (per scribble stroke k):
1) Select N_k points inside mask with mixture of two priors:
   - center-prior (near centroid)
   - boundary-prior (near boundary)
2) Build MST edges among points (deterministic).
3) For each edge, route a path inside mask using Dijkstra with jittered cost.
4) Combine K strokes with OR.
5) Thicken strokes. Optional soft labels.

Determinism:
- base_seed = CRC32(mask bytes)
- per-stroke rng: default_rng(base_seed + k)

Dependencies:
- numpy
- opencv-python
- scikit-image (route_through_array)
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .geometry import build_mst_edges, compute_centroid_yx
from .param_utils import pick_float, pick_int
from .rendering import compute_stroke_radius, dilate_with_radius, generate_soft_labels
from .routing import make_jitter_field, route_path_inside_mask
from .sampling import (
    choose_num_points,
    compute_min_distance,
    fallback_points,
    sample_points_with_mixture,
)
from .types import FloatSpec, IntSpec, ScribbleConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -----------------------------
# Generator
# -----------------------------
class ScribbleGenerator:
    """Generator for deterministic multi-stroke scribbles from binary masks."""

    def __init__(self, config: ScribbleConfig | None = None) -> None:
        """Initialize generator with configuration.

        Args:
            config: Scribble generation configuration

        """
        self.cfg = config or ScribbleConfig()

    def from_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8] | NDArray[np.float32]:
        """Generate scribble from binary mask.

        Args:
            mask: Binary mask (uint8, 0 or 255)

        Returns:
            Scribble image (uint8 if binary, float32 if soft labels)

        """
        self._validate_mask(mask)

        if mask.max() == 0:
            return np.zeros_like(mask, dtype=np.float32 if self.cfg.use_soft_labels else np.uint8)

        m = (mask > 0).astype(np.uint8)  # 0/1
        area = int(m.sum())
        if area <= 0:
            return np.zeros_like(mask, dtype=np.float32 if self.cfg.use_soft_labels else np.uint8)

        base_seed = self._stable_seed(mask)

        # precompute geometry that does NOT depend on per-stroke params
        dt_inside = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 3).astype(
            np.float32,
        )
        centroid_yx = compute_centroid_yx(m)

        final = np.zeros_like(mask, dtype=np.uint8)

        # generate K scribbles
        for k in range(int(max(1, self.cfg.num_scribbles))):
            rng = np.random.default_rng(base_seed + k)

            # per-stroke params
            n_k = choose_num_points(area, rng=rng, k=k, config=self.cfg)
            mix_center_k = float(np.clip(pick_float(self.cfg.mix_center, rng, k, 0.5), 0.0, 1.0))
            center_sigma_scale_k = max(
                0.01,
                pick_float(self.cfg.center_sigma_scale, rng, k, 0.25),
            )
            boundary_sigma_k = max(0.5, pick_float(self.cfg.boundary_sigma, rng, k, 6.0))
            min_dist_scale_k = max(0.0, pick_float(self.cfg.min_dist_scale, rng, k, 0.10))
            jitter_strength_k = max(0.0, pick_float(self.cfg.jitter_strength, rng, k, 0.25))
            jitter_ksize_k = int(pick_int(self.cfg.jitter_smooth_ksize, rng, k, 9))
            center_bias_k = max(0.0, pick_float(self.cfg.center_bias, rng, k, 0.10))

            jitter_field = make_jitter_field(m.shape, rng=rng, ksize=jitter_ksize_k)
            min_dist = compute_min_distance(
                area,
                min_dist_scale=min_dist_scale_k,
                min_dist_min=self.cfg.min_dist_min,
                min_dist_max=self.cfg.min_dist_max,
            )

            # sample points
            pts_xy = sample_points_with_mixture(
                m=m,
                dt_inside=dt_inside,
                centroid_yx=centroid_yx,
                area=area,
                n=n_k,
                rng=rng,
                mix_center=mix_center_k,
                center_sigma_scale=center_sigma_scale_k,
                boundary_sigma=boundary_sigma_k,
                min_dist=min_dist,
                max_sampling_trials=self.cfg.max_sampling_trials,
            )

            if len(pts_xy) < 2:  # noqa: PLR2004
                pts_xy = fallback_points(m, dt_inside, centroid_yx)

            edges = build_mst_edges(pts_xy)

            stroke = np.zeros_like(mask, dtype=np.uint8)
            for i, j in edges:
                a = pts_xy[i]
                b = pts_xy[j]
                path = route_path_inside_mask(
                    m=m,
                    dt_inside=dt_inside,
                    a_xy=a,
                    b_xy=b,
                    jitter=jitter_field,
                    jitter_strength=jitter_strength_k,
                    center_bias=center_bias_k,
                )
                if path is not None and len(path) > 0:
                    stroke[path[:, 0], path[:, 1]] = 255  # path is (y,x)

            # keep within defect
            stroke = cv2.bitwise_and(stroke, (m * 255).astype(np.uint8))

            # merge
            final = cv2.bitwise_or(final, stroke)

            # coverage cap (optional per-stroke)
            if self.cfg.coverage_check_each_stroke and self._exceeds_coverage(final, m):
                break

        # thickness
        r = compute_stroke_radius(
            area,
            min_radius=self.cfg.min_radius,
            radius_scale=self.cfg.radius_scale,
            max_radius=self.cfg.max_radius,
        )
        final = dilate_with_radius(final, r)

        # keep within defect
        final = cv2.bitwise_and(final, (m * 255).astype(np.uint8))

        # final coverage cap (safety)
        if self._exceeds_coverage(final, m):
            # If exceeded after dilation, lightly erode once to reduce coverage.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final = cv2.erode(final, kernel, iterations=1)

        if self.cfg.use_soft_labels:
            return generate_soft_labels(final, self.cfg.soft_label_sigma)

        return final

    def _stable_seed(self, mask: NDArray[np.uint8]) -> int:
        """Generate deterministic seed from mask content.

        Args:
            mask: Binary mask

        Returns:
            CRC32 checksum as seed

        """
        data = memoryview(mask).tobytes()
        return int(zlib.crc32(data) & 0xFFFFFFFF)

    def _exceeds_coverage(self, scribble_255: NDArray[np.uint8], m01: NDArray[np.uint8]) -> bool:
        """Check if scribble exceeds coverage cap.

        Args:
            scribble_255: Binary scribble (0/255)
            m01: Binary mask (0/1)

        Returns:
            True if coverage exceeds cap

        """
        cap = float(self.cfg.coverage_cap)
        if cap <= 0:
            return False
        mask_area = int(m01.sum())
        if mask_area <= 0:
            return False
        scribble_area = int(np.sum((scribble_255 > 0) & (m01 > 0)))
        cov = scribble_area / float(mask_area)
        return cov > cap

    def _validate_mask(self, mask: NDArray[np.uint8]) -> None:
        """Validate mask format.

        Args:
            mask: Binary mask to validate

        Raises:
            TypeError: If mask is not uint8
            ValueError: If mask is not 2D

        """
        if mask.dtype != np.uint8:
            msg = f"mask must be uint8, got {mask.dtype}"
            raise TypeError(msg)
        if mask.ndim != 2:  # noqa: PLR2004
            msg = f"mask must be 2D (H,W), got {mask.shape}"
            raise ValueError(msg)


# -----------------------------
# Functional API
# -----------------------------
def generate_scribble(  # noqa: PLR0913
    mask: NDArray[np.uint8],
    *,
    num_scribbles: int = 3,
    coverage_cap: float = 0.15,
    coverage_check_each_stroke: bool = True,
    fixed_num_points: int | None = None,
    min_points: int = 3,
    max_points: int = 7,
    mix_center: FloatSpec = 0.5,
    center_sigma_scale: FloatSpec = 0.25,
    boundary_sigma: FloatSpec = 6.0,
    min_dist_scale: FloatSpec = 0.10,
    jitter_strength: FloatSpec = 0.25,
    jitter_smooth_ksize: IntSpec = 9,
    center_bias: FloatSpec = 0.10,
    min_radius: int = 2,
    radius_scale: float = 0.012,
    max_radius: int = 20,
    use_soft_labels: bool = False,
    soft_label_sigma: float = 2.0,
) -> NDArray[np.uint8] | NDArray[np.float32]:
    cfg = ScribbleConfig(
        num_scribbles=num_scribbles,
        coverage_cap=coverage_cap,
        coverage_check_each_stroke=coverage_check_each_stroke,
        fixed_num_points=fixed_num_points,
        min_points=min_points,
        max_points=max_points,
        mix_center=mix_center,
        center_sigma_scale=center_sigma_scale,
        boundary_sigma=boundary_sigma,
        min_dist_scale=min_dist_scale,
        jitter_strength=jitter_strength,
        jitter_smooth_ksize=jitter_smooth_ksize,
        center_bias=center_bias,
        min_radius=min_radius,
        radius_scale=radius_scale,
        max_radius=max_radius,
        use_soft_labels=use_soft_labels,
        soft_label_sigma=soft_label_sigma,
    )
    return ScribbleGenerator(cfg).from_mask(mask)
