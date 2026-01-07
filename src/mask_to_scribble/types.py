"""Type definitions and configuration for scribble generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RangeF:
    """Float range [lo, hi] for random sampling."""

    lo: float
    hi: float


@dataclass(frozen=True)
class RangeI:
    """Integer range [lo, hi] (inclusive) for random sampling."""

    lo: int
    hi: int


FloatSpec = float | Sequence[float] | RangeF | None
IntSpec = int | Sequence[int] | RangeI | None


@dataclass
class ScribbleConfig:
    """Configuration for scribble generation.

    Multi-scribble parameters:
        num_scribbles: Number of scribble strokes to generate
        coverage_cap: Maximum coverage ratio (scribble_pixels / mask_pixels)
        coverage_check_each_stroke: Check coverage after each stroke

    Point selection parameters:
        fixed_num_points: Fixed number of points (overrides area-based)
        min_points: Minimum number of points
        max_points: Maximum number of points
        mix_center: Mixture ratio (1.0=center only, 0.0=boundary only)
        center_sigma_scale: Center prior spread scale
        boundary_sigma: Boundary prior spread
        min_dist_scale: Minimum distance scale between points
        min_dist_min: Minimum absolute distance between points
        min_dist_max: Maximum absolute distance between points
        max_sampling_trials: Maximum sampling attempts

    Path routing parameters:
        jitter_strength: Strength of path jitter
        jitter_smooth_ksize: Kernel size for jitter smoothing
        center_bias: Bias towards mask center in routing

    Stroke thickness parameters:
        min_radius: Minimum stroke radius
        radius_scale: Radius scale factor (relative to sqrt(area))
        max_radius: Maximum stroke radius

    Soft label parameters:
        use_soft_labels: Generate soft labels instead of binary
        soft_label_sigma: Gaussian sigma for soft labels
    """

    # ===== Multi-scribble =====
    num_scribbles: int = 3
    coverage_cap: float = 0.15
    coverage_check_each_stroke: bool = True

    # ===== Point selection (base rules) =====
    fixed_num_points: int | None = None
    min_points: int = 3
    max_points: int = 7
    mix_center: FloatSpec = 0.5
    center_sigma_scale: FloatSpec = 0.25
    boundary_sigma: FloatSpec = 6.0
    min_dist_scale: FloatSpec = 0.10
    min_dist_min: int = 8
    min_dist_max: int = 40
    max_sampling_trials: int = 500

    # ===== Path routing =====
    jitter_strength: FloatSpec = 0.25
    jitter_smooth_ksize: IntSpec = 9
    center_bias: FloatSpec = 0.10

    # ===== Stroke thickness =====
    min_radius: int = 2
    radius_scale: float = 0.012
    max_radius: int = 20

    # ===== Soft labels =====
    use_soft_labels: bool = False
    soft_label_sigma: float = 2.0
