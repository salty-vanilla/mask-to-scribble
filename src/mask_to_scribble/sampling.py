"""Point sampling utilities for scribble generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .types import ScribbleConfig


def compute_min_distance(
    area: int,
    *,
    min_dist_scale: float,
    min_dist_min: int,
    min_dist_max: int,
) -> int:
    """Compute minimum distance between sampled points.

    Args:
        area: Mask area in pixels
        min_dist_scale: Scale factor relative to sqrt(area)
        min_dist_min: Minimum absolute distance
        min_dist_max: Maximum absolute distance

    Returns:
        Minimum distance in pixels

    """
    d = int(round(float(min_dist_scale) * np.sqrt(float(area))))
    d = max(d, int(min_dist_min))
    d = min(d, int(min_dist_max))
    return d


def sample_points_with_mixture(
    *,
    m: NDArray[np.uint8],
    dt_inside: NDArray[np.float32],
    centroid_yx: tuple[int, int],
    area: int,
    n: int,
    rng: np.random.Generator,
    mix_center: float,
    center_sigma_scale: float,
    boundary_sigma: float,
    min_dist: int,
    max_sampling_trials: int,
) -> NDArray[np.int32]:
    """Sample points using mixture of center and boundary priors.

    Args:
        m: Binary mask (0/1)
        dt_inside: Distance transform inside mask
        centroid_yx: Mask centroid (y, x)
        area: Mask area in pixels
        n: Number of points to sample
        rng: Random number generator
        mix_center: Mixture ratio (1.0=center only, 0.0=boundary only)
        center_sigma_scale: Center prior spread scale
        boundary_sigma: Boundary prior spread
        min_dist: Minimum distance between points
        max_sampling_trials: Maximum sampling attempts

    Returns:
        (N, 2) array of (x, y) coordinates

    """
    h, w = m.shape
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    cy, cx = centroid_yx
    dx = xs.astype(np.float32) - float(cx)
    dy = ys.astype(np.float32) - float(cy)
    d_center = np.sqrt(dx * dx + dy * dy)

    sigma_c = max(1.0, float(center_sigma_scale) * np.sqrt(float(area)))
    w_center = np.exp(-(d_center * d_center) / (2.0 * sigma_c * sigma_c))

    d_boundary = dt_inside[ys, xs]  # 0 at boundary
    sigma_b = max(1.0, float(boundary_sigma))
    w_boundary = np.exp(-(d_boundary * d_boundary) / (2.0 * sigma_b * sigma_b))

    mix = float(np.clip(mix_center, 0.0, 1.0))
    w_mix = mix * w_center + (1.0 - mix) * w_boundary

    w_sum = float(w_mix.sum())
    if w_sum <= 0:
        w_mix = np.ones_like(w_mix, dtype=np.float32)
        w_sum = float(w_mix.sum())
    p = (w_mix / w_sum).astype(np.float64)

    chosen: list[tuple[int, int]] = []
    trials = 0
    while len(chosen) < n and trials < int(max_sampling_trials):
        trials += 1
        idx = int(rng.choice(len(xs), p=p))
        x = int(xs[idx])
        y = int(ys[idx])

        if not chosen:
            chosen.append((x, y))
            continue

        ok = True
        for px, py in chosen:
            if (x - px) * (x - px) + (y - py) * (y - py) < (min_dist * min_dist):
                ok = False
                break
        if ok:
            chosen.append((x, y))

    return np.array(chosen, dtype=np.int32)


def fallback_points(
    m: NDArray[np.uint8],
    dt_inside: NDArray[np.float32],
    centroid_yx: tuple[int, int],
) -> NDArray[np.int32]:
    """Generate fallback points when sampling fails.

    Selects centroid + boundary extremes (left, right, top).

    Args:
        m: Binary mask (0/1)
        dt_inside: Distance transform inside mask
        centroid_yx: Mask centroid (y, x)

    Returns:
        (N, 2) array of (x, y) coordinates

    """
    cy, cx = centroid_yx
    ys, xs = np.where(m > 0)
    pts = np.stack([xs, ys], axis=1)

    # boundary candidates: smallest dt
    d = dt_inside[ys, xs]
    order = np.argsort(d)
    boundary_pts = pts[order[: max(10, len(order) // 200)]]

    bx = boundary_pts[:, 0]
    by = boundary_pts[:, 1]
    p_left = boundary_pts[int(np.argmin(bx))]
    p_right = boundary_pts[int(np.argmax(bx))]
    p_top = boundary_pts[int(np.argmin(by))]

    out = np.array([[cx, cy], p_left, p_right, p_top], dtype=np.int32)
    out = np.unique(out, axis=0).astype(np.int32)
    return out


def choose_num_points(
    area: int,
    *,
    rng: np.random.Generator,
    k: int,
    config: ScribbleConfig,
) -> int:
    """Choose number of points based on mask area and stroke index.

    Args:
        area: Mask area in pixels
        rng: Random number generator
        k: Stroke index
        config: Scribble configuration

    Returns:
        Number of points to sample

    """
    if config.fixed_num_points is not None:
        return int(max(2, config.fixed_num_points))

    # area-based base (deterministic)
    if area < 2_000:
        base = config.min_points
    elif area < 8_000:
        base = min(config.min_points + 1, config.max_points)
    elif area < 20_000:
        base = min(config.min_points + 2, config.max_points)
    else:
        base = config.max_points

    # allow per-stroke variation for later strokes
    jitter = 0
    if k >= 1:
        jitter = int(rng.integers(-1, 2))  # -1,0,1
    return int(np.clip(base + jitter, 2, config.max_points))
