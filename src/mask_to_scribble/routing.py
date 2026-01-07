"""Path routing and jitter field generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.graph import route_through_array

if TYPE_CHECKING:
    from numpy.typing import NDArray


def make_jitter_field(
    shape: tuple[int, int],
    *,
    rng: np.random.Generator,
    ksize: int,
) -> NDArray[np.float32]:
    """Generate smoothed random jitter field.

    Args:
        shape: Output shape (h, w)
        rng: Random number generator
        ksize: Gaussian blur kernel size

    Returns:
        Normalized jitter field [0, 1]

    """
    h, w = shape
    noise = rng.random((h, w), dtype=np.float32)

    k = int(ksize)
    k = max(k, 1)
    if k % 2 == 0:
        k += 1
    if k > 1:
        noise = cv2.GaussianBlur(noise, (k, k), 0).astype(np.float32)

    mn = float(noise.min())
    mx = float(noise.max())
    return (noise - mn) / (mx - mn) if mx > mn else np.zeros_like(noise, dtype=np.float32)


def route_path_inside_mask(
    *,
    m: NDArray[np.uint8],
    dt_inside: NDArray[np.float32],
    a_xy: NDArray[np.int32],
    b_xy: NDArray[np.int32],
    jitter: NDArray[np.float32],
    jitter_strength: float,
    center_bias: float,
) -> NDArray[np.int32] | None:
    """Route a path between two points inside mask using Dijkstra.

    Args:
        m: Binary mask (0/1)
        dt_inside: Distance transform inside mask
        a_xy: Start point (x, y)
        b_xy: End point (x, y)
        jitter: Jitter field
        jitter_strength: Jitter strength coefficient
        center_bias: Center bias coefficient (penalizes boundary)

    Returns:
        Path as (N, 2) array of (y, x) coordinates, or None if routing fails

    """
    h, w = m.shape
    ax, ay = int(a_xy[0]), int(a_xy[1])
    bx, by = int(b_xy[0]), int(b_xy[1])

    ax = int(np.clip(ax, 0, w - 1))
    ay = int(np.clip(ay, 0, h - 1))
    bx = int(np.clip(bx, 0, w - 1))
    by = int(np.clip(by, 0, h - 1))

    if m[ay, ax] == 0 or m[by, bx] == 0:
        return None

    cost = np.full((h, w), np.inf, dtype=np.float32)
    cost[m > 0] = 1.0

    cb = float(center_bias)
    if cb > 0:
        boundary_penalty = 1.0 / (dt_inside + 1e-3)  # big near boundary
        cost[m > 0] = cost[m > 0] + cb * boundary_penalty[m > 0]

    js = float(jitter_strength)
    if js > 0:
        cost[m > 0] = cost[m > 0] + js * jitter[m > 0]

    try:
        path_rc, _ = route_through_array(
            cost,
            start=(ay, ax),
            end=(by, bx),
            fully_connected=True,
        )
    except Exception:
        return None

    return np.array(path_rc, dtype=np.int32)  # (y,x)
