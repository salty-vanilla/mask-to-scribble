"""Geometric utilities for mask analysis and MST construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_centroid_yx(m01: NDArray[np.uint8]) -> tuple[int, int]:
    """Compute the centroid of a binary mask.

    Args:
        m01: Binary mask (0/1)

    Returns:
        (y, x) coordinates of centroid

    """
    ys, xs = np.where(m01 > 0)
    return (int(np.round(ys.mean())), int(np.round(xs.mean())))


def build_mst_edges(pts_xy: NDArray[np.int32]) -> list[tuple[int, int]]:
    """Build minimum spanning tree edges using Prim's algorithm.

    Deterministic: ties broken by lexicographic order of (y, x, index).

    Args:
        pts_xy: Points as (N, 2) array of (x, y) coordinates

    Returns:
        List of edge tuples (parent_idx, child_idx)

    """
    n = len(pts_xy)
    if n < 2:
        return []

    dx = pts_xy[:, 0].astype(np.int64)
    dy = pts_xy[:, 1].astype(np.int64)
    dist2 = (dx[:, None] - dx[None, :]) ** 2 + (dy[:, None] - dy[None, :]) ** 2

    in_tree = np.zeros(n, dtype=bool)
    start = int(np.lexsort((pts_xy[:, 0], pts_xy[:, 1]))[0])  # smallest (y,x)
    in_tree[start] = True

    best_cost = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    best_parent = np.full(n, -1, dtype=np.int32)

    for v in range(n):
        if v == start:
            continue
        best_cost[v] = dist2[start, v]
        best_parent[v] = start

    edges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        candidates = np.where(~in_tree)[0]
        if len(candidates) == 0:
            break

        costs = best_cost[candidates]
        min_cost = costs.min()
        tied = candidates[costs == min_cost]
        if len(tied) > 1:
            ty = pts_xy[tied, 1]
            tx = pts_xy[tied, 0]
            order = np.lexsort((tied, tx, ty))
            u = int(tied[order[0]])
        else:
            u = int(tied[0])

        p = int(best_parent[u])
        if p >= 0:
            edges.append((p, u))
        in_tree[u] = True

        for v in range(n):
            if in_tree[v] or v == u:
                continue
            c = dist2[u, v]
            if c < best_cost[v]:
                best_cost[v] = c
                best_parent[v] = u
            elif c == best_cost[v]:
                prev = int(best_parent[v])
                if prev >= 0:
                    prev_key = (int(pts_xy[prev, 1]), int(pts_xy[prev, 0]), prev)
                    new_key = (int(pts_xy[u, 1]), int(pts_xy[u, 0]), u)
                    if new_key < prev_key:
                        best_parent[v] = u

    return edges
