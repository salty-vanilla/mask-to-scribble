"""Utilities for parameter selection from specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .types import FloatSpec, IntSpec, RangeF, RangeI

if TYPE_CHECKING:
    import numpy as np


def pick_float(spec: FloatSpec, rng: np.random.Generator, k: int, default: float) -> float:
    """Select a float value from a FloatSpec.

    Args:
        spec: Float specification (scalar, sequence, range, or None)
        rng: Random number generator
        k: Stroke index (for sequence lookup)
        default: Default value if spec is None

    Returns:
        Selected float value

    """
    if spec is None:
        return float(default)
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, RangeF):
        return float(rng.uniform(spec.lo, spec.hi))
    if isinstance(spec, Sequence):
        if len(spec) == 0:
            return float(default)
        return float(spec[min(k, len(spec) - 1)])
    msg = f"Unsupported FloatSpec: {type(spec)}"
    raise TypeError(msg)


def pick_int(spec: IntSpec, rng: np.random.Generator, k: int, default: int) -> int:
    """Select an integer value from an IntSpec.

    Args:
        spec: Integer specification (scalar, sequence, range, or None)
        rng: Random number generator
        k: Stroke index (for sequence lookup)
        default: Default value if spec is None

    Returns:
        Selected integer value

    """
    if spec is None:
        return int(default)
    if isinstance(spec, int):
        return int(spec)
    if isinstance(spec, RangeI):
        return int(rng.integers(spec.lo, spec.hi + 1))  # inclusive
    if isinstance(spec, Sequence):
        if len(spec) == 0:
            return int(default)
        return int(spec[min(k, len(spec) - 1)])
    msg = f"Unsupported IntSpec: {type(spec)}"
    raise TypeError(msg)
