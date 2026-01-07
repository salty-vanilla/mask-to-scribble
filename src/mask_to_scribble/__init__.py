"""mask-to-scribble: Generate human-like scribble annotations from defect masks."""

from mask_to_scribble.generator import ScribbleGenerator, generate_scribble
from mask_to_scribble.types import FloatSpec, IntSpec, RangeF, RangeI, ScribbleConfig

__all__ = [
    "FloatSpec",
    "IntSpec",
    "RangeF",
    "RangeI",
    "ScribbleConfig",
    "ScribbleGenerator",
    "generate_scribble",
]
