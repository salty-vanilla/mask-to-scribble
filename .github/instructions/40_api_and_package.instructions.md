# API & Package Expectations

Public API:

- Class-based:
  - `ScribbleGenerator(config).from_mask(mask) -> scribble`
- Functional:
  - `generate_scribble(mask, **kwargs) -> scribble`

Config:

- Use a dataclass `ScribbleConfig`.
- Support per-stroke parameter variation via Spec types:
  - Scalar (same for all strokes)
  - List (per-stroke fixed values)
  - Range (deterministically sampled per stroke)
- Keep defaults tuned for “human-like but sparse”.

Dependency policy:

- Keep core dependencies minimal:
  - numpy, opencv-python, scikit-image
- marimo is optional and should live in apps/tools, not required by core.

Error handling:

- Validate dtype and shape.
- Return empty scribble for empty masks.

Performance:

- Avoid unnecessary recomputation across strokes:
  - dt_inside / centroid should be computed once per mask.
