# Labels, Thickness, and Coverage

Binary labels:

- Default output is `uint8` with values {0, 255}.

Soft labels (optional):

- If enabled, return `float32` in [0, 1] using distance-based Gaussian decay from strokes.

Thickness:

- Determine stroke radius from defect area:
  - radius ~ sqrt(area) \* radius_scale
  - clamp by min_radius and max_radius
- Apply dilation once at the end (preferred).
- Clamp scribble within defect after dilation.

Coverage cap (critical):

- Define coverage:
  - `coverage = scribble_pixels_inside_defect / defect_pixels`
- Enforce:
  - stop adding strokes if coverage exceeds `coverage_cap`
  - optionally reduce after dilation if needed

Target ranges:

- `coverage_cap`: 0.10 – 0.20 (default ~0.15)
- `num_scribbles`: typically 1–3
