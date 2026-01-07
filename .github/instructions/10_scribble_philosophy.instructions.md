# Scribble Philosophy (PU Context)

A scribble represents one human annotation action:

- Sparse strokes indicating representative parts of the defect.
- Most defect pixels remain **unlabeled**.

Why:

- PU learning requires keeping uncertainty in unlabeled defect regions.
- Full-mask-like scribbles collapse PU into supervised segmentation.

Desired scribble properties:

- Sparse (line-like), not area-like.
- Covers both interior and boundary tendencies across multiple strokes.
- Slightly imperfect / wobbly paths (human-like), but still within defect region.
- Reproducible and stable.

Multiple strokes:

- Humans often add 1–3 extra strokes to clarify.
- We support multiple strokes, but cap coverage to prevent overspecification.
