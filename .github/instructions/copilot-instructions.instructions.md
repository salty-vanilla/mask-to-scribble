# Copilot Instructions (Entry)

This repository implements a **deterministic, human-like scribble generator**
for **Positive-Unlabeled (PU) anomaly detection**.

Copilot MUST:

- Preserve **determinism**: same mask → bitwise identical scribble.
- Preserve **sparsity**: scribble is NOT a full mask; coverage is capped.
- Use the **canonical multi-scribble algorithm** (K strokes with per-stroke variation).
- Avoid forbidden approaches (skeletonize/contours/PCA/full-mask approximation).

Read and follow all files under `./.github/instructions/` in order:

- 00_project_scope
- 10_scribble_philosophy
- 20_algorithm_design
- 30_determinism_and_repro
- 40_api_and_package
- 50_label_and_thickness
- 60_marimo_usage
- 90_prohibitions
