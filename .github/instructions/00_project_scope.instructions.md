# Project Scope

Goal:

- Generate **human-like scribble annotations** from **binary defect masks**.
- Scribbles are used for **PU learning** (Positive = scribble pixels, Unlabeled = remaining pixels).

Non-goals:

- Approximating the full defect mask.
- Creating stochastic augmentation during training.
- Any method that reconstructs unlabeled defect regions as positive labels.

Inputs / outputs:

- Input mask: `uint8 (H, W)` where defect pixels are `>0` (typically 255).
- Output scribble:
  - `uint8` in `{0, 255}` by default
  - `float32` in `[0, 1]` if soft labels are enabled

Constraints:

- Deterministic generation (bitwise identical for the same mask).
- Coverage cap enforced to avoid scribble becoming a full mask.
