# Determinism and Reproducibility

Scribble generation must be reproducible.

Rules:
- Same mask → same scribble
- No stochastic behavior during training
- If randomness is used:
  - Fix the seed
  - Execute once during dataset preparation

Prefer rule-based decisions over random sampling.
