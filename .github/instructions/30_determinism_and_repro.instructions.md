# Determinism & Reproducibility (Hard Requirement)

All variability MUST be deterministic.

Allowed:

- `seed = CRC32(mask_bytes)`
- `rng = np.random.default_rng(seed + k)` for stroke index k
- Deterministic math (no global RNG dependence)

Forbidden:

- `np.random.*` without explicit generator
- Python `random`
- time-based seeds
- reliance on global state
- any augmentation that changes scribble per epoch

Definition:

- For the same `mask` input, `from_mask(mask)` MUST return bitwise identical output.

Testing:

- Add a unit test that calls generation twice on the same mask and asserts equality.
