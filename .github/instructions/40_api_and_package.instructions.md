# API and Package Design

The package must be importable from PU-SAC.

Design principles:
- Core logic must be pure (no I/O)
- Clear and minimal API

Expected usage:
- generate_scribble(mask: np.ndarray) -> np.ndarray
- ScribbleGenerator(config).from_mask(mask)

Avoid:
- Hard-coded dataset assumptions
- Training-time side effects
