# mask-to-scribble

Generate **human-like scribble annotations** from binary defect masks for Positive-Unlabeled (PU) learning.

## Overview

This package converts binary defect masks into sparse, line-like scribble annotations that mimic human annotation behavior. Unlike dense segmentation masks, scribbles intentionally leave most defect pixels unlabeled, making them ideal for PU learning scenarios where:

- **Positive labels** = scribble pixels
- **Unlabeled pixels** = everything else (including unlabeled defect regions)

### Key Features

- **Deterministic generation**: Same mask always produces identical scribbles (bitwise reproducible)
- **Human-like strokes**: Slightly wobbly paths with mixture of center and boundary tendencies
- **Multi-stroke support**: Generate 1-3 strokes per mask with coverage cap enforcement
- **Configurable parameters**: Fine-tune point sampling, routing, and thickness per stroke
- **Soft labels**: Optional Gaussian-weighted soft labels for each stroke

## Installation

```bash
pip install mask-to-scribble
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add mask-to-scribble
```

## Quick Start

### Functional API

```python
import numpy as np
from mask_to_scribble import generate_scribble

# Create or load a binary mask (uint8, defect pixels = 255)
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:150, 100:150] = 255

# Generate scribble with defaults
scribble = generate_scribble(mask)

# Or customize parameters
scribble = generate_scribble(
    mask,
    num_scribbles=2,
    num_points=(5, 10),
    thickness=(1.5, 3.0),
    use_soft_labels=True,
)
```

### Class-based API

```python
from mask_to_scribble import ScribbleGenerator, ScribbleConfig

# Configure generator
config = ScribbleConfig(
    num_scribbles=2,
    num_points=(5, 10),  # Range: sample 5-10 points per stroke
    thickness=(1.5, 3.0),  # Range: thickness between 1.5-3.0
    mix_center=0.5,  # 50% center prior, 50% boundary prior
    use_soft_labels=True,
)

generator = ScribbleGenerator(config)

# Generate scribbles
scribble = generator.from_mask(mask)
```

## Algorithm

Each scribble stroke is generated through the following pipeline:

1. **Point Sampling**: Select N points inside the mask using a mixture of:

   - **Center prior**: Points near the mask centroid (Gaussian distribution)
   - **Boundary prior**: Points near the mask boundary (distance transform weighted)

2. **MST Construction**: Build a Minimum Spanning Tree (MST) to connect points deterministically

3. **Path Routing**: Route each MST edge through the mask interior using Dijkstra's algorithm with jittered cost fields for natural-looking paths

4. **Stroke Combination**: Combine K strokes with logical OR operation

5. **Thickening**: Dilate strokes with configurable radius (optional soft labels via Gaussian weighting)

### Determinism

Reproducibility is guaranteed through:

- Base seed = CRC32(mask bytes)
- Per-stroke RNG: `numpy.random.default_rng(base_seed + k)`
- All operations use seeded random number generation

## Configuration

### Parameter Specification Types

Parameters can be specified as:

- **Scalar**: Same value for all strokes (e.g., `thickness=2.0`)
- **List**: Fixed per-stroke values (e.g., `thickness=[1.5, 2.0, 2.5]`)
- **Range tuple**: Deterministically sampled per stroke (e.g., `thickness=(1.5, 3.0)`)

### ScribbleConfig Options

| Parameter              | Type        | Default   | Description                                          |
| ---------------------- | ----------- | --------- | ---------------------------------------------------- |
| `num_scribbles`        | `int`       | `1`       | Number of strokes to generate                        |
| `num_points`           | `IntSpec`   | `(5, 15)` | Points to sample per stroke                          |
| `mix_center`           | `FloatSpec` | `0.5`     | Center prior weight (0=boundary only, 1=center only) |
| `center_sigma_scale`   | `FloatSpec` | `0.25`    | Center prior spread (× sqrt(area))                   |
| `boundary_sigma_scale` | `FloatSpec` | `0.15`    | Boundary prior spread (× max distance)               |
| `min_dist_scale`       | `FloatSpec` | `0.08`    | Minimum point separation (× sqrt(area))              |
| `jitter_scale`         | `FloatSpec` | `0.15`    | Path wobbliness (× local distance)                   |
| `thickness`            | `FloatSpec` | `2.0`     | Stroke radius in pixels                              |
| `thickness_scale`      | `FloatSpec` | `0.0`     | Thickness scaling by defect size                     |
| `use_soft_labels`      | `bool`      | `False`   | Generate Gaussian soft labels                        |
| `soft_sigma_ratio`     | `float`     | `0.4`     | Soft label spread (× radius)                         |
| `coverage_cap`         | `float`     | `0.3`     | Maximum coverage ratio (prevents full masks)         |

## Input/Output

### Input

- Binary mask: `numpy.ndarray` with shape `(H, W)` and dtype `uint8`
- Defect pixels should have value > 0 (typically 255)
- Background pixels should be 0

### Output

- **Binary scribble**: `uint8` array with values in `{0, 255}`
- **Soft scribble**: `float32` array with values in `[0, 1]` (when `use_soft_labels=True`)

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mask-to-scribble.git
cd mask-to-scribble

# Install with dev dependencies using uv
uv sync --all-groups
```

### Visualization Notebooks

Interactive [marimo](https://github.com/marimo-team/marimo) notebooks are provided for exploration:

```bash
# Visualize scribble generation
uv run marimo edit notebooks/visualize_scribbles.py

# MVTec dataset experiments
uv run marimo edit notebooks/mvtec.py
```

### Testing

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check --fix
```

## Non-Goals

This package is **not** designed for:

- Approximating or reconstructing the full defect mask
- Creating stochastic augmentation during training
- Any method that treats unlabeled defect regions as positive labels

The goal is to generate **sparse, human-like annotations** that preserve uncertainty in unlabeled regions for PU learning.

## License

MIT
