# Algorithm Design

Scribbles are composed of:

1. Skeleton stroke
- Extract skeleton
- Keep only the main branch
- Remove short noisy branches

2. Contour stroke
- Extract defect contour
- Use only a partial segment (20–40%)
- Avoid tracing the full boundary

3. Optional connecting stroke
- Connect skeleton and contour
- Use shortest-path inside defect region
- Avoid unconstrained random walk

Each component must be interpretable and minimal.
