# marimo Usage (Tuning & Debug Only)

marimo is used for:

- interactive parameter tuning
- visualization (mask, scribble, overlay)
- debugging (sampled points, MST edges, routed paths)

marimo must not:

- be required for importing the package
- change the core algorithm
- introduce non-deterministic behavior in the library

Recommended marimo features:

- sliders for K, coverage_cap, jitter_strength, mix_center, N points
- debug toggle to show points + MST edges + routed paths
- export selected parameters as a config snippet
