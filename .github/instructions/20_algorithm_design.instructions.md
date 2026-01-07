# Canonical Algorithm Design (No Skeleton/Contour/PCA)

Copilot MUST implement scribble generation using this canonical pipeline only.

Pre-computation (once per mask):

1. `m`: binary defect mask (0/1)
2. `dt_inside`: distance-to-boundary inside defect region (distance transform)
3. `centroid`: defect centroid

Multi-scribble generation:

- Generate `K` strokes and OR-merge them.
- Each stroke k uses its own deterministic RNG:
  - `rng_k = np.random.default_rng(CRC32(mask_bytes) + k)`

For each stroke k:

1. Sample `N_k` points inside defect:
   - Weighted sampling with a mixture prior:
     - Center prior: favors points near centroid
     - Boundary prior: favors points near boundary (small dt_inside)
   - Enforce minimum inter-point distance.
2. Connect the sampled points:
   - Build a deterministic MST (Prim) among points.
3. Draw the paths:
   - For each MST edge (A,B), route a path inside defect using shortest-path routing
     with a cost map:
     - base cost inside defect = 1
     - outside defect = INF
     - optional center bias (penalize near-boundary pixels)
     - smooth jitter field added (deterministic, per stroke)
4. Merge stroke results into final scribble.

After all strokes:

- Apply thickness (dilation) using radius derived from defect area.
- Clamp scribble inside defect region.
- Apply coverage cap:
  - If exceeded during stroke accumulation: stop adding strokes.
  - If exceeded after dilation: reduce slightly (e.g., small erosion) or tune parameters.

Notes:

- This algorithm intentionally avoids skeletonize/contour/PCA to prevent
  “too clean” geometry and to maintain PU sparsity.
