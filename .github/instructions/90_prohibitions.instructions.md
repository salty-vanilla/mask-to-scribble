# Prohibitions (Do Not Add)

Copilot MUST NOT introduce these approaches:

- Skeletonization (e.g., `skimage.morphology.skeletonize`)
- Contour-driven main logic (e.g., “take contour segment and connect it”)
- PCA / major-axis straight-line strokes
- True random walks without deterministic seed
- Per-epoch / per-iteration scribble regeneration during training
- Any method intended to approximate or reconstruct the full defect mask
- Heavy new dependencies without strong justification
