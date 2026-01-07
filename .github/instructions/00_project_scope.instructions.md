# Project Scope

This project provides a deterministic scribble generator
from binary defect masks.

Primary use case:
- Positive-Unlabeled Anomaly Detection (PU-SAC, Dinomaly)
- Datasets with masks only (e.g. MVTecAD)

Key constraints:
- Scribbles are generated ONCE per mask
- No random regeneration during training
- Mimics a single human annotation action

Non-goals:
- Full mask supervision
- Online data augmentation
- Interactive labeling tools
