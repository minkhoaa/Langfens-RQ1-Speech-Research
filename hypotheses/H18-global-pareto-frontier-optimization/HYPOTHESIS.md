# Hypothesis H18: Global Quad-Model Composite Pareto Optimization (QC-Pareto)

## 1. Proposition
Each evolutionary stage in this campaign achieved peak performance on a distinct quality dimension:
1. `PSC-Net` (EXP-22): Peak Substitution AP (0.3874) and Substitution Precision (51.29%).
2. `SRAD-Net` (EXP-17): Peak Omission F1 (0.3926) and Macro F1 (0.5671) via tempo normalization.
3. `NNA-Adapter` (EXP-21): Peak Omission Recall (44.98% dev search, 40.86% replication) via adapted acoustic features.
4. `ConformerScorer` (EXP-10): Peak Insertion Precision (28.93%) and lowest ECE (0.0353).

A convex Quad-Model Bayesian Mixture:
$$\mathbf{p}_{\text{target}} = w_1 \mathbf{p}_{\text{PSC}} + w_2 \mathbf{p}_{\text{SRAD}} + w_3 \mathbf{p}_{\text{NNA}} + w_4 \mathbf{p}_{\text{Conf}}$$
optimized via exhaustive grid sweep on `DEV_CAL` will:
1. Eliminate remaining localized error modes by consensus voting across the 4 diverse model architectures,
2. Drive `OMISSION` F1 past the **0.40 milestone** on `DEV_REPLICATION`,
3. Elevate Macro F1 beyond **0.575**,
4. Push total Weighted Utility beyond **0.435** on `DEV_REPLICATION`.

## 2. Experimental Plan
- Method ID: `M23-global-pareto-sweeper`
- Run ID: `EXP-23-COMPOSITE-UTILITY-SWEEP`
- Load checkpoints from EXP-10, EXP-17, EXP-21, EXP-22.
- Perform global grid search over mixture weights and thresholds on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
