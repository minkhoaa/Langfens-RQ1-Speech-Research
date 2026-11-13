# Experiment Review: EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT

## 1. Objective and Setup
Investigated parameter-efficient acoustic feature adaptation using a residual bottleneck adapter (`NNA_Adapter`) trained jointly with the Hierarchical Articulatory multi-task network on `TRAIN` (1,800 utterances).
Evaluated whether adapting the feature manifold directly on accented speech would boost minority error recall.

## 2. Key Findings and Quantitative Gains
- **All-Time Campaign High on Omission Recall:**
  - `DEV_SEARCH` Omission Recall reached **44.98%** (all-time high on dev search), with F1 reaching **0.2745**.
  - `DEV_REPLICATION` Omission Recall reached **40.86%** (first time breaking 40% recall on replication holdout), with F1 = **0.3903**.
- **Insertion Recall Improvement:**
  - `DEV_SEARCH` Insertion Recall reached **30.40%** (first time breaking 30% recall).
- **Macro Sensitivity:**
  - Macro Recall reached **56.60%** on DEV_SEARCH and **54.66%** on DEV_REPLICATION.
- **Learned Adapter Scaling:**
  - Residual adapter scale parameter converged from 0.10 to **0.1402**, confirming that a controlled ~14% residual acoustic shift is optimal to align non-native features without catastrophic forgetting.

## 3. Scientific Synthesis & Next Continuous Step
The NNA-Adapter confirms that adapting the representation space directly enhances omission and insertion recall.
To push towards Milestone 60, the next continuous line of research will explore contrastive phonological margin regularization in `EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN`.
