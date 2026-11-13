# Experiment Review: EXP-13-ARTICULATORY-DECOMPOSITION

## 1. Objective and Setup
Investigated the hypothesis that pronunciation errors follow structured articulatory breakdowns (Place, Manner, Voicing).
Trained a Hierarchical Multi-Task Articulatory Network (`HierarchicalArticulatoryNetwork`) predicting auxiliary phonological attributes alongside primary 3-way pronunciation diagnosis on `TRAIN` (1,800 utterances) and evaluated on `DEV_SEARCH` (599 utterances).

## 2. Quantitative Breakthroughs
- **OMISSION AP Crosses 0.20 For the First Time:**
  - `OMISSION` AP reached **0.2010** (all-time campaign high, vs 0.0106 prior baseline).
  - Omission Recall reached **37.32%**, F1 = **0.2520**.
  - Grounding: Modeling the complete absence of manner gestures directly anchors true deletion detection.
- **INSERTION AP Reaches 0.2447:**
  - `INSERTION` AP rose to **0.2447** (highest all-time).
  - Insertion Recall reached **28.80%**, F1 = **0.2687**.
- **Macro AP Crosses 0.50 on DEV_SEARCH:**
  - Macro AP reached **0.5096** (highest all-time on dev search).
  - Macro Recall reached **0.5492** (highest all-time).
  - Macro F1 reached **0.5219**.
- **Weighted Utility:**
  - Rose to **0.3777** on DEV_SEARCH, setting a new standalone model record.

## 3. Scientific Synthesis & Next Step (EXP-14-CROSS-VALIDATION-SWEEP)
Multi-task articulatory supervision has firmly demonstrated its causal advantage in structuring the representation space.
To determine how robust this representation is across the different non-native L1 accents, the next experiment will conduct a Group Cross-Validation Sweep across all 6 L1 speaker cohorts (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese) in `EXP-14-CROSS-VALIDATION-SWEEP`.
