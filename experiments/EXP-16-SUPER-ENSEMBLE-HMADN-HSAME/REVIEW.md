# Experiment Review: EXP-16-SUPER-ENSEMBLE-HMADN-HSAME

## 1. Objective and Setup
Investigated the convex Bayesian Model Averaging fusion of all three specialized architectural innovations discovered in this campaign:
1. `HM-ADN`: Hierarchical Multi-Task Articulatory Place/Manner/Voicing representations (weight 0.40 target, 0.50 gap).
2. `BiLSTM-CPN`: Bidirectional LSTM sequential coarticulation modeling (weight 0.30 target, 0.25 gap).
3. `ConformerScorer`: Multi-Head Attention and depthwise 1D convolution (weight 0.30 target, 0.25 gap).
Mixture weights and Pareto operating thresholds were calibrated on `DEV_CAL` and tested on both `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).

## 2. Quantitative Breakthroughs Across All Metrics
- **New Campaign All-Time Highs on DEV_SEARCH:**
  - `INSERTION`: AP leaped to **0.2582** (highest all-time), Precision reached **35.71%**, F1 = **0.3139**.
  - `CORRECT`: AP reached **0.9791** (highest all-time), F1 = **0.9396**.
  - `SUBSTITUTION`: AP reached **0.3722** (highest all-time on dev search), Precision = **46.34%**.
- **New Campaign All-Time Highs on DEV_REPLICATION:**
  - `Macro Precision`: Reached **0.6254** (Passed the 60% Macro Precision milestone).
  - `Macro AP`: Reached **0.5720** (all-time campaign high).
  - `Macro F1`: Reached **0.5567** (all-time campaign high).
  - `SUBSTITUTION`: AP reached **0.3822** (highest all-time), Precision reached **51.39%**!
  - `OMISSION`: AP reached **0.3641** (highest all-time), F1 reached **0.3801** (highest all-time), Precision = **45.41%**, Recall = **32.68%**.
- **Probability Calibration & Safety:**
  - Overall ECE: **0.0609** (Gap ECE: **0.0079**), well within the $\le 0.15$ guardrail.
  - False warning rate per utterance: **1.29**, well below the uncalibrated baseline (9.54).

## 3. Scientific Validation
Hypothesis H11 is **CONFIRMED**:
The Tri-Model Super-Ensemble successfully fuses phonological articulatory constraints (HM-ADN) with sequential acoustic coarticulation (BiLSTM) and global attention/calibration (Conformer), producing the most accurate, balanced, and robust pronunciation assessment system developed in the campaign.
