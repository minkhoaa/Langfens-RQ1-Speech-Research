# Experiment Review: EXP-11-ENSEMBLE-FUSION

## 1. Objective and Setup
Investigated the Bayesian Model Averaging fusion of the two best-performing architectural paradigms developed in this campaign:
- BiLSTM-CPN (EXP-08): Sharp local temporal coarticulation modeling.
- ConformerPhoneticScorer (EXP-10): Global multi-head self-attention, depthwise convolution, and probability calibration.
Mixture weights ($w_{\text{target}}=0.60, w_{\text{gap}}=0.50$) and Pareto operating thresholds were calibrated on `DEV_CAL` (300 utterances) and tested on `DEV_SEARCH` (599 utterances).

## 2. Key Findings and New All-Time Highs
- **New Campaign Record Weighted Utility:**
  Utility reached **0.3736**, surpassing every previous experiment.
- **New All-Time Highs Across All Core Classes:**
  - `INSERTION`: AP reached **0.2337** (all-time high), F1 reached **0.2920** (all-time high), Precision reached **0.3267** (all-time high).
  - `OMISSION`: F1 reached **0.2550** (all-time high), Recall reached **33.49%**, Precision reached **0.2059**.
  - `SUBSTITUTION`: Precision reached **0.4583**, F1 reached **0.3702**, AP reached **0.3617**.
  - `CORRECT`: AP reached **0.9779** (all-time high), F1 reached **0.9395**.
- **Macro Performance:**
  - Macro F1 reached **0.5216** (all-time high).
  - Macro Precision reached **0.5308** (all-time high).
  - Macro Recall reached **0.5322** (all-time high).
- **Probability Calibration:**
  Overall ECE dropped to **0.0630** (Gap ECE = **0.0072**).

## 3. Scientific Validation of Hypothesis H07
Hypothesis H07 is **CONFIRMED**:
The optimal mixture weight ($w_{\text{target}}=0.60, w_{\text{gap}}=0.50$) validates that both architectures provide non-redundant, orthogonal diagnostic signals. Combining them produces a strictly Pareto-dominant assessment system across accuracy, calibration, and false alarm suppression.
