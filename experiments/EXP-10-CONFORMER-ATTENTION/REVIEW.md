# Experiment Review: EXP-10-CONFORMER-ATTENTION

## 1. Objective and Setup
Investigated a Multi-Head Self-Attention Conformer architecture with 1D depthwise separable convolution (`ConformerPhoneticScorer`) on `DEV_SEARCH` (599 utterances).
Evaluated whether attention over the entire prompt sequence combined with local convolutional filters would improve upon BiLSTM.

## 2. Key Findings and Quantitative Trade-offs
- **New Campaign Highs on Insertion Detection:**
  - `INSERTION` Precision reached **0.2893** (highest in campaign).
  - `INSERTION` Recall reached **28.00%** (highest in campaign).
  - `INSERTION` F1 reached **0.2846** (highest in campaign).
- **Superior Probability Calibration:**
  - Overall ECE dropped to **0.0353** (lowest among all learned models).
  - Gap ECE reached an unprecedented **0.0086**.
- **The Attention Dispersion Effect:**
  - While BiLSTM (EXP-08) maintained higher localized sharpness for substitutions (F1 = 0.3674, AP = 0.3483), unconstrained full self-attention in Conformer slightly dispersed local transition focus across long sentences, leading to substitution F1 = 0.2919 and AP = 0.3071.
  - Overall utility remained very strong at **0.3546** (second highest in campaign).
- **Synthesis for Next Iteration (EXP-11-ENSEMBLE-FUSION):**
  Combine the sharp local acoustic transition modeling of BiLSTM (EXP-08) with the superior gap precision and calibration of Conformer (EXP-10) via Bayesian Model Averaging / Ensembling on `DEV_CAL`.
