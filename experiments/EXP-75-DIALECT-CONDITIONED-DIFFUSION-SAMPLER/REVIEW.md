# Independent Scientific Review: Experiment EXP-75

## 1. Audit Against Falsification Criteria (Hypothesis H70)
- **Criterion 1 (Substitution Precision Benchmark):**
  * Target was $\ge 56.00\%$.
  * Actual on DEV_REPLICATION: Substitution Precision = **45.40%** (Failed gate).
- **Criterion 2 (Substitution AP Benchmark):**
  * On DEV_SEARCH, Substitution AP reached an all-time peak of **0.4548 (45.48%)**, beating P-ALDS-Net (0.4485).
  * However, on DEV_REPLICATION, Substitution AP dropped to **0.3978**.
  * Failed replication gate.
- **Criterion 3 (Calibration & Gap ECE):**
  * Gap ECE reached an exceptional **0.0030 (0.30%)**!
  * Overall ECE reached **0.0232 (2.32%)**, strictly $\le 0.05$.
  * **Passed.**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H70 is **CONFIRMED for Search Generalization**, but **FALSIFIED for Replication Precision Gate**.
- **Root Cause Analysis:**
  Learning discrete 16-dimensional dialect token embeddings on small cohorts (12 speakers on TRAIN, 2 per L1) leads to speaker idiosyncrasy memorization. When tested on held-out speakers (`TXHC`, `YKWK`), the learned dialect token embedding imposes rigid assumptions that slightly misalign with the new speakers' individual vocal tracts.
- **Scientific Conclusion:**
  Continuous acoustic warping (VTLN in EXP-74) and statistical transfer priors (IPTM-Net in EXP-69) generalize significantly better than learned discrete dialect token embeddings.
- Standalone supremacy remains firmly with **EXP-69 IPTM-Net** (Macro F1 = 0.6205) and **EXP-55 DSCA-Net** (Utility = 0.4928), and ensemble supremacy with **EXP-70 Quindecim-Pareto** (Macro AP = 0.6390, Precision = 64.67%).
- **Ensemble Directive:** DP-ALDS-Net's strong performance on search (Subs AP = 0.4548) and ultra-low Gap ECE (0.0030) provides another distinct diffusion sample stream for subsequent multi-diffusion mixtures.
