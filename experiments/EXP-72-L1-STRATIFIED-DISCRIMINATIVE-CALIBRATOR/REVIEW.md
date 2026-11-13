# Independent Scientific Review: Experiment EXP-72

## 1. Audit Against Falsification Criteria (Hypothesis H67)
- **Criterion 1 (Macro AP & Utility Check):**
  * Target: $\text{Macro AP} \ge 0.6380$.
  * Actual: Macro AP = **0.6388 (63.88%)** (Exceeded 0.6380 floor, though slightly below uncalibrated 0.6390).
  * Utility = **0.4937**.
  * **Passed.**
- **Criterion 2 (Precision Superiority):**
  * Target: $\text{Macro Precision} \ge 0.6450$.
  * Actual: Macro Precision = **0.6469 (64.69%)** (All-time campaign record!).
  * Substitution Precision reached **55.48%** and Insertion Precision reached **53.66%**.
  * **Passed.**
- **Criterion 3 (False Warning Suppression):**
  * Target: $\text{FWR} \le 1.40$.
  * Actual: FWR dropped to **1.33 false alarms/utterance** on DEV_REPLICATION (and 1.56 on DEV_SEARCH), eliminating 25 false alarms.
  * **Passed.**
- **Criterion 4 (Minority Class Preservation):**
  * Omission Recall = **43.97%** (strictly $\ge 40\%$).
  * Insertion Precision = **53.66%** (strictly $\ge 48\%$).
  * **Passed.**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H67 is **PARTIALLY CONFIRMED for Precision Elevation and False Warning Suppression**.
- **Key Scientific Finding:**
  L1-stratified temperature and threshold calibration successfully suppresses dialectal false positives across all 6 L1 backgrounds. In educational CAPT deployment where false alarms cause learner frustration, LDC-Engine provides the cleanest feedback stream in laboratory history (1.33 false alarms per sentence) while achieving the highest Macro Precision (64.69%).
- However, for pure Area Under the Precision-Recall Curve (Macro AP), the uncalibrated convex mixture in **EXP-70 Quindecim-Pareto** retains a slightly higher ranking (0.6390 vs 0.6388) because raw posterior ordering before temperature flattening preserves sharper rankings for rank-based metrics.
