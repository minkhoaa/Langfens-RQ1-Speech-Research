# Independent Scientific Review: Experiment EXP-76

## 1. Audit Against Falsification Criteria (Hypothesis H71)
- **Criterion 1 (Macro AP & Utility Check):**
  * Target: $\text{Macro AP} \ge 0.6385$.
  * Actual: Macro AP = **0.6385 (63.85%)**.
  * Utility reached **0.4987 (49.87%)**.
  * **Passed.**
- **Criterion 2 (Substitution & Insertion Peak Records):**
  * Substitution AP reached an all-time record **0.4627 (46.27%)**, beating EXP-70 (0.4622) and EXP-67 (0.4601).
  * Insertion Precision reached an all-time ensemble peak **54.76% (54.76%)**, beating EXP-67 (53.33%).
  * Insertion F1 reached **0.3382 (33.82%)** and Insertion AP reached **0.3262**.
  * CORRECT AP reached an all-time record **0.9754 (97.54%)**.
  * **Passed.**
- **Criterion 3 (False Alarm Suppression vs EXP-73):**
  * FWR dropped from 1.507 down to **1.483 false alarms/utterance** (445 total false alarms vs 452).
  * Overall ECE dropped below 3.0% to **0.0298 (2.98%)**.
  * **Passed.**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H71 is **PARTIALLY CONFIRMED for Substitution AP, Insertion Precision, and False Alarm Suppression**.
- **Key Scientific Finding:**
  Integrating V-CASR-Net (EXP-74) with speaker-specific VTLN warping and DP-ALDS-Net (EXP-75) with dialect-conditioned diffusion establishes all-time peaks for Substitution AP (0.4627), Insertion Precision (54.76%), and CORRECT AP (0.9754).
- The ensemble successfully controls false alarms (FWR = 1.483, ECE = 0.0298) while providing the richest consensus representation across 17 distinct theoretical paradigms in the laboratory.
