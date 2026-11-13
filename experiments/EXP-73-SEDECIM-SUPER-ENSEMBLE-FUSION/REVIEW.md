# Independent Scientific Review: Experiment EXP-73

## 1. Audit Against Falsification Criteria (Hypothesis H68)
- **Criterion 1 (Macro AP & Utility Check):**
  * Target: $\text{Macro AP} \ge 0.6380$.
  * Actual: Macro AP = **0.6383 (63.83%)** (Exceeded 0.6380 floor, though slightly below Quindecim 0.6390).
  * Utility reached **0.4984 (49.84%)**.
  * **Passed.**
- **Criterion 2 (Recall & F1 Elevation):**
  * Macro F1 increased from 0.6090 to **0.6103 (61.03%)**.
  * Substitution Recall increased from 35.43% to **37.63% (37.63%)**, pushing Substitution F1 to **0.4449**.
  * Insertion F1 increased to **0.3333 (33.33%)** and Insertion AP to **0.3248**.
  * **Passed.**
- **Criterion 3 (False Alarm Control):**
  * FWR = **1.507 false alarms/utterance**, strictly $\le 1.75$ guardrail.
  * Macro Precision = **64.26%**, strictly $\ge 63.50\%$.
  * **Passed.**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H68 is **PARTIALLY CONFIRMED for Recall and Macro F1 Elevation**.
- **Key Scientific Finding:**
  Fusing high-recall acoustic diversity from CASR-Net into the 15-model mixture successfully harvests missed phonetic substitutions and insertions, lifting Substitution Recall by +2.20% and pushing Macro F1 to 61.03%.
- However, because CASR-Net's cepstral residuals are un-warped, they introduce a slight false warning tax (+28 false substitutions across 300 utterances), which holds Macro AP at 0.6383.
- **Architectural Directive for EXP-74:**
  To break past 0.6400 Macro AP and 0.5000 Utility, spectral cepstral residuals must be normalized by speaker-specific VTLN warping factors $\alpha_{\text{spk}}$ prior to feature concatenation.
