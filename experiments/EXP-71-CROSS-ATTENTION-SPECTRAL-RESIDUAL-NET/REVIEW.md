# Independent Scientific Review: Experiment EXP-71

## 1. Audit Against Falsification Criteria (Hypothesis H66)
- **Criterion 1 (Macro AP Superiority):**
  * Target: $\text{Macro AP} \ge 0.6250$.
  * Actual: Macro AP = **0.5921 (59.21%)**.
  * **Failed (Falsified).**
- **Criterion 2 (Utility Superiority):**
  * Target: $\text{Weighted Utility} \ge 0.4950$.
  * Actual: Utility = **0.4673**.
  * **Failed (Falsified).**
- **Criterion 3 (False Warning Rate & Precision):**
  * Target: $\text{FWR} \le 1.85$.
  * Actual: FWR = **3.517 false alarms/utterance**, Precision = **58.46%**.
  * **Failed (Falsified).**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H66 is declared **FALSIFIED** for raw, unadapted cepstral residuals.
- **Root Cause Analysis:**
  Concatenating 8-dimensional static spectral cepstral residuals $\Delta_{\text{cep}} = |c_{\text{learner}} - \bar{c}_{\text{canonical}}|$ directly into an unadapted acoustic stream increases model sensitivity to all acoustic discrepancies indiscriminately. This pushes Substitution Recall to 57.65% and Macro Recall to 63.78%, but conflates harmless non-native vocal tract length variations and dialectal formant offsets with genuine phonetic mispronunciations.
- **Architectural Directive for Next Steps:**
  Spectral residuals cannot be used naively as raw features without speaker-level vocal tract length normalization (VTLN) or dialect FiLM adaptation. Standalone championship remains firmly with **EXP-69 IPTM-Net** (Macro F1 = 0.6205, Omission Recall = 55.64%) and **EXP-55 DSCA-Net** (Utility = 0.4928), while ensemble championship remains with **EXP-70 Quindecim-Pareto** (Macro AP = 0.6390, Precision = 64.67%, Utility = 0.4989).
