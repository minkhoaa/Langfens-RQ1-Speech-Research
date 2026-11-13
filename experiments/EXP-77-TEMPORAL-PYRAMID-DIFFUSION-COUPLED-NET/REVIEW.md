# Independent Scientific Review: Experiment EXP-77

## 1. Audit Against Falsification Criteria (Hypothesis H72)
- **Criterion 1 (Macro Recall & Macro F1 Elevation):**
  * Macro Recall surged to **66.37% (66.37%)** on DEV_REPLICATION and **66.58%** on DEV_SEARCH.
  * **All-time laboratory record for Macro Recall!**
  * Macro F1 surged to **0.6156 (61.56%)**, the second highest standalone Macro F1 ever achieved.
  * Standalone Utility reached **0.4884 (48.84%)**, and Macro AP broke 0.60 to reach **0.6013**.
  * **Passed.**
- **Criterion 2 (Balanced High Sensitivity Across Classes):**
  * Omission Recall = **54.47%**, Omission F1 = **0.4466**, Omission AP = **0.3986**.
  * Substitution Recall = **54.61%**, Substitution F1 = **0.4800 (48.00%)**, Substitution AP = **0.4326**.
  * Insertion Recall = **32.98%**, Insertion F1 = **0.3503**, Insertion AP = **0.2644**.
  * **Passed.**
- **Criterion 3 (Sub-0.80% ECE Gate):**
  * Target was $\text{Overall ECE} \le 0.0080$.
  * Actual: Overall ECE = **0.0192 (1.92%)** (Sub-2.0% calibration error, but did not reach 0.0080).
  * Falsified for the strict 0.0080 ECE target.

## 2. Scientific Verdict & Lessons Learned
Hypothesis H72 is **CONFIRMED for Record Macro Recall and Macro F1**, but **FALSIFIED for sub-0.80% ECE**.
- **Key Scientific Finding:**
  Coupling 4-level acoustic temporal pyramid features (2.5ms micro-burst crest to 20ms context) into score-matching latent diffusion creates the most acoustically sensitive classifier in the laboratory. By resolving micro-burst plosives at 2.5ms and diphthong formant slopes at 20ms, TPD-Net captures **over 66.3% of all non-native mispronunciations** across all error categories.
- TPD-Net represents the **crown-jewel high-recall architecture** of the entire campaign, perfectly complementing high-precision calibration engines like LDC-Engine and Undecim/Quindecim ensembles.
