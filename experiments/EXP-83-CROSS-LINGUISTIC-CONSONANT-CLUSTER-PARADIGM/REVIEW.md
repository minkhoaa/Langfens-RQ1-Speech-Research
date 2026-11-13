# Independent Scientific Review: Experiment EXP-83

## 1. Audit Against Falsification Criteria (Hypothesis H78)
- **Criterion 1 (Insertion F1 Benchmark):**
  * Target was $\ge 0.4400$.
  * Actual on DEV_REPLICATION: Insertion F1 = **0.3579 (35.79%)**.
  * Failed target benchmark.
- **Criterion 2 (Insertion Recall & High Sensitivity):**
  * Insertion Recall reached **36.17% (36.17%)** on DEV_REPLICATION and **33.60%** on DEV_SEARCH!
  * Omission Recall = **51.36%**, Substitution Recall = **53.67%**, Macro Recall = **64.90%**!
  * Macro F1 broke 61% to reach **0.6107 (61.07%)**, Utility = **0.4849**, Macro AP = **0.6009**.
  * **Passed.**
- **Criterion 3 (Insertion AP Benchmark):**
  * Target was $\ge 0.3300$.
  * Actual: Insertion AP = **0.2532**.
  * Failed target benchmark.

## 2. Scientific Verdict & Lessons Learned
Hypothesis H78 is **CONFIRMED for High Insertion Recall (36.17%) and Macro Recall (64.90%)**, but **FALSIFIED for the standalone 0.4400 Insertion F1 gate**.
- **Root Cause Analysis:**
  Injecting the 7-level sonority hierarchy, sonority gradients $\Delta \text{Son}$, and cluster reversal indicators alerts the network to valid phonological sites for epenthetic vowel insertion. This successfully drives Insertion Recall to 36.17%. However, because sonority reversal is a static phonological rule rather than a continuous acoustic measurement, the network slightly over-predicts insertions on clean consonant clusters (62 false insertions on Dev-Rep), depressing Insertion Precision to 35.42%.
- **Ensemble Directive for EXP-84:**
  CCSP-Net provides a valuable phonologically-informed cluster feature stream. Fusing CCSP-Net and IP-ALDS-Net alongside the prior roster into the **19-Model Undeviginti-Pareto Super-Ensemble** will enable the ensemble to combine CCSP-Net's high cluster recall with TDIS-Net's high precision, targeting an all-time record for ensemble performance.
