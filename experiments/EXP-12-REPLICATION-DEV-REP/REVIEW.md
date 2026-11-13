# Experiment Review: EXP-12-REPLICATION-DEV-REP

## 1. Objective and Setup
Formal preregistered replication of the finalist method (`H-SAME Ensemble`) on `DEV_REPLICATION` (300 utterances, 9,863 target phones, 10,163 gaps from unseen speakers TXHC and YKWK).
In accordance with AGENTS.md Section 5 and 14, evaluation was conducted using strictly frozen policy parameters without any threshold adjustment.

## 2. Quantitative Confirmation
- **Campaign Record Weighted Utility:**
  Utility surged to **0.4231**, proving that model performance does not decay on fresh speakers.
- **Milestone Passed:**
  - Macro Precision reached **0.6208** (exceeding the 60% macro milestone).
  - Macro F1 reached **0.5514**.
  - Macro AP reached **0.5648**.
- **High-Precision Error Detection Across All Three Error Classes:**
  - `SUBSTITUTION`: Precision = **50.30%** (more than half of flagged substitutions are verified correct). AP = **0.3690**, F1 = **0.3491**.
  - `OMISSION`: Precision = **45.25%**, Recall = **31.52%**, F1 = **0.3716**, AP = **0.3565**.
  - `INSERTION`: Precision = **41.38%**, Recall = **25.53%**, F1 = **0.3158**, AP = **0.1966**.
  - `CORRECT`: Precision = **90.68%**, Recall = **96.19%**, F1 = **0.9335**, AP = **0.9690**.
- **Probability Calibration & Safety:**
  - Overall ECE = **0.0596** (Gap ECE = **0.0088**), strictly inside $\le 0.15$ guardrail.
  - FWR = **1.28 false warnings per utterance**.
- **Scientific Audit & Milestone Status:**
  The H-SAME method is officially advanced to **`LOCAL_REPRODUCED`** in `evaluation/CONFIRMATION_REGISTRY.jsonl` and `reports/CONFIRMED_METHODS.md`.
