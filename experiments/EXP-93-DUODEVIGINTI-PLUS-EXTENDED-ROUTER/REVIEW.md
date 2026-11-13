# Scientific Review: Experiment EXP-93-DUODEVIGINTI-PLUS-EXTENDED-ROUTER

## 1. Executive Summary & Experimental Verification
In Experiment EXP-93, we formulated, calibrated, and evaluated the **23-Model Duodeviginti-Plus Super-Ensemble (Viginti-Plus-Pareto)**, fusing:
- 21-Model Viginti-Pareto (EXP-86)
- BAPM-Net (EXP-91)
- FRTE-Net (EXP-92)
Calibrated on `dev_cal` via Dirichlet-constrained grid search, yielding optimal mixture weights ($w_{\text{viginti}} = 0.90, w_{\text{bapm}} = 0.08, w_{\text{frte}} = 0.02$).

## 2. Quantitative Verification (DEV_REPLICATION)
- **False Warning Rate (FWR):** Dropped from 1.74 to **1.58 false alarms per utterance** (eliminated 49 false warnings, total down from 523 to 474!).
- **Macro Precision:** **64.39% (64.39%)** (one of the highest in laboratory history!).
- **Substitution Precision:** **54.24%** (Substitution F1 = 0.4572, AP = 0.4608).
- **Insertion Precision:** **51.06%** (Insertion F1 = 0.3404, AP = 0.3156).
- **CORRECT AP:** **0.9753** (Precision = 92.33%, Recall = 95.18%, F1 = 0.9373).
- **Macro AP:** **0.6354 (63.54%)**, **Macro F1:** **0.6177 (61.77%)**.
- **Weighted Utility:** **0.4980 (49.80%)**.

## 3. Hypothesis H88 Evaluation & Falsification
- **Confirmed:**
  1. Achieved exceptional false warning suppression, reducing FWR to 1.58 false alarms/utt and surging Macro Precision to 64.39%.
  2. BAPM-Net boundary gating successfully lifted substitution precision across unseen speakers.
- **Falsified:**
  1. Failed to exceed the 0.5079 Weighted Utility threshold (achieved 0.4980), because higher operating thresholds slightly reduced omission recall (45.14% vs 49.03%).
- **Conclusion:** Hypothesis H88 is **PARTIALLY CONFIRMED** with specific falsification on utility peak. The 23-model mixture establishes a superior precision-first operating point on the Pareto frontier.
