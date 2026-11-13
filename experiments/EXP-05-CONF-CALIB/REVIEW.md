# Experiment Review: EXP-05-CONF-CALIB

## 1. Objective and Setup
Investigated confidence-gated calibration on `DEV_CAL` to test whether explicit threshold gating could suppress false warnings without destroying minority class recall.

## 2. Key Findings
- **Substantial F1 and Precision Gains:**
  - `SUBSTITUTION` F1 increased to **0.4026** (Precision reached 0.3127).
  - `OMISSION` F1 increased to **0.2034** (Precision reached 0.1253, Recall maintained at 54.07%).
  - `CORRECT` F1 reached **0.8925** (Precision reached 0.9529).
  - Macro F1 reached **0.4995** (close to 0.50 milestone).
  - Total false alarms dropped by 47% (from 5,714 down to 3,011).
- **Remaining Limitation:**
  FWR remains at 5.03 false alarms/utterance (guardrail is $\le 0.20$). Because EXP-04 was trained with an aggressive loss weight ($\alpha = [0.2, 1.0, 3.0]$), the model's raw logits are inherently biased toward high false positive rates.
- **Next Hypothesis & Architecture (EXP-06-CALIB-TRAIN):**
  Train a Well-Calibrated Cost-Sensitive Network with Margin Regularization and Balanced Priors, combined with a dynamic FWR-constrained threshold search on `DEV_CAL`.
