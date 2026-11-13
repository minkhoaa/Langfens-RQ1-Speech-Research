# Experiment Review: EXP-07-PARETO-CALIB

## 1. Objective and Setup
Pareto-optimal threshold calibration on `DEV_CAL` to find the operating trade-off between minority class recall and false alarm suppression.

## 2. Key Findings and Milestones
- **First Run to Cross 50% Macro-F1:**
  - Macro F1 reached **0.5082** (vs 0.3158 in EXP-01, 0.3914 in EXP-03, 0.4401 in EXP-04).
  - Macro Precision reached **0.5220**.
  - Macro Recall reached **0.5030**.
- **Balanced Class Metrics:**
  - `CORRECT`: All 4 core metrics exceed 0.80 (AP=0.9760, P=0.9286, R=0.9511, F1=0.9397).
  - `SUBSTITUTION`: Precision reached **0.4594**, Recall = **33.31%**, F1 = **0.3862**, AP = **0.3640**.
  - `OMISSION`: Precision reached **0.1780**, Recall = **22.49%**, F1 = **0.1987**, AP = **0.1354**.
  - `INSERTION`: Precision reached **0.1538**, AP = **0.0332**.
- **False Alarm Suppression:**
  False alarms reduced by 84% compared to EXP-04 (from 5,714 down to 921), bringing FWR to 1.54 false alarms per utterance.
- **Overall ECE:** 0.1047 (comfortably within the $\le 0.15$ guardrail).
