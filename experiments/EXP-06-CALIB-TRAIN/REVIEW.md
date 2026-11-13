# Experiment Review: EXP-06-CALIB-TRAIN

## 1. Objective and Setup
Investigated balanced focal loss training ($\alpha = [1.0, 1.5, 2.0]$) with explicit FWR guardrail calibration on `DEV_CAL` to test whether false alarm suppression could unlock the coverage risk utility.

## 2. Quantitative Breakthroughs
- **Guardrail Satisfied:** False Warning Rate dropped to **0.1119 false alarms/utterance**, strictly satisfying the $\le 0.20$ guardrail for the first time in the campaign (only 67 false alarms across 599 utterances).
- **Campaign Record Weighted Utility:** Utility rose to **0.3404**, surpassing EXP-01-PRIOR (0.3183), EXP-03-SSL-GOP (0.2522), and EXP-04-LPDN (0.2981).
- **Record Average Precision:**
  - `CORRECT` AP reached **0.9760** (F1 = 0.9461).
  - `SUBSTITUTION` AP reached **0.3640** (highest in the campaign).
  - `OMISSION` AP reached **0.1354** (highest in the campaign).
  - `Macro AP` reached **0.4918**.

## 3. Critical Failure Diagnostics
- **Over-Conservative Substitution Threshold:** Because $\theta_{\text{subs}} = 0.75$ on `DEV_CAL` was calibrated on speakers with higher accent intensity, when applied to `DEV_SEARCH`, no substitutions cleared the 0.75 threshold (Precision and Recall = 0.0).
- **Core Insight for Next Iteration (EXP-07-PARETO-CALIB):**
  Instead of an extreme threshold of 0.75, explore a Pareto-calibrated operating policy that balances FWR and minority F1, testing softer margin thresholds ($\theta_{\text{subs}} \in [0.45, 0.55]$, $\theta_{\text{omis}} \in [0.40, 0.50]$) to achieve non-zero F1 across ALL classes while keeping FWR reasonably bounded.
