# Experiment Review: EXP-04-LPDN

## 1. Objective and Setup
Second candidate method family: a learned Residual Multi-Layer Perceptron (LPDN) trained on 59,562 phoneme tokens and 61,362 gap tokens from `TRAIN`.
Evaluated on `DEV_SEARCH` (4 speakers: YBAA, NCC, RRBI, YDCK; 599 utterances).

## 2. Quantitative Breakthroughs
- **Massive Gains Across All Error AP Metrics:**
  - `CORRECT` AP reached **0.9730** (highest in the campaign).
  - `SUBSTITUTION` AP reached **0.3418** (vs 0.1828 in EXP-03, +87% relative gain; vs 0.0895 in EXP-01, +282% relative gain).
  - `OMISSION` AP reached **0.1197** (vs 0.0427 in EXP-03, +180% relative gain; vs 0.0106 in EXP-01, +1,032% relative gain).
  - `INSERTION` AP reached **0.0352** (vs 0.0144 in EXP-03, +144% relative gain).
- **High Sensitivity & Recall:**
  - Omission recall reached **76.56%** (vs 0.96% in EXP-03).
  - Substitution recall reached **70.08%** (vs 25.17% in EXP-03).
  - Macro recall reached **72.02%**!

## 3. Critical Failure Diagnostics & Root Cause
- **Severe Precision Penalty & False Alarm Explosion:**
  Because the training loss heavily penalized missed errors ($\alpha = [0.2, 1.0, 3.0]$), the model over-flagged minor accent deviations as substitutions (3,962 false substitutions) and omissions (1,704 false omissions).
  False Warning Rate spiked to **9.54 false warnings per utterance**, violating the guardrail ($\le 0.20$) and collapsing the coverage/risk utility score to 0.0.
- **Scientific Conclusion:**
  The learned representation space contains strong discriminative signal (proven by AP surging across all 4 classes simultaneously), but the decision threshold policy is currently operating in an over-sensitive regime.
- **Direct Next Action (EXP-05-CALIB):**
  Implement Conservative Confidence & Margin-Gated Decision Policy on `DEV_CAL` that enforces the FWR $\le 0.20$ constraint, balancing high precision with strong recall to unlock the full utility score.
