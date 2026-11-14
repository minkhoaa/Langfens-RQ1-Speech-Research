# Falsification Criteria for Hypothesis H04

1. **Context Superiority Falsification:**
   If BiLSTM-CPN fails to achieve higher Average Precision on `SUBSTITUTION` ($> 0.364$) and `OMISSION` ($> 0.135$) than the non-sequential MLP baseline (EXP-06 / EXP-07), H04 is falsified.

2. **Sequential Overfitting Falsification:**
   If training sequence loss decreases but validation loss or Macro AP on `DEV_SEARCH` drops below 0.45 due to sequential prompt memorization, the model is rejected as overfitted to training prompt sequences.

3. **Insertion Gain Falsification:**
   If contextual modeling does not improve `INSERTION` F1 over 0.05 on `DEV_SEARCH`, the contextual hypothesis for gap epenthesis is rejected.
