# Falsification Criteria for Hypothesis H01

1. **Prior Superiority Falsification:**
   If the acoustic verifier fails to outperform the prior class-frequency predictor on `DEV_SEARCH` in either Weighted Utility or class-specific AP, H01 is falsified.

2. **Leakage Control Check:**
   In `EXP-02-MISMATCH`, if the evaluation metrics on mismatched audio do not collapse to random/chance level (Utility $< 0.10$, AP near class prevalence), the pipeline suffers from oracle leakage, immediately invalidating the experiment.

3. **Omission & Insertion Asymmetry:**
   If standard frame-averaged GOP fails to separate short but realized sounds from genuine deletions (yielding OMISSION AP $< 0.30$), standard GOP alone is insufficient and requires explicit sequence delta or duration modeling.
