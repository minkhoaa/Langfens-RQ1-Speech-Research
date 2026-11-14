# Falsification Criteria for Hypothesis H02

1. **Utility Non-Inferiority:**
   If the learned diagnostic network fails to achieve a Weighted Utility $> 0.35$ (higher than both EXP-01-PRIOR at 0.3183 and EXP-03-SSL-GOP at 0.2522) on `DEV_SEARCH`, H02 is rejected.

2. **False Warning Reduction:**
   If the false warning rate per utterance remains above 1.50 on `DEV_SEARCH`, H02's claim of false alarm suppression is falsified.

3. **Multi-Class Recall:**
   If the learned model collapses to predicting only `CORRECT` (achieving 0 recall on omissions or insertions), H02 is rejected.
