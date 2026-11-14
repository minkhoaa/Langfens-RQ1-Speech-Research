# Falsification Criteria for Hypothesis H07

1. **Ensemble Superiority Falsification:**
   If the ensemble fails to achieve a Weighted Utility $> 0.3610$ (higher than both standalone BiLSTM at 0.3610 and Conformer at 0.3546) on `DEV_SEARCH`, H07 is rejected.

2. **Orthogonal Information Test:**
   If the optimal mixture weight $w$ collapses to 0.0 or 1.0 on `DEV_CAL`, the two model families lack complementary information, falsifying the complementarity hypothesis.
