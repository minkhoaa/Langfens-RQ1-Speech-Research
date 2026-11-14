# Falsification Criteria for Hypothesis H06

1. **Conformer Superiority Falsification:**
   If the Conformer architecture fails to achieve a higher Weighted Utility than EXP-08 ($> 0.3610$) on `DEV_SEARCH`, H06 is falsified.

2. **Attention Degradation:**
   If self-attention causes attention dispersion on short utterances resulting in lower Macro F1 ($< 0.50$), H06 is rejected.
