# Falsification Criteria for Hypothesis H17

1. **Substitution Precision Falsification:**
   If PSC-Net fails to achieve higher Substitution Precision ($> 51.39\%$) or higher Substitution F1 ($> 0.3763$) on `DEV_REPLICATION` than the previous models, H17 is falsified.

2. **Cluster Distortion:**
   If the contrastive margin causes cluster distortion that degrades `CORRECT` F1 below 0.90, the contrastive margin is rejected as over-constraining.
