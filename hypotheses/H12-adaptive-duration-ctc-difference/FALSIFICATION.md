# Falsification Criteria for Hypothesis H12

1. **Duration Normalization Falsification:**
   If SRAD-Net fails to achieve higher Omission F1 or higher Weighted Utility on `DEV_REPLICATION` than the unnormalized model (EXP-12 at 0.4231), H12 is rejected.

2. **Degradation of Substitution:**
   If duration normalization reduces Substitution Precision below 45% or Substitution F1 below 0.33, the feature augmentation is rejected as introducing harmful feature collinearity.
