# Falsification Criteria for Hypothesis H13

1. **Seed Instability Falsification:**
   If the standard deviation of Macro F1 across the 5 seeds exceeds 0.020 ($\sigma_{\text{seed}} > 0.020$), H13 is falsified, indicating high sensitivity to random initialization.

2. **Milestone Reversal:**
   If any individual seed causes Macro Precision to drop below 0.55 on `DEV_REPLICATION`, the claim of a stable $>60\%$ milestone is falsified.
