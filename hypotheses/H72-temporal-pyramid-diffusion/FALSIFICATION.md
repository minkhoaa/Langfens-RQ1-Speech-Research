# Falsification Criteria: Hypothesis H72

Hypothesis H72 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating TPD-Net on held-out `DEV_REPLICATION`:

1. **Calibration Regression:**
   - Overall ECE fails to stay below 0.0120 (inferior to MRTP-Net 0.0083).
2. **False Omission Explosion:**
   - False omissions exceed 130 across 9,863 target phones (failing to preserve MRTP-Net's false omission suppression).
3. **Substitution Regression:**
   - Substitution AP falls below 0.4400 or Substitution Precision falls below 52.00%.
4. **Utility Failure:**
   - Weighted Utility falls below 0.4750.
