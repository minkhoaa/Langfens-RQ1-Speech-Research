# Falsification Criteria: Hypothesis H78

Hypothesis H78 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating CCSP-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Insertion F1 Benchmark:**
   - CCSP-Net Insertion F1 fails to exceed 0.4200 (inferior to TDIS-Net 0.4321).
2. **Failure to Beat Insertion AP Benchmark:**
   - Insertion AP falls below 0.3100 (inferior to TDIS-Net 0.3151).
3. **Precision / Recall Imbalance:**
   - Insertion Precision falls below 48.00% or Insertion Recall falls below 32.00%.
4. **General Model Degradation:**
   - Standalone Weighted Utility falls below 0.4750.
