# Falsification Criteria: Hypothesis H70

Hypothesis H70 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating DP-ALDS-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Substitution Precision Benchmark:**
   - Substitution Precision fails to exceed 55.00% (inferior to P-ALDS-Net 55.83%).
2. **Failure to Beat Substitution AP Benchmark:**
   - Substitution AP falls below 0.4450 (inferior to P-ALDS-Net 0.4485).
3. **Macro Regression:**
   - Macro F1 falls below 0.6000 or Weighted Utility falls below 0.4800.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.70 false alarms per utterance.
