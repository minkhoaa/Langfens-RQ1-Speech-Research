# Falsification Criteria: Hypothesis H86

Hypothesis H86 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating BAPM-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Standalone Macro F1 Benchmark:**
   - BAPM-Net Macro F1 fails to exceed 0.6190 (inferior to IPTM-Net 0.6205).
2. **Failure to Beat Standalone Macro AP Benchmark:**
   - Standalone Macro AP falls below 0.6160 (inferior to IPTM-Net 0.6186 and HRTD-Net 0.6202).
3. **Precision / Utility Degradation:**
   - Substitution Precision falls below 51.50% or Weighted Utility falls below 0.4850.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.70 false alarms per utterance.
