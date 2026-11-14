# Falsification Criteria: Hypothesis H77

Hypothesis H77 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating IP-ALDS-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Standalone Macro F1 Benchmark:**
   - IP-ALDS-Net Macro F1 fails to exceed 0.6180 (inferior to IPTM-Net 0.6205).
2. **Failure to Beat Standalone Macro AP Benchmark:**
   - Standalone Macro AP falls below 0.6150 (inferior to IPTM-Net 0.6186 and HRTD-Net 0.6202).
3. **Precision / Utility Degradation:**
   - Substitution Precision falls below 53.00% or Weighted Utility falls below 0.4850.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.75 false alarms per utterance.
