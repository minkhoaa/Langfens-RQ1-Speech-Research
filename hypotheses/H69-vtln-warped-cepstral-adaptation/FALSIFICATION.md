# Falsification Criteria: Hypothesis H69

Hypothesis H69 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating V-CASR-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Standalone Benchmark:**
   - V-CASR-Net Macro AP fails to exceed 0.6180 (inferior to IPTM-Net 0.6186).
   - Weighted Utility falls below 0.4800.
2. **False Warning Persistence:**
   - False Warning Rate (FWR) remains above 2.20 false alarms per utterance (failing to eliminate vocal tract false alarms).
   - Macro Precision drops below 60.00%.
3. **Recall Collapse:**
   - Macro Recall falls below 56.00% or Substitution Recall falls below 40.00%.
4. **Calibration Degradation:**
   - Overall ECE exceeds 0.0200 (2.00%).
