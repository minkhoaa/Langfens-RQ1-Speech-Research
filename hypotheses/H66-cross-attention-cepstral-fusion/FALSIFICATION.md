# Falsification Criteria: Hypothesis H66

Hypothesis H66 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating CASR-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Standalone Macro AP Benchmark:**
   - CASR-Net Macro AP fails to exceed 0.6200 (inferior to HRTD-Net 0.6202).
2. **Failure to Beat Standalone Utility Benchmark:**
   - Weighted Utility falls below 0.4900 (inferior to DSCA-Net 0.4928).
3. **Cepstral Residual Interference:**
   - Adding 8-dim cepstral residuals degrades Substitution AP below 0.4350 or Substitution Precision below 50.00%.
4. **Calibration Degradation:**
   - Overall ECE exceeds 0.0200 or False Warning Rate (FWR) exceeds 2.00 false alarms per utterance.
