# Falsification Criteria: Hypothesis H48

Hypothesis H48 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating DIAA-Net on `DEV_SEARCH` and `DEV_REPLICATION`:

1. **Failure to Reduce Cross-L1 Variance:**
   - Standard deviation across the 6 L1 cohorts exceeds 0.0750 ($\sigma_{\text{L1}} > 0.0750$), indicating that FiLM conditioning failed to normalize accent variability.
2. **Macro Quality Regression:**
   - Standalone Macro AP falls below 0.6050 (worse than baseline HRTD-Net 0.6202).
   - Weighted Utility falls below 0.4600.
3. **Severe Omission Degradation:**
   - Omission AP falls below 0.4100 on `DEV_REPLICATION` or below 0.2800 on `DEV_SEARCH`.
4. **False Warning Runaway:**
   - False Warning Rate (FWR) exceeds 2.20 false alarms per utterance.
