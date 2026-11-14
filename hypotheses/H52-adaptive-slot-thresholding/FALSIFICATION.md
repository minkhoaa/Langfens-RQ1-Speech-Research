# Falsification Criteria: Hypothesis H52

Hypothesis H52 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating AMT-Engine on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Failure to Outperform Static Baseline:**
   - Weighted Utility falls below 0.4850 on `DEV_REPLICATION` (failing to exceed static DSCA-Net 0.4928 or Undecim-Pareto 0.4792).
   - Macro AP falls below 0.6150.
2. **Threshold Over-Fitting on DEV_CAL:**
   - Improvement on `DEV_CAL` fails to generalize to `DEV_REPLICATION`, resulting in a utility drop $\Delta \text{Utility} < 0$.
3. **Manner Bias Distortion:**
   - Omission AP falls below 0.4200, indicating that duration adjustments over-suppressed genuine plosive omissions.
4. **False Alarm Runaway:**
   - False Warning Rate (FWR) exceeds 2.00 false alarms per utterance.
