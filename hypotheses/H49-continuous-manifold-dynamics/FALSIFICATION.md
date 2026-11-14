# Falsification Criteria: Hypothesis H49

Hypothesis H49 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating CLMD-Net on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Failure to Outperform HRTD-Net:**
   - Standalone Macro AP falls below 0.6150 (inferior to HRTD-Net 0.6202).
   - Weighted Utility falls below 0.4750.
2. **Manifold Over-Smoothing:**
   - Omission AP falls below 0.4200, indicating that temporal smoothness regularizer blurred genuine missing plosive boundaries.
3. **Substitution Degradation:**
   - Substitution AP falls below 0.4350.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.20 false alarms per utterance.
