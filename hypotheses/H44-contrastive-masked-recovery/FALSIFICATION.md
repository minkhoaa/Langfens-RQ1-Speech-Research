# Falsification Criteria: Hypothesis H44

Hypothesis H44 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating CDMR-Net on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Contrastive Gradient Conflict / Regression:**
   - Macro AP falls below 0.6050 (worse than HRTD-Net 0.6202).
   - Weighted Utility falls below 0.4600.
2. **Failure to Break Substitution Target:**
   - Substitution AP fails to exceed 0.4300 on `DEV_REPLICATION`.
3. **Omission Degradation:**
   - Omission AP falls below 0.4100 on `DEV_REPLICATION`.
4. **False Alarm Runaway:**
   - False Warning Rate (FWR) exceeds 2.50 false alarms per utterance.
5. **Numerical Instability:**
   - Normalized cosine similarity scores collapse to NaN or collapse into a single degenerate prototype cluster.
