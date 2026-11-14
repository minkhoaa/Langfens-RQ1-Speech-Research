# Falsification Criteria: Hypothesis H50

Hypothesis H50 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating DSCA-Net on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Failure to Outperform HRTD-Net Baseline:**
   - Standalone Macro AP falls below 0.6150 (inferior to baseline HRTD-Net 0.6202).
   - Weighted Utility falls below 0.4700.
2. **Cross-Attention Disconnect / Regression:**
   - Substitution AP falls below 0.4300, indicating that cross-attention failed to sharpen substitution detection.
   - Omission AP falls below 0.4150.
3. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.20 false alarms per utterance.
4. **Computational / Memory Collapse:**
   - Bidirectional cross-attention causes out-of-memory errors or increases inference latency by more than $5\times$ relative to HRTD-Net.
