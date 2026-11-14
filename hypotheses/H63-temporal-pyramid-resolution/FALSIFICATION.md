# Falsification Criteria: Hypothesis H63

Hypothesis H63 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating MRTP-Net on held-out `DEV_REPLICATION`:

1. **Failure to Outperform DSCA-Net Baseline:**
   - Standalone Weighted Utility falls below 0.4880 (inferior to baseline DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6120.
2. **Pyramid Feature Redundancy / Regression:**
   - Standalone Omission AP falls below 0.4250, indicating that multi-scale features added noise rather than resolving bursts.
   - Standalone Substitution AP falls below 0.4150.
3. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.10 false alarms per utterance.
4. **Latency Regression:**
   - Single-utterance CPU inference latency exceeds 15.0 ms.
