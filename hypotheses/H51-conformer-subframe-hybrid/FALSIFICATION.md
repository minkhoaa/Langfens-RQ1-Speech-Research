# Falsification Criteria: Hypothesis H51

Hypothesis H51 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating CHTD-Net on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Failure to Outperform DSCA-Net Baseline:**
   - Standalone Weighted Utility falls below 0.4850 (worse than DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6100.
2. **Convolutional Overfitting / Instability:**
   - Depthwise convolution over-smooths acoustic frame boundaries, causing Omission AP to drop below 0.4200.
3. **Substitution Degradation:**
   - Substitution AP falls below 0.4200.
4. **False Warning Explosion:**
   - False Warning Rate (FWR) exceeds 2.20 false alarms per utterance on `DEV_REPLICATION`.
5. **Latency Target Miss:**
   - CPU single-utterance inference latency exceeds 25.0 ms.
