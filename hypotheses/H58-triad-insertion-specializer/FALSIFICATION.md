# Falsification Criteria: Hypothesis H58

Hypothesis H58 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating TDIS-Net on held-out `DEV_REPLICATION`:

1. **Failure to Break Insertion Plateau:**
   - Insertion F1 fails to exceed 0.3800 (inferior to previous peak 0.3959 in EXP-44).
   - Insertion Recall falls below 35.00%.
2. **Precision Regression:**
   - Insertion Precision drops below 30.00% due to over-triggering in breathing pauses.
3. **Macro Quality Degradation:**
   - Standalone Weighted Utility falls below 0.4800 (worse than baseline DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6100.
4. **False Warning Runaway:**
   - False Warning Rate (FWR) exceeds 2.00 false alarms per utterance.
