# Falsification Criteria: Hypothesis H57

Hypothesis H57 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating DAMH-Net on `SpeechOcean762-Test` or `L2-ARCTIC DEV_REPLICATION`:

1. **Cross-Corpus Transfer Stagnation:**
   - Zero-shot Pearson correlation on SpeechOcean762 fails to exceed 0.2500 ($r < 0.2500$).
   - Spearman rank correlation falls below 0.2800.
2. **In-Domain L2-ARCTIC Regression:**
   - Macro AP on `DEV_REPLICATION` falls below 0.6050 (worse than baseline DSCA-Net 0.6159).
   - Weighted Utility falls below 0.4750.
3. **Severe Omission or Substitution Loss:**
   - Omission AP falls below 0.4200.
   - Substitution AP falls below 0.4100.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.10 false alarms per utterance on L2-ARCTIC.
