# Falsification Criteria: Hypothesis H60

Hypothesis H60 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating P-ALDS-Net on held-out `DEV_REPLICATION`:

1. **Failure to Outperform Baseline Models:**
   - Standalone Weighted Utility falls below 0.4850 (failing to exceed DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6100.
2. **Persistent Substitution Over-Denoising:**
   - Substitution AP fails to exceed 0.4200 (failing to resolve the limitation identified in EXP-59).
3. **Insertion F1 Failure:**
   - Insertion F1 falls below 0.3800.
   - Insertion Precision drops below 45.00%.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.10 false alarms per utterance.
5. **Runtime Latency Regression:**
   - CPU single-utterance inference latency exceeds 25.0 ms.
