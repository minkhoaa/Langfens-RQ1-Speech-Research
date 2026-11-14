# Falsification Criteria: Hypothesis H56

Hypothesis H56 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating UCR-Net on held-out `DEV_REPLICATION`:

1. **Failure to Outperform DSCA-Net Baseline:**
   - Standalone Weighted Utility falls below 0.4850 (inferior to DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6120.
2. **Substitution Degradation:**
   - Standalone Substitution AP fails to exceed 0.4350.
3. **Omission Sensitivity Loss:**
   - Omission AP falls below 0.4200, indicating that cepstral residuals distracted the network from transient deletion cues.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 2.10 false alarms per utterance.
5. **Data Leakage Violation:**
   - Canonical centroids $\bar{C}_{\text{canonical}}$ computed from non-training splits or human realization labels (must strictly derive from `TRAIN` partition using canonical prompt phones).
