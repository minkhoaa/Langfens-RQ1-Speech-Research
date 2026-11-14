# Falsification Criteria: Hypothesis H68

Hypothesis H68 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Sedecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Sedecim-Pareto Macro AP fails to exceed 0.6380 (inferior to Quindecim-Pareto 0.6390).
   - Weighted Utility falls below 0.4930.
2. **False Alarm Contamination:**
   - Fusing CASR-Net inflates False Warning Rate (FWR) above 1.75 false alarms per utterance.
   - Macro Precision drops below 63.50%.
3. **Class Degradation:**
   - Omission AP falls below 0.4700.
   - Substitution AP falls below 0.4550.
   - Insertion AP falls below 0.3100.
