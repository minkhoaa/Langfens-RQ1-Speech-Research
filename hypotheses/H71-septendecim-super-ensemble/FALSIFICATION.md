# Falsification Criteria: Hypothesis H71

Hypothesis H71 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Septendecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Septendecim-Pareto Macro AP fails to exceed 0.6385 (inferior to Quindecim-Pareto 0.6390).
   - Weighted Utility falls below 0.4950.
2. **False Alarm Re-Emergence:**
   - False Warning Rate (FWR) exceeds 1.65 false alarms per utterance.
   - Macro Precision drops below 63.80%.
3. **Class Degradation:**
   - Omission AP falls below 0.4720.
   - Substitution AP falls below 0.4580.
   - Insertion AP falls below 0.3150.
