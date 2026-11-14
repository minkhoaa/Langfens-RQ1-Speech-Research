# Falsification Criteria: Hypothesis H53

Hypothesis H53 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Duodecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Duodecim-Pareto Macro AP fails to exceed 0.6200 (inferior to Undecim-Pareto 0.6244).
   - Weighted Utility fails to exceed 0.4850.
2. **False Alarm Runaway:**
   - False Warning Rate (FWR) exceeds 1.85 false alarms per utterance.
3. **Precision Regression:**
   - Substitution Precision falls below 51.00%.
   - Macro Precision falls below 62.00%.
4. **Omission Sensitivity Loss:**
   - Omission AP falls below 0.4350.
