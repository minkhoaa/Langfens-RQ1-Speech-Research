# Falsification Criteria: Hypothesis H83

Hypothesis H83 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating LCC-Engine on held-out `DEV_REPLICATION`:

1. **Calibration Regression:**
   - LCC-Engine Macro AP fails to exceed 0.6360 (inferior to uncalibrated Viginti-Pareto 0.6357).
   - Weighted Utility falls below 0.5050 (inferior to Viginti-Pareto 0.5079).
2. **False Alarm Persistence:**
   - False Warning Rate (FWR) fails to improve below 1.60 false alarms per utterance.
   - Macro Precision fails to exceed 64.50%.
3. **Recall Collapse:**
   - Omission Recall falls below 44.00% or Substitution Recall falls below 38.00%.
