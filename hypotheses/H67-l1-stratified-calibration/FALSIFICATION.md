# Falsification Criteria: Hypothesis H67

Hypothesis H67 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating LDC-Engine on held-out `DEV_REPLICATION`:

1. **Calibration Regression:**
   - Macro AP fails to exceed 0.6380 (inferior to uncalibrated Quindecim-Pareto 0.6390).
   - Weighted Utility falls below 0.4950.
2. **Precision Stagnation:**
   - Macro Precision fails to exceed 0.6450.
3. **False Warning Disruption:**
   - False Warning Rate (FWR) fails to improve below 1.40 false alarms per utterance.
4. **Minority Class Collapse:**
   - Omission Recall falls below 40.0% or Insertion Precision falls below 48.0%.
