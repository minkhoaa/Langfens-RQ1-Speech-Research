# Falsification Criteria: Hypothesis H79

Hypothesis H79 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Undeviginti-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Undeviginti-Pareto Macro AP fails to exceed 0.6385 (inferior to Quindecim-Pareto 0.6390).
   - Weighted Utility falls below 0.4980 (inferior to Duodeviginti-Pareto 0.5005).
2. **False Alarm Contamination:**
   - False Warning Rate (FWR) exceeds 1.70 false alarms per utterance.
   - Macro Precision drops below 63.80%.
3. **Class Degradation:**
   - Omission AP falls below 0.4730.
   - Substitution AP falls below 0.4580.
   - Insertion AP falls below 0.3180.
