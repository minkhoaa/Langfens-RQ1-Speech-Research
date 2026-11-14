# Falsification Criteria: Hypothesis H81

Hypothesis H81 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Viginti-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Viginti-Pareto Macro AP fails to exceed 0.6375 (inferior to Undeviginti-Pareto 0.6372).
   - Weighted Utility falls below 0.5000 (inferior to Duodeviginti-Pareto 0.5005 and Undeviginti-Pareto 0.5021).
2. **False Alarm Contamination:**
   - False Warning Rate (FWR) exceeds 1.70 false alarms per utterance.
   - Macro Precision drops below 63.80%.
3. **Class Degradation:**
   - Omission AP falls below 0.4720.
   - Substitution AP falls below 0.4580.
   - Insertion AP falls below 0.3180.
