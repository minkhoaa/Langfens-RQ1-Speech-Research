# Falsification Criteria: Hypothesis H84

Hypothesis H84 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating PCR-Engine on held-out `DEV_REPLICATION`:

1. **Parity Collapse:**
   - Weighted Utility falls below 0.5000 (losing Viginti-Pareto's 0.5079 advantage).
   - Macro F1 drops below 0.6150.
2. **Insufficient Exit Efficiency:**
   - Early exit rate at Tier 1 falls below 60.0% (failing to deliver significant computational savings).
3. **Accuracy Degradation:**
   - Substitution AP falls below 0.4550 or Omission AP falls below 0.4650.
4. **False Alarm Spike:**
   - False Warning Rate (FWR) exceeds 1.85 false alarms per utterance.
