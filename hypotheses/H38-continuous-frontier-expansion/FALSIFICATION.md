# Falsification Criteria: Hypothesis H38

## 1. Primary Falsification Gates
Hypothesis H38 will be considered **FALSIFIED** if on the frozen `DEV_REPLICATION` cohort:
1. **Macro AP Regression:** Nonad-Pareto Macro AP falls below 0.5800 (worse than Octad-Pareto's 0.5942).
2. **Precision Collapse:** Macro Precision drops below 0.6000 (failing the 60% milestone).
3. **Utility Regression:** Weighted Utility falls below 0.4300.
4. **False Warning Escalation:** False Warning Rate increases to $\ge 1.80$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Macro AP improves by $< 0.0010$ and difference from Octad-Pareto is statistically non-significant at $\alpha = 0.05$.
