# Falsification Criteria: Hypothesis H31

## 1. Primary Falsification Gates
Hypothesis H31 will be considered **FALSIFIED** if on the frozen `DEV_REPLICATION` cohort:
1. **Utility Regression:** Septad-Pareto Utility falls below 0.4280 (inferior to Hexad-Pareto).
2. **Macro F1 Regression:** Macro F1 falls below 0.5700.
3. **Precision Collapse:** Macro Precision drops below 0.6000.
4. **False Warning Escalation:** False Warning Rate increases to $\ge 2.20$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Utility improvement is strictly marginal ($< 0.0005$) and does not exceed bootstrap variance ($p \ge 0.05$).
