# Falsification Criteria: Hypothesis H40

## 1. Primary Falsification Gates
Hypothesis H40 will be considered **FALSIFIED** if on the frozen `DEV_REPLICATION` cohort:
1. **Macro AP Regression:** Decimal-Pareto Macro AP falls below 0.5900 (failing to exceed DMCT-Net's 0.6062).
2. **Utility Regression:** Weighted Utility falls below 0.4400.
3. **Precision Collapse:** Macro Precision drops below 0.6000.
4. **False Warning Escalation:** False Warning Rate increases to $\ge 1.60$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Macro AP improves by $< 0.0020$ while Macro F1 degrades, indicating an ambiguous multi-objective trade-off.
