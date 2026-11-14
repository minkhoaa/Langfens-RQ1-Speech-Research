# Falsification Criteria: Hypothesis H24

## 1. Primary Falsification Gates
Hypothesis H24 will be considered **FALSIFIED** if on the frozen `DEV_REPLICATION` cohort:
1. **Macro AP Regression:** Hexad-Pareto Macro AP falls below 0.5700 (worse than Pentad-Pareto's 0.5760).
2. **Macro F1 Regression:** Macro F1 falls below 0.5500.
3. **Precision Collapse:** Macro Precision drops below 0.6000 (failing the 60% macro milestone).
4. **False Warning Explosion:** False Warning Rate increases to $\ge 1.80$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Macro AP improves slightly ($< 0.0020$) while False Warning Rate increases significantly ($> 0.30$ false alarms/utt).
2. Improvements are confined to a single class while another core class suffers a $> 10\%$ drop in F1.
