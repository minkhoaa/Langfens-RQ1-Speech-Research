# Falsification Criteria: Hypothesis H23

## 1. Primary Falsification Gates
Hypothesis H23 will be considered **FALSIFIED** if on the frozen `DEV_SEARCH` cohort:
1. **Omission AP Regression:** Standalone Omission AP drops below 0.1600 (indicating prosodic energy added noise rather than discriminative power).
2. **False Warning Explosion:** False Warning Rate increases to $\ge 2.00$ false alarms/utterance.
3. **Utility Degradation:** Overall Utility falls below 0.3500.
4. **Substitution Degradation:** Standalone Substitution AP drops below 0.3300.

## 2. Inconclusive Conditions
1. Omission metrics improve but Substitution metrics degrade symmetrically, indicating zero net Pareto expansion.
2. Computational extraction overhead exceeds 50ms/utterance, violating real-time production feasibility.
