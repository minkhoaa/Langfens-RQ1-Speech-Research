# Falsification Criteria: Hypothesis H22

## 1. Primary Falsification Gates
Hypothesis H22 will be considered **FALSIFIED** if:
1. **Utility Regression:** Pentad-Pareto Utility on `DEV_REPLICATION` drops below $0.4250$ (worse than the Quad-Model baseline of $0.4310$).
2. **Omission Collapse:** Omission F1 on `DEV_REPLICATION` drops below $0.3800$ (indicating JSAC-Net probabilities diluting rather than enhancing the ensemble).
3. **Substitution Degradation:** Substitution AP drops below $0.3800$.
4. **False Warning Explosion:** False Warning Rate increases to $\ge 2.00$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Gains occur only by excessively loosening thresholds such that False Warning Rate exceeds $1.80$.
2. Improvement is restricted to marginal increments ($< 0.0010$ Utility delta) within random seed noise variance.
