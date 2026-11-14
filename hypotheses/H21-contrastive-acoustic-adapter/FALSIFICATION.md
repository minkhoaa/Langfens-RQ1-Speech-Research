# Falsification Criteria: Hypothesis H21

## 1. Primary Falsification Gates
Hypothesis H21 will be considered **FALSIFIED** if any of the following occur on the frozen `DEV_SEARCH` cohort:
1. **Degraded Substitution AP:** Standalone Substitution AP drops below 0.3300 (worse than unadapted baseline).
2. **Degraded Omission AP:** Standalone Omission AP drops below 0.1700 (indicating contrastive margin destroyed low-energy deletion cues).
3. **Utility Regression:** Overall Utility on `DEV_SEARCH` falls below 0.3500.
4. **Optimization Instability:** Loss fails to converge or exhibits gradient explosion during joint contrastive-adapter backpropagation.

## 2. Inconclusive Conditions
1. Substitution AP improves while Omission AP degrades significantly ($> 15\%$ drop), indicating a destructive Pareto trade-off between prototype margin and deletion detection.
2. Improvements are restricted to high False Warning Rates ($\text{FWR} > 2.50$ false alarms/utterance).
