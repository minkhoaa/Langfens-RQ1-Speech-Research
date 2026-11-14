# Falsification Criteria: Hypothesis H29

## 1. Primary Falsification Gates
Hypothesis H29 will be considered **FALSIFIED** if on `DEV_SEARCH`:
1. **Omission AP Degradation:** Standalone Omission AP drops below 0.1600 (indicating phonotactic priors suppressed genuine omissions).
2. **False Warning Escalation:** False Warning Rate increases to $\ge 2.20$ false alarms/utterance.
3. **Utility Degradation:** Overall Utility falls below 0.3600.

## 2. Inconclusive Conditions
1. Precision increases while Recall decreases symmetrically, producing zero net F1 gain across all classes.
