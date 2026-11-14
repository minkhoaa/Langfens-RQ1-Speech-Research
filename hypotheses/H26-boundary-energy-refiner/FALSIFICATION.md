# Falsification Criteria: Hypothesis H26

## 1. Primary Falsification Gates
Hypothesis H26 will be considered **FALSIFIED** if on the frozen `DEV_SEARCH` cohort:
1. **Omission AP Degradation:** Standalone Omission AP drops below 0.1700.
2. **False Warning Escalation:** False Warning Rate exceeds 2.50 false alarms/utterance.
3. **Utility Regression:** Overall Utility falls below 0.3600.

## 2. Inconclusive Conditions
1. Stop consonant detection improves while vowel/fricative accuracy degrades by an equal amount, indicating a feature trade-off rather than net diagnostic improvement.
