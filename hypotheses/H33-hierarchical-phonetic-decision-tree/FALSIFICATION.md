# Falsification Criteria: Hypothesis H33

## 1. Primary Falsification Gates
Hypothesis H33 will be considered **FALSIFIED** if on the frozen `DEV_SEARCH` cohort:
1. **Utility Regression:** Standalone HTC-Net Utility falls below 0.3600 (worse than APED-Net / PNC-Net).
2. **Omission AP Collapse:** Standalone Omission AP drops below 0.1700.
3. **Severe False Warning Inflation:** False Warning Rate increases to $\ge 2.20$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Vowel substitution accuracy increases while consonant substitution accuracy drops by a reciprocal amount, yielding no net gain in Macro AP.
