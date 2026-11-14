# Falsification Criteria: Hypothesis H36

## 1. Primary Falsification Gates
Hypothesis H36 will be considered **FALSIFIED** if on the SpeechOcean762 test benchmark:
1. **Correlation Regression:** Pearson correlation drops below $r = 0.2500$ (inferior to early baseline EXP-09).
2. **Error Detection Failure:** Error AP drops below $0.1100$.
3. **Loss of Significance:** Correlation p-value fails to reach statistical significance ($\alpha = 0.05$).

## 2. Inconclusive Conditions
1. Pearson correlation increases while Spearman rank correlation decreases, indicating non-monotonic distortion at score extremes.
