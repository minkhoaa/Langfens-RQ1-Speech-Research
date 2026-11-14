# Falsification Criteria: Hypothesis H27

## 1. Primary Falsification Gates
Hypothesis H27 will be considered **FALSIFIED** if:
1. **Catastrophic Failure under Mild Noise:** At $+20\text{ dB}$ SNR, Hexad-Pareto Utility drops by $> 30\%$ (falling below 0.3000).
2. **Zero Ensemble Advantage under Noise:** At $0\text{ dB}$ SNR, Hexad-Pareto performs equal to or worse than a single standalone baseline.
3. **Severe False Warning Explosion:** Under $+10\text{ dB}$ noise, False Warning Rate explodes to $\ge 5.0$ false alarms/utterance.

## 2. Inconclusive Conditions
1. Noise corrupts only insertion detection while leaving target substitution and omission unaffected due to gap duration masking.
