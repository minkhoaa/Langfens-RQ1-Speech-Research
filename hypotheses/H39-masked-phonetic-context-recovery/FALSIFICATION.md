# Falsification Criteria: Hypothesis H39

## 1. Primary Falsification Gates
Hypothesis H39 will be considered **FALSIFIED** if on the frozen `DEV_SEARCH` cohort:
1. **Omission AP Regression:** Standalone Omission AP drops below 0.2000 (failing to preserve HTC-Net's baseline of 0.2748).
2. **Reconstruction Divergence:** Auxiliary reconstruction loss fails to converge or destabilizes diagnostic focal loss.
3. **Severe False Warning Inflation:** False Warning Rate increases to $\ge 2.20$ false alarms/utterance.
4. **Utility Regression:** Overall Utility falls below 0.3600.

## 2. Inconclusive Conditions
1. Masking improves omission recall but degrades substitution precision by more than 10% relative, indicating an imbalanced trade-off.
