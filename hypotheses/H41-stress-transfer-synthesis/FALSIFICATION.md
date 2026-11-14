# Falsification Criteria: Hypothesis H41

Hypothesis H41 will be declared **FALSIFIED** if any of the following conditions occur when evaluating the champion models (DMCT-Net EXP-44 and Decimal-Pareto EXP-45) under combined stress testing on the held-out `DEV_REPLICATION` cohort:

1. **Catastrophic Precision Collapse:**
   - Either DMCT-Net or Decimal-Pareto drops Macro Precision below $0.5000$ (50%) under moderate perturbation (jitter $\pm 10\%$, tempo dilation $\pm 10\%$).
2. **False Alarm Runaway:**
   - False Warning Rate (FWR) exceeds $2.00$ false alarms per utterance for Decimal-Pareto under perturbation.
3. **Severe Utility Degradation:**
   - Retained Utility drops by more than $25\%$ relative to clean baseline (Utility $< 0.3400$).
4. **Omission Sensitivity Vanishing:**
   - Omission AP drops below $0.2500$ under perturbation, indicating that acoustic dynamics collapse under tempo warping.

If falsified, the mechanism of failure will be isolated to determine whether data augmentation during training (e.g. SpecAugment or continuous tempo perturbation) is strictly necessary to bridge the acoustic degradation gap.
