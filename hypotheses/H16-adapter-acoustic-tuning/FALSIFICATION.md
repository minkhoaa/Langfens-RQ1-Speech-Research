# Falsification Criteria for Hypothesis H16

1. **Acoustic Adaptation Superiority:**
   If the NNA-Adapter fails to achieve higher Average Precision on `SUBSTITUTION` ($> 0.372$) and `OMISSION` ($> 0.201$) than the unadapted HM-ADN baseline on `DEV_SEARCH`, H16 is falsified.

2. **Accent Over-fitting:**
   If adaptation on the 12 training speakers improves `TRAIN` loss but degrades cross-accent invariance on held-out L1 groups (increasing L1 variance $\sigma_{\text{L1}} > 0.08$), the adaptation is rejected as accent-overfitted.
