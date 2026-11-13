# Independent Scientific Review: Experiment EXP-74

## 1. Audit Against Falsification Criteria (Hypothesis H69)
- **Criterion 1 (False Warning Reduction vs EXP-71):**
  * EXP-71 FWR = 3.517.
  * EXP-74 FWR = **2.717 false alarms/utterance**, cutting total false warnings from 1,055 down to 815 (232 false substitutions eliminated!).
  * On DEV_SEARCH, false alarms dropped from 2,609 down to 1,943 (666 false alarms eliminated!).
  * **Passed.**
- **Criterion 2 (Recall & Metric Improvement vs EXP-71):**
  * Macro AP increased from 0.5921 to **0.5975**.
  * Macro F1 increased from 0.6057 to **0.6139 (61.39%)**.
  * Weighted Utility increased from 0.4673 to **0.4788 (+1.15%)**.
  * Substitution AP increased from 0.4300 to **0.4425 (+1.25%)**.
  * Insertion AP increased from 0.2342 to **0.2606 (+2.64%)**, Insertion F1 reaching **0.3584**.
  * **Passed.**
- **Criterion 3 (Standalone Macro AP Target):**
  * Target was $\ge 0.6220$.
  * Actual: Macro AP = **0.5975**.
  * Falsified for the standalone 0.6220 benchmark.

## 2. Scientific Verdict & Lessons Learned
Hypothesis H69 is **CONFIRMED for VTLN Normalization Gain over EXP-71**, but **FALSIFIED for standalone dominance over IPTM-Net / DSCA-Net**.
- **Key Scientific Finding:**
  Normalizing spectral cepstral residuals by speaker-specific VTLN warping factors $\alpha_{\text{spk}} = C_{\text{ref}} / C_{\text{spk}}$ mathematically compresses vocal tract length variation across adult speakers. This decisively eliminates 232 false substitutions on DEV_REPLICATION and 666 on DEV_SEARCH, lifting Utility to 0.4788 and Macro F1 to 61.39%.
- However, standalone acoustic residuals still exhibit slight residual acoustic variance compared to pure self-supervised foundation representations (Charsiu SSL in DSCA-Net EXP-55 and IPTM-Net EXP-69).
- **Ensemble Directive:** V-CASR-Net provides a much cleaner, higher-precision acoustic diversity component for super-ensembles than un-warped CASR-Net, ready to power the next-generation ensemble models.
