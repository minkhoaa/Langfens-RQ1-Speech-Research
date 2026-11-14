# Hypothesis H30: Cross-Accent Invariance Across 6 L1 Native Language Cohorts

## 1. Context and Problem Statement
A critical failure mode of computer-assisted pronunciation training (CAPT) models is accent bias: models often overfit to specific phonetic substitution patterns of one dominant non-native language (e.g. Mandarin L1) while failing catastrophically on another (e.g. Arabic or Vietnamese L1).
In EXP-14, an early 3-model ensemble achieved $\sigma_{\text{L1}} = 0.0716 \le 0.08$.
Now, with the development of the 6-model Hexad-Pareto (EXP-29, Utility=0.4321) and the phonotactically conditioned PNC-Net (EXP-34, Insertion F1=0.3692), it is essential to determine whether these advanced architectures maintain accent-invariant performance across all 6 L1 cohorts represented in L2-ARCTIC:
1. Arabic
2. Mandarin
3. Spanish
4. Hindi
5. Korean
6. Vietnamese

## 2. Formal Hypothesis
Evaluating the frozen Hexad-Pareto and PNC-Net models across the 8 held-out speakers of the unsealed evaluation sets (1,199 utterances across all 6 L1 cohorts) will demonstrate:
1. **Accent Invariance:** Cross-accent variation in Macro F1 across all 6 L1 cohorts will remain strictly bounded:
   $$\sigma_{\text{L1}}(\text{Macro F1}) \le 0.0600$$
   demonstrating higher cross-accent consistency than early baselines.
2. **Universal Milestone 60 for Macro Precision:** Every individual L1 cohort will achieve $\text{Macro Precision} \ge 0.5800$, with at least 4 of the 6 cohorts exceeding the $0.6000$ milestone.
3. **High Vietnamese Omission Sensitivity:** Vietnamese speakers will maintain high Omission F1 ($\ge 0.5500$) and high Omission AP ($\ge 0.3500$) due to physical prosodic energy resolution of final consonant deletions.

## 3. Predicted Evidence & Significance
If validated, this proves that combining multi-task articulatory decomposition, physical waveform energy dynamics, and lexical phonotactic priors produces an accent-agnostic diagnostic engine robust to diverse mother-tongue phonetic transfer phenomena.
