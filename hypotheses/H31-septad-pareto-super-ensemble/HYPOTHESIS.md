# Hypothesis H31: 7-Model Septad Super-Ensemble (Septad-Pareto)

## 1. Context and Problem Statement
In previous iterations:
- **Hexad-Pareto (EXP-29)** established campaign records for Macro AP (0.5810), Macro F1 (0.5801), Substitution AP (0.4019), and Weighted Utility (0.4321) by fusing six diverse neural sequence and prosodic architectures.
- **PNC-Net (EXP-34)** established campaign records for Insertion F1 (0.3692) and standalone Substitution Precision (50.67%) by conditioning on explicit lexical-phonotactic n-gram cluster transition probabilities from CMUDict.

The seven constituent models span four distinct, mutually complementary paradigms:
1. **Phonetic Posterior Sequences:** Conformer (EXP-10), SRAD-Net (EXP-17), NNA-Adapter (EXP-21).
2. **Representation Metric Learning:** PSC-Net (EXP-22) supervised prototype contrastive margin.
3. **Temporal Alignment Smoothing:** JSAC-Net (EXP-24) monotonic local Gaussian-windowed attention.
4. **Physical Energy & Linguistic Knowledge:** APED-Net (EXP-28) waveform RMS dynamics and PNC-Net (EXP-34) CMUDict phonotactic n-gram priors.

## 2. Formal Hypothesis
Forming a convex Bayesian mixture over all seven models:
$$P_{\text{septad}}(y \mid x) = \sum_{k=1}^7 w_k P_k(y \mid x), \quad \sum_{k=1}^7 w_k = 1, \quad w_k \ge 0$$
calibrated via global Pareto optimization on `DEV_CAL`, will:
1. Break the previous all-time campaign high for Weighted Utility, achieving $\text{Utility} \ge 0.4330$ on `DEV_REPLICATION`.
2. Push Macro F1 to $\ge 0.5820$ and Macro AP to $\ge 0.5820$.
3. Maintain Insertion F1 $\ge 0.3500$, Substitution AP $\ge 0.4000$, and Macro Precision $\ge 0.6150$.

## 3. Predicted Evidence & Significance
If validated, Septad-Pareto represents the culmination of multi-paradigm acoustic, prosodic, phonetic, and phonotactic fusion, achieving the highest overall diagnostic performance in this autonomous research project.
