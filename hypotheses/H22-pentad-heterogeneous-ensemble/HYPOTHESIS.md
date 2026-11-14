# Hypothesis H22: 5-Model Heterogeneous Pentad Ensemble (Pentad-Pareto)

## 1. Context and Problem Statement
In previous iterations, individual models developed specialized strengths:
1. **JSAC-Net (EXP-24)** achieved record standalone Omission AP ($0.2251$) via Monotonic Local Attention.
2. **PSC-Net (EXP-22)** achieved record standalone Substitution AP ($0.3874$) via Cosine Margin Contrastive learning.
3. **SRAD-Net (EXP-17)** achieved record standalone Substitution F1 ($0.3763$) via speaking-rate normalized tempo modeling.
4. **NNA-Adapter (EXP-21)** achieved record standalone Omission Recall ($44.98\%$) via acoustic bottleneck adaptation.
5. **Conformer (EXP-10)** achieved record calibration ($0.0353$ ECE) and sharp insertion localization via multi-head self-attention.

EXP-26 proved that attempting to force all five mechanisms into a single gradient backpropagation graph creates gradient interference between discrete contrastive prototype clustering and continuous low-energy omission gradients. Therefore, an unconstrained Bayesian mixture over all five independently trained models—**Pentad-Pareto**—is hypothesized to maximize the Pareto frontier without gradient conflict.

## 2. Formal Hypothesis
Forming a convex Bayesian mixture:
$$P(y \mid x) = \sum_{k=1}^5 w_k P_k(y \mid x), \quad \sum_{k=1}^5 w_k = 1, \quad w_k \ge 0$$
calibrated via grid optimization on `DEV_CAL`, will:
1. Surpass the previous quad-ensemble Utility record ($0.4310$ on `DEV_REPLICATION`).
2. Maintain high Omission F1 ($\ge 0.4000$) while pushing Substitution AP beyond $0.3950$.
3. Preserve Macro Precision above $0.6150$ and low calibration error ($\text{ECE} \le 0.0650$).

## 3. Predicted Evidence & Significance
If validated, Pentad-Pareto will establish an all-time campaign high in weighted utility and cross-speaker replication robustness, demonstrating that architectural diversity under Bayesian fusion effectively circumvents single-model multi-task gradient conflict.
