# Hypothesis H24: 6-Model Hexad Super-Ensemble (Hexad-Pareto)

## 1. Context and Problem Statement
In previous iterations:
- **Pentad-Pareto (EXP-27)** achieved all-time campaign records on `DEV_REPLICATION` for **Macro AP (0.5760)**, **Macro Precision (0.6272)**, and **Substitution AP (0.3943)** with record low FWR ($1.23$).
- **APED-Net (EXP-28)** achieved all-time campaign records for **Macro F1 (0.5725)**, **Substitution F1 (0.4461)**, **Insertion F1 (0.3579)**, and **Insertion AP (0.2589)** by incorporating physical waveform RMS energy dynamics.

These two approaches exhibit orthogonal, complementary strengths:
- Pentad-Pareto excels at precision and discrimination via diverse sequence architectures.
- APED-Net excels at recall and physical boundary verification via raw waveform energy contrast.

## 2. Formal Hypothesis
Forming a convex Bayesian mixture over all 6 breakthrough models:
$$P_{\text{hexad}}(y \mid x) = \sum_{k=1}^6 w_k P_k(y \mid x), \quad \sum_{k=1}^6 w_k = 1$$
calibrated via global multi-objective Pareto optimization on `DEV_CAL`, will:
1. Surpass the previous all-time campaign high Macro AP, achieving $\text{Macro AP} \ge 0.5800$.
2. Push Macro F1 to $\ge 0.5750$ on independent `DEV_REPLICATION`.
3. Simultaneously satisfy:
   - Substitution AP $\ge 0.3950$ and Substitution F1 $\ge 0.4000$.
   - Omission AP $\ge 0.3600$ and Omission F1 $\ge 0.3800$.
   - Insertion AP $\ge 0.2500$ and Insertion F1 $\ge 0.3000$.
   - Macro Precision $\ge 0.6200$ (solidly above the 60% milestone).
   - False Warning Rate $\le 1.40$ false alarms/utterance.

## 3. Predicted Evidence & Significance
If validated, Hexad-Pareto will establish the highest diagnostic capability in the history of this autonomous research lab, confirming that fusing multi-architecture sequence modeling with physical acoustic energy dynamics is the optimal formulation for English L2 pronunciation assessment.
