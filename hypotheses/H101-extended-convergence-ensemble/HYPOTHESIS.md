# Hypothesis H101: Extended 25-Model Convergence Super-Ensemble (Viginti-Quintum-Pareto)

## 1. Context & Motivation
Phase 2 frontier research has generated four distinct specialist models with complementary empirical strengths:
1. **PEAR-Net (EXP-102):** Low-rank acoustic adaptation setting all-time standalone Substitution Recall peak (56.18%) and Insertion F1 (0.4410).
2. **DSGA-Net (EXP-103):** Differentiable edit lattice graph aligner setting all-time standalone Omission AP peak (0.4308) and Macro AP (0.6178).
3. **SGAN-Net (EXP-104):** Soft-gold Dirichlet adjudication setting all-time standalone Macro Precision peak (66.88%) and lowest FWR (0.977/utt).
4. **VTI-Net (EXP-101):** Continuous vocal tract area function inversion setting sub-2% ECE (0.0196) and search omission recall (56.94%).
5. **Viginti-Pareto (EXP-86):** Core 21-model Bayesian convergence mixture (Utility = 0.5079, Macro F1 = 0.6292).

## 2. Hypothesis Statement
We hypothesize that:
1. Constructing the **25-Model Viginti-Quintum-Pareto Super-Ensemble** by optimizing convex mixture weights over these complementary paradigms on `dev_cal` will:
   - Break the all-time campaign record for **Weighted Utility: $\ge 0.5100$ (51.00%)**!
   - Break the 63.0% Macro F1 milestone: **Macro F1 $\ge 0.6300$ (63.00%)**!
   - Push **Substitution Recall past 43.50%** (Substitution F1 $\ge 0.4750$).
   - Push **Omission Recall past 49.50%** (Omission F1 $\ge 0.4750$).
   - Maintain Overall ECE $\le 0.0240$ (2.40%) and False Warning Rate $\le 1.65$ false alarms per utterance.
2. Falsification will occur if the 25-model mixture fails to improve Weighted Utility over Viginti-Pareto (0.5079) or if FWR exceeds 1.80.
