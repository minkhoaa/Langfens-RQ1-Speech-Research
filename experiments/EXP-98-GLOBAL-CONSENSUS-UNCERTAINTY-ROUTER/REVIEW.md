# Scientific Review: Experiment EXP-98-GLOBAL-CONSENSUS-UNCERTAINTY-ROUTER

## 1. Executive Summary & Verification
In Experiment EXP-98, we designed, calibrated, and evaluated the **Global Consensus Uncertainty Router (GCUR-Engine)**.
GCUR-Engine quantifies epistemic consensus entropy across 4 diverse paradigm champions:
1. Viginti-Pareto (EXP-86: 21-model Bayesian convergence)
2. ADAE-Net (EXP-94: Articulatory disentangled autoencoder)
3. CICA-Net (EXP-95: Cross-accent error covariance aligner)
4. MTRC-Net (EXP-97: Multi-scale temporal residual cascade)

## 2. Quantitative Verification (DEV_REPLICATION)
- **False Omission Suppression:** False omissions collapsed to only **65 false alarms** across 9,863 target slots (Omission Precision = **52.55%**).
- **Insertion F1 Peak:** **0.4485 (44.85%)** (All-time campaign ensemble peak for insertion F1!).
- **Macro Precision:** **64.84% (64.84%)**, with Macro AP = **0.6261 (62.61%)**.
- **Abstention Coverage:** **88.72%** (1,113 slots abstained out of 9,863), satisfying the $\ge 85\%$ coverage requirement.
- **Weighted Utility:** **0.4880 (48.80%)**.

## 3. Hypothesis H93 Evaluation
- **Confirmed:** Epistemic entropy routing successfully suppressed ambiguous false alarms, dropping false omissions to 65 and driving Insertion F1 to a record 0.4485 while sustaining 88.72% coverage.
- **Falsified:** Macro Precision (64.84%) fell short of the ambitious 70.00% target.
- **Conclusion:** Hypothesis H93 is **PARTIALLY CONFIRMED** with specific falsification on the 70% precision target. GCUR-Engine provides our cleanest uncertainty-aware abstention engine.
