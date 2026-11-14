# Hypothesis H47: 11-Model Undecim Super-Ensemble (Undecim-Pareto)

## 1. Context & Motivation
Across 51 experiments, two pinnacle systems have emerged with complementary strengths:
1. **EXP-48 HRTD-Net (Standalone Champion):** Powered by 5ms sub-frame acoustic micro-dynamics, HRTD-Net broke the 60% Macro F1 milestone (0.6050) and achieved all-time campaign records: Macro AP = 0.6202, Weighted Utility = 0.4846, Substitution AP = 0.4487, Omission AP = 0.4375, and ECE = 0.0117.
2. **EXP-45 Decimal-Pareto (10-Model Ensemble Champion):** Combines 10 diverse models (including VTLN formant warping, CMUDict bigrams, and prosodic energy dynamics), achieving the lowest false alarm rate in laboratory history (1.09 - 1.11 FWR) and 55.13% Substitution Precision.

Crucially, Decimal-Pareto was trained *before* HRTD-Net was invented, meaning the 10-model ensemble completely lacks 5ms sub-frame micro-temporal dynamics!

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Formulating the **11-Model Undecim Super-Ensemble ("Undecim-Pareto")** by fusing the 10 foundational models of Decimal-Pareto with the 5ms sub-frame micro-dynamics of **HRTD-Net** via a convex Bayesian mixture calibrated on `DEV_CAL` will unite the ultra-low false alarm rate of the ensemble with the sharp plosive/affricate boundary sensitivity of HRTD-Net.
2. Undecim-Pareto will achieve:
   - **Macro AP** shattering the 63% milestone ($\text{Macro AP} \ge 0.6300$).
   - **Weighted Utility** breaking the 0.5000 milestone ($\text{Utility} \ge 0.5000$).
   - **Macro F1** sustaining $\ge 0.6050$ (solidly above the 60% milestone).
   - **Substitution Precision** rising to $\ge 56.00\%$.
   - **False Warning Rate** maintaining $\le 1.15$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6300$ (breaking previous 0.6202).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (breaking previous 0.4846).
3. **DEV_REPLICATION Substitution Precision:** $\ge 56.00\%$.
4. **DEV_REPLICATION FWR:** $\le 1.15$.
