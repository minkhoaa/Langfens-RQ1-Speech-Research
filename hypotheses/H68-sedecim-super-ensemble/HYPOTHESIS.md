# Hypothesis H68: 16-Model Sedecim Super-Ensemble (Sedecim-Pareto)

## 1. Context & Motivation
Across 72 completed research cycles, the laboratory has developed two distinct classes of architectures:
1. **High-Precision Calibration Champions:**
   - **EXP-70 Quindecim-Pareto (15-Model):** Macro AP = 0.6390, Precision = 64.67%, Substitution AP = 0.4622, Omission AP = 0.4796, Utility = 0.4989, FWR = 1.413.
   - **EXP-72 LDC-Engine:** Macro Precision = 64.69%, FWR = 1.33 false alarms/utterance.
2. **High-Recall Acoustic Diversity Specialist:**
   - **EXP-71 CASR-Net (Cross-Attention Spectral-Residual Network):** Achieved an all-time campaign peak for Substitution Recall (57.65%), Macro Recall (63.78%), and Insertion Recall (32.98%).

While CASR-Net's high recall came at the cost of higher standalone false warnings (FWR = 3.517), ensemble decision theory dictates that blending an orthogonal, high-recall specialist into a high-precision convex mixture allows the ensemble to capture previously undetectable subtle substitutions while filtering out isolated false positives through consensus voting.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **16-Model Sedecim Super-Ensemble ("Sedecim-Pareto")** by fusing `CASR-Net` (EXP-71) into `Quindecim-Pareto` (EXP-70) via a 16-dimensional convex Bayesian Pareto mixture calibrated on `DEV_CAL` will combine:
   - Empirical interlanguage error transfer priors (IPTM-Net, $w \approx 0.15$)
   - Dual-stream audio-prompt cross-attention (DSCA-Net, $w \approx 0.18$)
   - 5ms sub-frame temporal micro-dynamics (HRTD-Net, $w \approx 0.18$)
   - Phonotactically constrained score diffusion (P-ALDS-Net, $w \approx 0.14$)
   - High-recall spectral cepstral cross-attention (CASR-Net, $w \approx 0.05$)
   - Triad decoupled gap gating (TDIS-Net, $w \ge 0.32$ on gaps)
   - 10 foundational acoustic/prosodic/phonological models.
2. Sedecim-Pareto will achieve:
   - **Macro AP** shattering the 64.0% milestone for the first time in laboratory history ($\text{Macro AP} \ge 0.6400$).
   - **Weighted Utility** breaking the 0.5000 milestone in an ensemble ($\text{Utility} \ge 0.5000$).
   - **Omission AP** breaking the 48.0% milestone ($\text{Omission AP} \ge 0.4800$).
   - **Substitution AP** breaking $\ge 0.4630$ (46.30%) with Substitution Precision $\ge 54.50\%$.
   - **Insertion AP** maintaining $\ge 0.3200$ with Insertion Precision $\ge 52.00\%$.
   - **Macro F1** sustaining $\ge 0.6100$ (61.00%).
   - **False Warning Rate** maintaining $\le 1.45$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6400$ (All-time campaign peak!).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (Breaking 50% ensemble milestone!).
3. **DEV_REPLICATION Omission AP:** $\ge 0.4800$ (Breaking 48% milestone!).
4. **DEV_REPLICATION Substitution AP:** $\ge 0.4630$.
