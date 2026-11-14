# Hypothesis H59: 13-Model Tredecim Super-Ensemble (Tredecim-Pareto)

## 1. Context & Motivation
Across 63 completed research experiments, the laboratory has pushed each error class past historic milestones:
1. **Omission AP & Substitution AP:** EXP-58 (Duodecim-Pareto) achieved **0.4733 Omission AP** and **0.4540 Substitution AP**, driving Macro AP to **0.6341**.
2. **Insertion Breakthrough:** EXP-63 (TDIS-Net) decoupled inter-phone gaps into three structural phonetic regimes (intra-word consonant clusters, coda releases, and pause gaps), shattering the insertion bottleneck with **0.4321 Insertion F1**, **0.3151 Insertion AP**, and **51.47% Insertion Precision**.
3. **Standalone Dual-Stream Representation:** EXP-55 (DSCA-Net) achieved **0.4928 Weighted Utility**.

However, Duodecim-Pareto (EXP-58) was trained *before* TDIS-Net (EXP-63) was invented. Its gap predictions relied on flat generic networks, limiting its Insertion F1 to 0.3660.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **13-Model Tredecim Super-Ensemble ("Tredecim-Pareto")** by fusing `TDIS-Net` (EXP-63) into `Duodecim-Pareto` (EXP-58) via a convex Bayesian Pareto mixture calibrated on `DEV_CAL` will unite:
   - 12 diverse acoustic/phonotactic/cross-attention models for target phone classification.
   - Triad decoupled gap gating (cluster, coda, pause) for inter-phone gap classification ($w_{\text{tdis}} \ge 0.30$).
2. Tredecim-Pareto will achieve:
   - **Weighted Utility** breaking the 0.5000 milestone for the first time in laboratory history ($\text{Utility} \ge 0.5000$).
   - **Macro AP** shattering the 64% milestone ($\text{Macro AP} \ge 0.6400$).
   - **Macro F1** sustaining $\ge 0.6150$ (solidly above the 60% milestone).
   - **Insertion F1** sustaining $\ge 0.4200$ with Insertion Precision $\ge 50.00\%$.
   - **Omission AP** maintaining $\ge 0.4700$.
   - **Substitution AP** maintaining $\ge 0.4500$.
   - **False Warning Rate** maintaining $\le 1.65$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (First system in campaign history to break 50%!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6400$ (All-time campaign record).
3. **DEV_REPLICATION Macro F1:** $\ge 0.6150$.
4. **DEV_REPLICATION Insertion F1:** $\ge 0.4200$.
