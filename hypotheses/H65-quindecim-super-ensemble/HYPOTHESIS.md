# Hypothesis H65: 15-Model Quindecim Super-Ensemble (Quindecim-Pareto)

## 1. Context & Motivation
Across 69 completed research cycles, the laboratory has established preeminent records across two complementary systems:
1. **EXP-67 Quattuordecim-Pareto (14-Model):** All-time ensemble peak with Macro AP = 0.6377, Substitution AP = 0.4601, Omission AP = 0.4780, Insertion AP = 0.3262, Insertion Precision = 53.33%, and FWR = 1.41.
2. **EXP-69 IPTM-Net (Interlanguage Prior Mapping):** All-time standalone peak for Macro F1 (0.6205) and Omission Recall (55.64%), powered by empirical interlanguage transfer priors $\pi(p) = [P_c, P_s, P_o]$ computed strictly on `train.jsonl`.

Because IPTM-Net was invented in EXP-69, Quattuordecim-Pareto (EXP-67) does not incorporate empirical interlanguage error transfer priors.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **15-Model Quindecim Super-Ensemble ("Quindecim-Pareto")** by fusing `IPTM-Net` (EXP-69) into `Quattuordecim-Pareto` (EXP-67) via a convex Bayesian Pareto mixture calibrated on `DEV_CAL` will unite:
   - Empirical interlanguage error transfer priors ($w_{\text{iptm}} \approx 0.18$)
   - Dual-stream audio-prompt cross-attention ($w_{\text{dsca}} \approx 0.18$)
   - 5ms sub-frame temporal micro-dynamics ($w_{\text{hrtd}} \approx 0.18$)
   - Phonotactically constrained score diffusion ($w_{\text{p-alds}} \approx 0.14$)
   - Triad decoupled gap gating ($w_{\text{tdis}} \ge 0.35$ on gaps)
   - 10 foundational acoustic/prosodic/phonological models.
2. Quindecim-Pareto will achieve:
   - **Macro AP** shattering the 64% milestone for the first time in laboratory history ($\text{Macro AP} \ge 0.6400$).
   - **Weighted Utility** breaking the 0.5000 milestone in an ensemble ($\text{Utility} \ge 0.5000$).
   - **Omission AP** breaking the 48% milestone ($\text{Omission AP} \ge 0.4800$).
   - **Substitution AP** breaking $\ge 0.4620$ with Substitution Precision $\ge 54.00\%$.
   - **Insertion AP** maintaining $\ge 0.3200$ with Insertion Precision $\ge 52.00\%$.
   - **Macro F1** sustaining $\ge 0.6100$.
   - **False Warning Rate** maintaining $\le 1.45$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6400$ (All-time campaign peak!).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (Breaking 50% ensemble milestone!).
3. **DEV_REPLICATION Omission AP:** $\ge 0.4800$ (Breaking 48% milestone!).
4. **DEV_REPLICATION Substitution AP:** $\ge 0.4620$.
