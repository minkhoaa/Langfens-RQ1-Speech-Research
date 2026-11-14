# Hypothesis H71: 17-Model Septendecim Super-Ensemble (Septendecim-Pareto)

## 1. Context & Motivation
Across 75 completed research cycles, the laboratory has systematically isolated:
1. High-precision calibration in **Quindecim-Pareto** (EXP-70, Macro AP = 0.6390, Precision = 64.67%, Utility = 0.4989).
2. High-recall acoustic diversity in **CASR-Net** (EXP-71, Substitution Recall = 57.65%, Macro Recall = 63.78%).
3. Mathematical false alarm suppression via VTLN in **V-CASR-Net** (EXP-74, eliminating 232 false substitutions and lifting Utility to 0.4788 and Macro F1 to 0.6139).
4. Dialect-conditioned diffusion in **DP-ALDS-Net** (EXP-75, setting a search Substitution AP peak of 0.4548 and Gap ECE of 0.0030).

Because V-CASR-Net eliminates the vocal tract length false alarms that previously held Sedecim-Pareto at 0.6383, fusing V-CASR-Net and DP-ALDS-Net into a 17-model convex mixture enables the ensemble to break the historic 0.6400 Macro AP barrier.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **17-Model Septendecim Super-Ensemble ("Septendecim-Pareto")** by fusing:
   - 10 foundational acoustic/prosodic/phonological models (M1-M10)
   - HRTD-Net (M11: 5ms sub-frame micro-dynamics, $w \approx 0.18$)
   - DSCA-Net (M12: dual-stream audio-prompt cross-attention, $w \approx 0.18$)
   - TDIS-Net (M13: triad decoupled gap gating, $w_{\text{gap}} \ge 0.33$)
   - P-ALDS-Net (M14: phonotactic score diffusion, $w \approx 0.13$)
   - IPTM-Net (M15: empirical interlanguage error transfer priors, $w \approx 0.15$)
   - V-CASR-Net (M16: VTLN-warped spectral cepstral residuals, $w \approx 0.04$)
   - DP-ALDS-Net (M17: dialect-conditioned score diffusion, $w \approx 0.03$)
2. Septendecim-Pareto will achieve:
   - **Macro AP** shattering the 64.0% milestone for the first time in laboratory history ($\text{Macro AP} \ge 0.6400$).
   - **Weighted Utility** breaking the 0.5000 milestone in an ensemble ($\text{Utility} \ge 0.5000$).
   - **Omission AP** breaking the 48.0% milestone ($\text{Omission AP} \ge 0.4800$).
   - **Substitution AP** breaking $\ge 0.4630$ (46.30%) with Substitution Precision $\ge 54.50\%$.
   - **Insertion AP** maintaining $\ge 0.3250$ with Insertion Precision $\ge 52.50\%$.
   - **Macro F1** sustaining $\ge 0.6120$ (61.20%).
   - **False Warning Rate** maintaining $\le 1.45$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6400$ (All-time campaign peak!).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (Breaking 50% ensemble milestone!).
3. **DEV_REPLICATION Omission AP:** $\ge 0.4800$ (Breaking 48% milestone!).
4. **DEV_REPLICATION Macro F1:** $\ge 0.6120$.
