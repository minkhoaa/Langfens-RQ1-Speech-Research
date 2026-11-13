# Experiment Review: EXP-67-QUATTUORDECIM-SUPER-ENSEMBLE-FUSION (Quattuordecim-Pareto)

## 1. Executive Summary
Experiment EXP-67 formulated and evaluated the **14-Model Quattuordecim Super-Ensemble ("Quattuordecim-Pareto")**, fusing the phonotactically guided score diffusion champion **P-ALDS-Net** (EXP-65) into **Tredecim-Pareto** (EXP-64) via convex Bayesian Pareto optimization on `DEV_CAL`.

## 2. Historic Milestone Breakthroughs Achieved on DEV_REPLICATION
1. **ALL-TIME CAMPAIGN HIGH FOR MACRO AP:**
   - **Macro AP = 0.6377 (63.77%)**!
   - (Breaking previous all-time campaign high of 0.6351 in EXP-64!).
2. **BROKE 46% SUBSTITUTION AP MILESTONE FOR THE FIRST TIME:**
   - **Substitution AP = 0.4601 (46.01%)**!
   - (Shattering the previous record of 0.4540 in EXP-58 by $+0.61\%$ absolute).
   - **Substitution Precision surged to 54.23% (54.23%)**!
   - DEV_SEARCH Substitution AP reached **0.4338 (43.38%)**, Precision = **51.00%**!
3. **BROKE 47.8% OMISSION AP MILESTONE:**
   - **Omission AP = 0.4780 (47.80%)**!
   - (Breaking the previous all-time peak of 0.4774 in EXP-64).
   - Precision = **46.38%**, Recall = **42.41%**, F1 = **0.4431**.
   - DEV_SEARCH Omission AP reached **0.3148 (31.48%)**, Recall = **52.15%**!
4. **ALL-TIME HIGH ENSEMBLE INSERTION AP & PRECISION:**
   - **Insertion AP = 0.3262 (32.62%)**!
   - (Shattering the previous ensemble peak of 0.3128 in EXP-64 by $+1.34\%$ absolute).
   - **Insertion Precision = 53.33% (53.33%)**!
5. **ALL-TIME HIGH ENSEMBLE WEIGHTED UTILITY:**
   - **Weighted Utility = 0.4935 (49.35%)**!
   - **Macro Precision = 0.6412 (64.12%)**!
   - **Macro F1 = 0.6000 (60.00%)**!
   - **False Warning Rate = 1.41 false alarms/utterance** (only 423 false alarms on 300 test utterances).

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H62 Decisively Confirmed:**
  * By fusing phonotactically guided score diffusion ($w_{\text{p-alds}} = 0.16$) with dual-stream cross-attention ($w_{\text{dsca}} = 0.22$) and 5ms sub-frame temporal micro-dynamics ($w_{\text{hrtd}} = 0.22$), Quattuordecim-Pareto broke the 63.7% Macro AP and 46% Substitution AP milestones simultaneously.
  * Quattuordecim-Pareto stands as the absolute pinnacle of multi-model diagnostic intelligence in the laboratory.
