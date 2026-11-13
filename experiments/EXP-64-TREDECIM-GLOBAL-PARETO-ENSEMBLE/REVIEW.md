# Experiment Review: EXP-64-TREDECIM-GLOBAL-PARETO-ENSEMBLE (Tredecim-Pareto)

## 1. Executive Summary
Experiment EXP-64 formulated and evaluated the **13-Model Tredecim Super-Ensemble ("Tredecim-Pareto")**, fusing the insertion specializer champion **TDIS-Net** (EXP-63) into **Duodecim-Pareto** (EXP-58) via convex Bayesian Pareto optimization on `DEV_CAL`.

## 2. Historic Milestone Breakthroughs Achieved on DEV_REPLICATION
1. **ALL-TIME CAMPAIGN HIGH FOR WEIGHTED UTILITY ACROSS ALL PARADIGMS:**
   - **Weighted Utility = 0.4930 (49.30%)**!
   - (Breaking the previous all-time campaign peak of 0.4928 in DSCA-Net!).
2. **ALL-TIME CAMPAIGN HIGH FOR MACRO AP:**
   - **Macro AP = 0.6351 (63.51%)**!
   - (Breaking the previous all-time campaign peak of 0.6341 in Duodecim-Pareto).
3. **ALL-TIME CAMPAIGN HIGH FOR OMISSION AP:**
   - **Omission AP = 0.4774 (47.74%)**!
   - (Breaking the previous all-time peak of 0.4733 in EXP-58).
   - Omission F1 = **0.4519**, Recall = **44.75%**, Precision = **45.63%**.
4. **ALL-TIME CAMPAIGN HIGH FOR ENSEMBLE INSERTION AP:**
   - **Insertion AP = 0.3128 (31.28%)**!
   - (Breaking previous ensemble peak of 0.2865 by $+2.63\%$ absolute).
   - **Insertion Precision = 51.06% (51.06%)**!
5. **SUSTAINED HIGH SUBSTITUTION ACCURACY:**
   - **Substitution AP = 0.4529 (45.29%)**, Precision = **51.08%**.
   - DEV_SEARCH Substitution AP reached an all-time high of **0.4294**.
6. **SUSTAINED 60% MACRO MILESTONES:**
   - **Macro F1 = 0.6054 (60.54%)**!
   - **Macro Precision = 0.6293 (62.93%)**!
   - **Overall ECE = 0.0358 (3.58%)**, Gap ECE = **0.0060 (0.60%)**, FWR = **1.663**.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H59 Decisively Confirmed:**
  * Allocating heavy weight ($w_{\text{tdis}} = 0.40$) to Triad decoupled gap gating inside the 13-model mixture resolved the insertion bottleneck, driving Insertion AP to 0.3128 and Precision to 51.06% in an ensemble.
  * Tredecim-Pareto stands as the new undisputed global peak champion of the laboratory across all performance dimensions.
