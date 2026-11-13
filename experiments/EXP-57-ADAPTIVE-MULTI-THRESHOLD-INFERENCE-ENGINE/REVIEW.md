# Experiment Review: EXP-57-ADAPTIVE-MULTI-THRESHOLD-INFERENCE-ENGINE (AMT-Engine)

## 1. Executive Summary
Experiment EXP-57 developed and profiled the **Adaptive Multi-Threshold Inference Engine (AMT-Engine)**, modulating decision thresholds at the individual slot level conditioned on canonical manner of articulation, relative duration ratio, and posterior entropy.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **False Alarm Suppression:**
  * In DSCA-Net (EXP-55), total false alarms dropped to **495** (FWR = **1.65** false alarms/utterance).
  * In Undecim-Pareto (EXP-52), total false alarms dropped to **516** (FWR = **1.72** false alarms/utterance).
  * Stop consonant false omission alarms were successfully suppressed by the $+0.06$ adaptive threshold offset on plosives.
- **Sustained Milestone Performance:**
  * Undecim-Pareto Macro AP maintained **0.6244**, Macro F1 maintained **0.6060**, Omission AP maintained **0.4503**, and Substitution AP maintained **0.4481** (Precision = 51.78%).
  * DSCA-Net Weighted Utility maintained **0.4919**, Macro AP maintained **0.6159**, and Macro F1 maintained **0.6007**.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H52 Partially Confirmed:**
  * Dynamic threshold adaptation succeeds as a robust post-processing safety filter, significantly lowering false alarms for unreleased plosive consonants.
  * However, because AP is threshold-invariant by mathematical definition, achieving $\ge 80\%$ on all AP metrics requires continuing to expand representation capacity and multi-model consensus.
  * **Next Step (EXP-58):** Construct the **12-Model Duodecim Super-Ensemble ("Duodecim-Pareto")** uniting DSCA-Net into Undecim-Pareto to break the 0.5000 Utility milestone!
