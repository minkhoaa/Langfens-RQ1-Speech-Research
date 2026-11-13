# Experiment Review: EXP-49-ENDLESS-SEARCH-CONTRASTIVE-DYNAMIC-RECOVERY (CDMR-Net)

## 1. Executive Summary
Experiment EXP-49 investigated the integration of phonological contrastive prototype learning (InfoNCE loss, $\tau=0.10$) with 5ms sub-frame temporal micro-dynamics and dynamic sequence token masking in **CDMR-Net**.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **Strong Baseline Performance:**
  * Weighted Utility: **0.4680** (Exceeding DMCT-Net FP32 0.4535).
  * Macro AP: **0.5956**.
  * Macro F1: **0.5912**.
  * Omission AP: **0.4170** (Recall = 42.41%, F1 = 0.4308).
  * Overall ECE: **0.0166** (Superb calibration).
- **Comparative Analysis vs HRTD-Net (EXP-48):**
  * HRTD-Net achieved **Macro AP = 0.6202**, **Macro F1 = 0.6050**, and **Utility = 0.4846**.
  * CDMR-Net yielded slightly lower Substitution AP (0.3988 vs 0.4487) because hard prototype clustering penalizes non-native accent variations that deviate continuously from canonical centroids.

## 3. Crucial Scientific Insight
- Unconstrained continuous attention over 5ms sub-frame dynamics (HRTD-Net) is superior to metric prototype clustering for second-language pronunciation diagnosis.
- L2 pronunciations exist along a continuous acoustic spectrum rather than discrete clusters. Attempting to force them into rigid prototypes artificially inflates substitution false alarms.
- **HRTD-Net (EXP-48) is definitively established as the primary scientific champion.**
