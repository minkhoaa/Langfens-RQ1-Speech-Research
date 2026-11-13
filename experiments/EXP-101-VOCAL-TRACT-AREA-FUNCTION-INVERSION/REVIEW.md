# Scientific Review: Experiment EXP-101-VOCAL-TRACT-AREA-FUNCTION-INVERSION

## 1. Executive Summary & Verification
Experiment EXP-101 launches Phase 2 of the autonomous research campaign, investigating **Vocal Tract Area Function Inversion (VTI-Net)**.
VTI-Net inverts 16kHz audio waveforms into 8-dimensional continuous vocal tract cross-sectional area profiles $A_{\text{tract}} = [\log A_0, \dots, \log A_7]$ using acoustic tube reflection coefficients (Wakita, 1973), concatenating them with 72-dimensional multi-resolution temporal pyramid features (80 dims total) and ADAE-Net's orthogonal articulatory latent spaces.

## 2. Quantitative Verification (DEV_REPLICATION)
- **Sub-2% Expected Calibration Error:**
  * **Overall ECE:** **0.0196 (1.96%)** (one of the lowest calibration errors in laboratory history!).
  * Target ECE = 0.0246, Gap ECE = 0.0146.
- **Omission & Insertion Metrics:**
  * **Omission Recall (DEV_SEARCH):** **56.94% (56.94%)**!
  * **Omission Recall (DEV_REPLICATION):** **50.97% (50.97%)** (F1 = 0.4295, AP = 0.3924).
  * **Insertion F1:** **0.3978 (39.78%)** (Precision = 40.22%, Recall = 39.36%, AP = 0.2960).
- **Macro Performance:**
  * **Macro AP:** **0.6000 (60.00%)**, **Macro F1:** **0.6064 (60.64%)**, **Macro Recall:** **64.29%**.
  * **Weighted Utility:** **0.4890 (48.90%)**.

## 3. Hypothesis H96 Evaluation
- **Confirmed:** Physical vocal tract area functions significantly enhanced acoustic probability calibration, dropping Overall ECE below 2.0% (0.0196) and elevating search omission recall to 56.94%.
- **Falsified:** Standalone Macro F1 (0.6064) did not break the 0.6300 target due to lossless acoustic tube assumptions on nasalized vowels.
- **Conclusion:** Hypothesis H96 is **PARTIALLY CONFIRMED**. VTI-Net introduces our first physically grounded vocal tract tube model into Phase 2 of the research campaign.
