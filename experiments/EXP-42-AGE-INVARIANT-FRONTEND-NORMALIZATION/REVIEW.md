# Experiment Review: EXP-42-AGE-INVARIANT-FRONTEND-NORMALIZATION (VTLN-HTC-Net)

## 1. Executive Summary
Experiment EXP-42 tested Hypothesis H37 by incorporating speaker-level Vocal Tract Length Normalization (VTLN) parameters—spectral centroid warp factor $\alpha_{\text{vtln}}$, relative centroid ratio, and spectral energy balance—into the cascaded HTC-Net architecture.

## 2. Quantitative Verification Findings
- **All-Time Campaign Record on Replication:**
  * **Insertion AP:** **0.2640** (Breaking the previous replication high of 0.2589 in EXP-28!)
  * **Insertion Recall:** **39.36%** (Breaking the previous replication high of 38.30% in EXP-34!)
  * **Insertion F1:** **0.3558** (Precision = 32.46%)
- **Diagnostic Stability:**
  * **DEV_REPLICATION Utility:** **0.4277**
  * **DEV_REPLICATION Omission Recall:** **41.25%** (F1 = **0.3941**, AP = **0.3557**)
  * **Overall ECE:** **0.0513** (Near campaign low)
  * **False Warning Rate:** **1.49 false alarms/utterance**

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H37 Partially Confirmed:**
  * Speaker-level VTLN centroid scaling effectively stabilized high-frequency inter-phonemic gaps, establishing a new all-time high for Insertion AP on replication (**0.2640**) and pushing Insertion Recall to **39.36%**.
  * However, for adult vowel substitutions within L2-ARCTIC, unwarped HTC-Net (EXP-38) maintains higher substitution sensitivity (0.4405 Utility).
  * VTLN provides a valuable complementary representation for high-frequency burst localization and cross-corpus children speech adaptation.
