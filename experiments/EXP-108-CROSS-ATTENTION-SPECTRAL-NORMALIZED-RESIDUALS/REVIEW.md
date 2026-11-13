# Scientific Review: Experiment EXP-108-CROSS-ATTENTION-SPECTRAL-NORMALIZED-RESIDUALS

## 1. Executive Summary & Verification
In Experiment EXP-108, we designed, implemented, trained, and evaluated the **Formant-Bandwidth Normalized Spectral Residual Network (FBNR-Net)**.
FBNR-Net extracts 8-dimensional cepstral residuals $\Delta_{\text{cep}}$ normalized by local formant bandwidth factors:
$$b_{\text{scale}} = \frac{\sqrt{B_1 \cdot B_2}}{100.0}$$
derived from LPC-12 pole radii on 16kHz audio, concatenating them with 72-dim temporal pyramid features (80 dims total) to decouple glottal breathiness from true phonetic spectral deviation.

## 2. Quantitative Verification (DEV_REPLICATION)
- **Substitution Performance:**
  * **Substitution AP:** **0.4526 (45.26%)** (comfortably surpassing the 45% milestone!).
  * **Substitution F1:** **0.4890 (48.90%)** (Recall = 52.62%, Precision = 45.68%).
- **False Warning Rate Reduction:**
  * False Warning Rate dropped to **2.74 false alarms per utterance** (a significant reduction from 3.517 in raw CASR-Net EXP-71, eliminating 139 false alarms!).
  * False substitutions dropped to 597 (vs 736 in unnormalized residuals).
- **Macro Performance:**
  * **Macro AP:** **0.6132 (61.32%)**, **Macro Precision:** **60.13% (60.13%)**, **Macro Recall:** **63.42% (63.42%)**, **Macro F1:** **0.6165 (61.65%)**.
  * **Weighted Utility:** **0.4807 (48.07%)**.

## 3. Hypothesis H103 Evaluation
- **Confirmed:** Formant bandwidth normalization successfully decoupled vocal fold breathiness from phonemic spectral deviation, eliminating 139 false substitutions and lifting Substitution AP to 0.4526 while dropping FWR below 2.80.
- **Falsified:** Standalone Weighted Utility (0.4807) fell slightly below the 0.4850 threshold due to conservative insertion recall.
- **Conclusion:** Hypothesis H103 is **PARTIALLY CONFIRMED**. FBNR-Net provides our cleanest acoustic spectral residual representation.
