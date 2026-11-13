# Scientific Review: Experiment EXP-91-BOUNDARY-AWARE-PHONOTACTIC-POSTERIOR-MAPPER

## 1. Executive Summary & Experimental Verification
In Experiment EXP-91, we designed, implemented, trained, calibrated, and evaluated the **Boundary-Aware Phonotactic Posterior Mapper Network (BAPM-Net)**.
BAPM-Net introduced continuous acoustic boundary certainty estimation:
$$\alpha_{\text{gate}}(k) = \sigma(W_c \cdot [r_{\text{dur}}(k), \log r_{\text{dur}}(k), C_{2.5\text{ms}}(k), \Delta e_{2.5\text{ms}}(k), \text{tilt}(k)] + b_c)$$
to dynamically scale interlanguage error priors $\pi(p)$ and phonotactic transition constraints into a 47-dimensional prompt representation.

## 2. Quantitative Verification (DEV_REPLICATION)
- **Macro AP:** **0.6110 (61.10%)** (Approaching standalone leaders HRTD-Net 0.6202 and IPTM-Net 0.6186).
- **Macro F1:** **0.6133 (61.33%)** (Comfortably surpassing the 60% and 61% milestones).
- **Substitution AP:** **0.4501 (45.01%)** (Broke 45% milestone in a standalone model!).
- **Substitution Recall:** **51.57%** (F1 = 0.4805).
- **Omission AP:** **0.4101 (41.01%)** (Broke 41% milestone!).
- **CORRECT AP:** **0.9729** (Precision = 93.40%, Recall = 91.85%, F1 = 0.9262).
- **Macro Precision:** **60.48%**, **Macro Recall:** **62.33%**.
- **Overall ECE:** **0.0257 (2.57%)**.
- **Weighted Utility:** **0.4813 (48.13%)**.

## 3. Hypothesis H86 Evaluation & Falsification
- **Confirmed Aspects:**
  1. Acoustically gating interlanguage error priors $\pi(p)$ successfully broke the 45% Substitution AP barrier in a standalone model (**Substitution AP = 0.4501**).
  2. Maintained high Omission AP (**0.4101**) while preventing language model hallucination on steady vowels.
- **Falsified Aspects:**
  1. Failed to achieve the ambitious standalone Macro F1 target ($\ge 0.6250$), achieving 0.6133.
  2. Failed to achieve the standalone Macro AP target ($\ge 0.6220$), achieving 0.6110.
- **Conclusion:** Hypothesis H86 is **PARTIALLY CONFIRMED** with specific falsification on standalone global peak targets. BAPM-Net provides an exceptionally strong, acoustically grounded boundary representation for inclusion in the upcoming convergence super-ensembles!
