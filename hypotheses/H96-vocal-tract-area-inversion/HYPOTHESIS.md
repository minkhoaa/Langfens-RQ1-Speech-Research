# Hypothesis H96: Vocal Tract Area Function Inversion Network (VTI-Net)

## 1. Context & Motivation
As documented in the Century Retrospective (`reports/SCIENTIFIC_REVIEW_EXP100.md`, Section 6), traditional frame-level acoustic classifiers suffer from acoustic indeterminacy when evaluating non-native substitutions and omissions. Speech sounds are physically produced by spatial constriction gestures along the vocal tract tube.
According to acoustic tube theory (Fant, 1960; Wakita, 1973; Stevens, 1998):
LPC reflection coefficients $r_k$ directly map to vocal tract cross-sectional area ratios:
$$\frac{A_{k+1}}{A_k} = \frac{1 - r_k}{1 + r_k}$$
yielding an 8-dimensional continuous physical area profile $A_{\text{tract}} = [\log A_0, \dots, \log A_7] \in \mathbb{R}^8$ spanning the pharynx ($k=0, 1$), velum ($k=2, 3$), alveolar ridge ($k=4, 5$), and lips ($k=6, 7$).

## 2. Hypothesis Statement
We hypothesize that:
1. Extracting continuous 8-dimensional vocal tract cross-sectional area functions $A_{\text{tract}}$ and integrating them with the 72-dimensional multi-resolution acoustic stream (total 80 dims) inside the **Vocal Tract Area Function Inversion Network (VTI-Net)** will:
   - Physically ground place-of-articulation and vowel-height distinctions in vocal tract constriction geometry.
   - Achieve standalone **Macro F1 $\ge 0.6300$ (63.00%)**, breaking the 63.0% milestone for the first time in an individual model.
   - Achieve standalone **Weighted Utility $\ge 0.5060$ (50.60%)**.
   - Push standalone **Substitution F1 past 0.5000 (50.00%)** (with Substitution Recall $\ge 56.00\%$).
   - Maintain Overall ECE $\le 0.0240$ (2.40%) and clean FWR $\le 2.90$ false alarms per utterance.
2. Falsification will occur if vocal tract area inversion fails to improve standalone Macro F1 over ADAE-Net (0.6285) or degrades Weighted Utility below 0.4950.
