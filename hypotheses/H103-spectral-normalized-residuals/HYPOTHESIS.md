# Hypothesis H103: Formant-Bandwidth Normalized Spectral Residual Network (FBNR-Net)

## 1. Context & Motivation
In EXP-71 (CASR-Net), raw spectral cepstral residuals increased substitution sensitivity but caused vocal tract confounding (FWR = 3.517). While VTLN frequency warping (EXP-74) partially alleviated this, spectral deviations are heavily distorted by vocal fold breathiness and glottal leakage, which widen formant bandwidths ($B_1, B_2$).
Non-native speakers frequently exhibit breathy phonation, producing wider formant bandwidths that mimic vowel substitution deviations in raw cepstral space.

## 2. Hypothesis Statement
We hypothesize that:
1. Estimating formant bandwidths $B_1, B_2$ from LPC pole radii:
   $$B_i = -\frac{f_s}{\pi} \ln |r_i|$$
   and normalizing the 8-dimensional cepstral residuals by the geometric mean bandwidth factor:
   $$\tilde{\Delta}_{\text{cep}}(k) = \frac{\Delta_{\text{cep}}(k)}{\sqrt{B_{12}(k) + 0.05}} \in \mathbb{R}^8$$
   will decouple glottal phonation noise from true phonetic substitution gestures.
2. In FBNR-Net:
   - Standalone **Substitution AP will reach $\ge 0.4550$ (45.50%)** on held-out `DEV_REPLICATION`.
   - Standalone **Substitution Precision will exceed $\ge 52.00\%$** with Substitution F1 $\ge 0.4850$.
   - False Warning Rate (FWR) will drop below $\le 2.65$ false alarms per utterance (eliminating breathiness false alarms).
   - Standalone **Weighted Utility will exceed $\ge 0.4950$ (49.50%)** and **Macro F1 $\ge 0.6180$ (61.80%)**.
   - Overall ECE will remain sub-2.5% ($\le 0.0240$).
3. Falsification will occur if bandwidth normalization fails to suppress false substitutions, inflating FWR above 2.95 or dropping Weighted Utility below 0.4850.
