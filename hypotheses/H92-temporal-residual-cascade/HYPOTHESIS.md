# Hypothesis H92: Multi-Scale Temporal Residual Cascade Network (MTRC-Net)

## 1. Context & Motivation
Speech sounds operate on vastly different intrinsic timescales:
- Stop consonant release bursts: 5-15 ms.
- Fricative noise: 30-70 ms.
- Vowel nuclei: 80-250 ms.
Static multi-resolution concatenation treats all timescales equally. A stop consonant requires heavy weighting of 2.5ms micro-bursts, whereas a prolonged vowel is corrupted by micro-burst noise.

## 2. Hypothesis Statement
We hypothesize that:
1. Constructing a cascaded multi-scale acoustic residual network:
   $$z_{\text{cascade}}(k) = z_{10\text{ms}}(k) + g_1(k) \odot \Delta z_{5\text{ms}}(k) + g_0(k) \odot \Delta z_{2.5\text{ms}}(k) + g_3(k) \odot z_{40\text{ms}}(k)$$
   where $g_s(k) = \sigma(W_s z_{10\text{ms}}(k) + b_s)$ are phoneme-dependent gating gates will:
   - Allow stops and affricates to attend to micro-burst acoustics while allowing vowels to attend to stable contexts.
   - Sustain standalone **Macro F1 $\ge 0.6200$ (62.00%)** and **Weighted Utility $\ge 0.4950$ (49.50%)**.
   - Achieve stop consonant substitution precision $\ge 52.00\%$.
2. Falsification will occur if cascaded gating fails to achieve Macro F1 $\ge 0.6120$ or Weighted Utility $\ge 0.4800$.
