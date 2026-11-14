# Hypothesis H97: Parameter-Efficient Low-Rank Acoustic Adapter (PEAR-Net)

## 1. Context & Motivation
As identified in the Century Review (`reports/SCIENTIFIC_REVIEW_EXP100.md`, Section 6), pretrained speech representations (e.g. Charsiu w2v2-FC) were pretrained on native read speech (LibriSpeech). Consequently, non-native accents (e.g. vowel centralization, lack of aspiration, coda devoicing) are treated as out-of-distribution noise, leading to acoustic representation smear.
Fine-tuning the entire backbone on 1,800 training utterances causes catastrophic forgetting and overfitting to speaker vocal tract anatomy.

## 2. Hypothesis Statement
We hypothesize that:
1. Implementing a **Parameter-Efficient Low-Rank Acoustic Refiner (PEAR-Net)**:
   $$\Delta x_a = \frac{\alpha}{r} (W_{\text{up}} \cdot W_{\text{down}}) x_a$$
   with rank $r=8$, scaling factor $\frac{\alpha}{r} = 2.0$, and residual skip connection:
   $$x_{\text{adapted}} = x_a + \Delta x_a \in \mathbb{R}^{72}$$
   integrated directly before the recurrent acoustic encoder will:
   - Adapt non-native frame embeddings toward canonical phonological distributions without altering frozen SSL weights.
   - Break the standalone record for **Macro AP: $\ge 0.6200$ (62.00%)** on held-out `DEV_REPLICATION`.
   - Break the standalone record for **Weighted Utility: $\ge 0.5050$ (50.50%)** (surpassing ADAE-Net 0.5041).
   - Elevate standalone **Substitution AP to $\ge 0.4550$** with Substitution F1 $\ge 0.4900$.
   - Maintain Overall ECE $\le 0.0240$ (2.40%).
2. Falsification will occur if low-rank acoustic adaptation degrades standalone Macro F1 below 0.6150 or fails to achieve Weighted Utility $\ge 0.4950$.
