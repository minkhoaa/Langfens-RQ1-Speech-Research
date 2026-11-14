# Hypothesis H89: Articulatory Disentangled Autoencoder Network (ADAE-Net)

## 1. Context & Motivation
Phonetic mispronunciations occur along three distinct phonological dimensions:
1. **Manner of Articulation:** Plosive, Fricative, Nasal, Approximant, Vowel.
2. **Place of Articulation:** Bilabial, Alveolar, Velar, Palatal, Dental.
3. **Voicing:** Voiced vs Unvoiced.
However, in standard neural acoustic encoders, these articulatory dimensions are entangled with acoustic timbre (speaker vocal tract size, pitch, formants). When non-native speakers speak with high pitch or breathy voice, standard encoders frequently conflate vocal timbre with manner errors.

## 2. Hypothesis Statement
We hypothesize that:
1. Structuring the acoustic latent representation into partitioned subspaces:
   $$z = [z_{\text{manner}} \in \mathbb{R}^{32}, z_{\text{place}} \in \mathbb{R}^{32}, z_{\text{voicing}} \in \mathbb{R}^{16}, z_{\text{timbre}} \in \mathbb{R}^{48}]$$
   and enforcing a Frobenius norm orthogonality loss:
   $$\mathcal{L}_{\text{ortho}} = \| \text{Cov}(z_{\text{manner}}, z_{\text{place}}) \|_F^2 + \| \text{Cov}(z_{\text{manner}}, z_{\text{voicing}}) \|_F^2 + \| \text{Cov}(z_{\text{articulatory}}, z_{\text{timbre}}) \|_F^2$$
   will disentangle speaker anatomy from articulatory gestures.
2. In ADAE-Net:
   - Standalone **Macro F1 will exceed $\ge 0.6150$ (61.50%)**.
   - Standalone **Weighted Utility will exceed $\ge 0.4850$ (48.50%)**.
   - Substitution Precision will exceed $\ge 52.00\%$ with False Warning Rate $\le 2.20$ false alarms per utterance.
3. Falsification will occur if orthogonality constraints over-regularize the latent space, dropping Macro F1 below 0.6050 or Utility below 0.4750.
