# Hypothesis H54: Continuous Score-Based Latent Diffusion Denoising (ALDS-Net)

## 1. Context & Motivation
In English L2 pronunciation assessment, non-native acoustic trajectories are heavily corrupted by:
1. **Coarticulatory Interference:** L1 phonological habits drag the tongue toward native phonetic articulatory configurations.
2. **Hesitation and Articulatory Tremor:** Inconsistent phoneme boundaries and irregular vocal tract closures.

Deterministic networks (BiLSTM, Conformer, MHCA) attempt to map noisy, blended acoustic frames directly to discrete phonetic error categories in a single deterministic pass. When an acoustic frame falls midway between a canonical target and a substitution, deterministic models often produce uncalibrated probabilities, creating false alarms.

In generative modeling, **Score-Based Diffusion Models** excel at inverting complex noise processes by following the gradient of the data distribution (the score function $\nabla_z \log p(z)$). Conditioned on canonical prompt phonemes, a diffusion process can denoise the latent phonetic trajectory toward the true underlying articulatory target.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **ALDS-Net (Acoustic-Articulatory Latent Diffusion Sampler)**, which trains a score-matching denoising network $\epsilon_\theta(z_t, t, c_{\text{prompt}})$ over the 128-dimensional latent space of DSCA-Net (EXP-55), will learn to invert non-native acoustic blending.
2. At test time, applying a lightweight 5-step DDIM (Denoising Diffusion Implicit Models) reverse trajectory sampler:
   $$z_0^* = \frac{z_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(z_t, t, c_{\text{prompt}})}{\sqrt{\bar{\alpha}_t}}$$
   will strip away coarticulatory acoustic noise prior to hierarchical vowel/consonant diagnosis.
3. Because clean acoustic realizations are pulled directly toward canonical targets while genuine substitutions deviate along orthogonal error manifolds, ALDS-Net will achieve:
   - **Weighted Utility** breaking the 0.5000 milestone for the first time in a standalone model ($\text{Utility} \ge 0.5000$).
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Substitution AP** breaking $\ge 0.4500$.
   - **Omission AP** maintaining $\ge 0.4450$.
   - Inference latency remaining $\le 15.0$ ms on CPU with 5-step DDIM sampling.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (All-time standalone peak!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4500$.
4. **DEV_REPLICATION Omission AP:** $\ge 0.4450$.
