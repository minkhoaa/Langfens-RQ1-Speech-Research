# Hypothesis H70: Dialect-Conditioned Phonotactic Latent Diffusion Network (DP-ALDS-Net)

## 1. Context & Motivation
Across previous cycles, the laboratory established:
1. **EXP-65 (P-ALDS-Net):** Phonotactically guided score diffusion achieved an all-time standalone record for Substitution Precision (55.83%) and Substitution AP (0.4485), solving the over-denoising problem of naive diffusion.
2. **EXP-53 (DIAA-Net):** 16-dimensional L1 dialect embeddings reduced cross-accent variance by 20.0%, proving that phonetic deviations are structured by native language phonotactics.
3. **EXP-72 (LDC-Engine):** Demonstrated that dialect-stratified modeling cuts false warnings to an all-time record low of 1.33 false alarms per utterance.

Currently, P-ALDS-Net conditions diffusion only on CMUDict phonotactic n-grams, lacking explicit L1 dialect conditioning during the reverse score-matching sampling process.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Dialect-Conditioned Phonotactic Latent Diffusion Network (DP-ALDS-Net)** by:
   - Augmenting the conditioning prompt stream to 63 dimensions:
     * 42-dim canonical phone one-hot + phonetic manner/place embedding
     * 2-dim CMUDict phonotactic transition priors
     * 3-dim empirical interlanguage error priors $\pi(p)$ from TRAIN
     * 16-dim learned L1 dialect embedding $e_{\text{dialect}}$
   - Training a score-matching diffusion network with 4-step DDIM sampling to denoise acoustic latent representations conditioned on $[p, \text{phonotactics}, \text{L1}, \pi(p)]$.
2. DP-ALDS-Net will achieve:
   - **Substitution Precision:** $\ge 56.00\%$ (All-time standalone peak!).
   - **Substitution AP:** $\ge 0.4550$ (45.50%) (Surpassing P-ALDS-Net 0.4485).
   - **Macro Precision:** $\ge 0.6350$ (63.50%).
   - **Macro F1:** $\ge 0.6100$ (61.00%).
   - **Weighted Utility:** $\ge 0.4900$ (49.00%).
   - **Overall ECE:** $\le 0.0125$ (1.25%) with False Warning Rate $\le 1.50$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Substitution Precision:** $\ge 56.00\%$ (All-time standalone peak!).
2. **DEV_REPLICATION Substitution AP:** $\ge 0.4550$.
3. **DEV_REPLICATION Macro F1:** $\ge 0.6100$.
4. **Zero-Leakage Compliance:** 100% verified (L1 is learner demographic metadata; inference requires only audio waveform and canonical prompt text).
