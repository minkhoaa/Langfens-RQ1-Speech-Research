# Hypothesis H49: Continuous Latent Manifold Dynamics (CLMD-Net) for Phonetic Boundary Sharpening

## 1. Context & Motivation
Laboratory findings from EXP-48 (HRTD-Net) and EXP-49 (CDMR-Net) established two complementary scientific truths:
1. **5ms sub-frame micro-dynamics** provide decisive cues for plosive bursts, breaking the 60% Macro F1 milestone (0.6050) and setting all-time highs in Omission AP (0.4375) and Macro AP (0.6202).
2. **Discrete prototype clustering** (EXP-49) failed because non-native phonetic deviations exist along a continuous acoustic spectrum. Rigid clustering penalized natural continuous allophonic variance.

To advance toward the 80% outstanding gate, we require a **continuous, non-discrete manifold regularizer** that models physical articulatory continuity:
- **Intra-Slot Continuity:** Within a single realized phone, the acoustic vocal tract configuration changes smoothly; rapid latent oscillations represent noisy frame jitter that causes false substitution warnings.
- **Inter-Slot Boundary Discontinuity:** At phoneme transitions (especially plosive release bursts and vowel-to-consonant closures), the vocal tract undergoes rapid geometric shifts. A sharp latent discontinuity must be preserved to prevent omission smearing.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Formulating **CLMD-Net (Continuous Latent Manifold Dynamics Network)** by augmenting HRTD-Net with a dual-term continuous manifold regularization loss:
   $$\mathcal{L}_{\text{manifold}} = \lambda_{\text{smooth}} \sum_{t=1}^{T-1} \| h_{t+1} - h_t \|_2^2 + \lambda_{\text{boundary}} \sum_{k=1}^{N-1} \max\left(0, m - \| h_{k, \text{end}} - h_{k+1, \text{start}} \|_2\right)$$
   will regularize the 5ms sub-frame latent trajectory to be internally coherent while accentuating genuine phonetic transition edges.
2. CLMD-Net will achieve:
   - **Substitution AP** breaking the 46% milestone ($\text{Substitution AP} \ge 0.4600$).
   - **Omission AP** maintaining or exceeding $\ge 0.4400$.
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Weighted Utility** breaking the 0.4900 milestone ($\text{Utility} \ge 0.4900$).
   - Reduction of false substitution warnings by $\ge 10\%$ relative to HRTD-Net.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
2. **DEV_REPLICATION Macro F1:** $\ge 0.6100$.
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4600$.
4. **DEV_REPLICATION Weighted Utility:** $\ge 0.4900$.
