# Research Plan: Experiment EXP-49 (Contrastive Dynamic Masked Recovery Network - CDMR-Net)

## 1. Objective & Architecture
Implement and train **CDMR-Net** combining:
- **67-dimensional input**: 63 base acoustic/prosodic/phonotactic features + 4 high-resolution 5ms micro-temporal dynamics.
- **13-dimensional gap input**: 11 base gap features + 2 transition boundary gradients.
- **Dynamic sequence token masking**: 15% random slot corruption with auxiliary reconstruction loss.
- **Monotonic Local Gaussian Attention**: $\sigma = 3.0$ with LayerNorm.
- **Phonological Prototype Contrastive Head**:
  - Learnable 128-dimensional class prototypes ($c_{\text{corr}}, c_{\text{subs}}, c_{\text{omis}}$).
  - Cosine similarity projection concatenated with diagnostic features.
  - Temperature-scaled InfoNCE contrastive loss ($\tau = 0.10$).
- **Decoupled Hierarchical Vowel & Consonant Heads**: Conditioned on manner-of-articulation routing.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features from `EXP-48` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Warm-start core layers from `HRTD-Net` (EXP-48) and train for 10 epochs using AdamW ($\text{lr}=8 \times 10^{-4}$), multi-task articulatory loss, InfoNCE prototype loss ($\lambda_{\text{proto}}=0.20$), auxiliary reconstruction loss, and cost-sensitive classification loss.
3. Optimize decision thresholds on `DEV_CAL`.
4. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H44 falsification criteria.
