# Research Plan: Experiment EXP-65 (Phonotactically Guided Acoustic Latent Diffusion - P-ALDS-Net)

## 1. Technical Formulation & Architecture
Design, implement, train, and benchmark **P-ALDS-Net**:
- **Acoustic Backbone (67 dims):**
  - 51 base features + 4 tempo + 8 prosodic/phonotactic + 4 micro-dynamics 5ms.
  - Linear projection ($67 \rightarrow 128$) + 2-layer BiLSTM (96 units) + Monotonic Local Attention ($\sigma = 3.0$).
- **Canonical Prompt Stream (44 dims):**
  - 39 onehot canonical phones + 3 canonical articulatory priors + 2 CMUDict transition log-probabilities ($\log P(p_k \mid p_{k-1})$, $\log P(p_{k+1} \mid p_k)$).
  - Encoded via 2-layer MLP $\rightarrow 128$ dimensions.
- **Phonotactically Conditioned Score Denoiser:**
  - Denoiser Input: Latent $z_t \in \mathbb{R}^{128}$ + Sinusoidal timestep $t_{\text{emb}} \in \mathbb{R}^{64}$ + Prompt/Phonotactic conditioning $c_{\text{prompt}} \in \mathbb{R}^{128}$.
  - Architecture: 3-layer MLP ($320 \rightarrow 256 \rightarrow 256 \rightarrow 128$) with LayerNorm, GELU, and residual skip connections.
  - Reverse Sampling: 4-step DDIM deterministic sampler on CPU.
- **Triad Decoupled Insertion Gating (EXP-63):**
  - Decoupled heads: `cluster_head`, `coda_head`, `pause_head` routed by structural gap context.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features and CMUDict transition matrices from `EXP-48` and `EXP-55` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Construct 44-dimensional prompt stream incorporating phonotactic transition priors.
3. Train P-ALDS-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), joint score matching loss, multi-task articulatory loss, and focal cross-entropy.
4. Calibrate decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H60 falsification criteria.
