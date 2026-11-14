# Research Plan: Experiment EXP-59 (Acoustic-Articulatory Latent Diffusion Sampler - ALDS-Net)

## 1. Technical Formulation & Architecture
Design, implement, train, and benchmark **ALDS-Net**:
- **Base Architecture:**
  - DSCA-Net backbone (EXP-55): 67-dim acoustic stream, 42-dim canonical prompt stream, bidirectional multi-head cross-attention $\rightarrow z_{\text{init}} \in \mathbb{R}^{128}$.
- **Conditional Score-Matching Diffusion Module:**
  - Latent space dimension: 128.
  - Diffusion Timestep Embedding: Sinusoidal positional embedding ($t \in [0, 1] \rightarrow \mathbb{R}^{64}$).
  - Denoiser Network ($\epsilon_\theta$):
    - Input: Noisy latent $z_t \in \mathbb{R}^{128}$ + Timestep embedding $t_{\text{emb}} \in \mathbb{R}^{64}$ + Canonical prompt conditioning $c_{\text{prompt}} \in \mathbb{R}^{128}$.
    - Architecture: 3-layer MLP ($320 \rightarrow 256 \rightarrow 256 \rightarrow 128$) with LayerNorm, GELU, and residual skip connections.
  - Objective: Mean squared error between true Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ and predicted noise $\epsilon_\theta(z_t, t, c_{\text{prompt}})$.
- **Lightweight Inference Sampler (5-step DDIM):**
  - Timesteps: $t \in [1.0, 0.75, 0.50, 0.25, 0.0]$.
  - Fast deterministic reverse trajectory: $z_0^* = \text{DDIM}(z_{\text{init}}, 5\text{ steps})$.
- **Diagnosis:**
  - Pass denoised $z_0^*$ through decoupled vowel/consonant diagnosis heads and gap insertion network.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features and canonical prompt stream from `EXP-55` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Warm-start base acoustic and prompt encoders from `DSCA-Net` (EXP-55).
3. Jointly train score matching denoiser and diagnostic heads on CPU (4 threads) for 10 epochs.
4. Profile CPU inference latency using 5-step DDIM sampling.
5. Calibrate operating thresholds on `DEV_CAL`.
6. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
7. Audit against H54 falsification criteria.
