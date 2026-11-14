# Research Plan: Experiment EXP-75 (Dialect-Conditioned Phonotactic Latent Diffusion Network - DP-ALDS-Net)

## 1. Objective & Architecture Design
- **Score-Matching Latent Diffusion Head:**
  * Forward diffusion process: $q(z_t \mid z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) I)$ across $T=20$ discrete diffusion steps.
  * Reverse denoising process: 4-step DDIM deterministic posterior sampling.
- **Conditioning Stream ($X_c \in \mathbb{R}^{T \times 63}$):**
  * 42-dim canonical phone one-hot + manner/place embedding
  * 2-dim CMUDict forward/backward transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$
  * 16-dim learned L1 dialect embedding (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese)
- **Acoustic Representation ($X_a \in \mathbb{R}^{T \times 67}$):**
  * 67-dim base HRTD sub-frame micro-dynamics
- **Gap Representation ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * 13-dim gap vector with triad category embedding (pause, cluster, coda)

## 2. Experimental Steps
1. Map each utterance to its L1 dialect cohort.
2. Build 63-dim conditioning prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train DP-ALDS-Net on GPU (GTX 1650) for 10 epochs using AdamW and multi-task score-matching loss + diagnostic classification loss.
4. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Evaluate on `dev_search` and `dev_replication`.
6. Audit against H70 falsification criteria.
7. Save artifacts, weights, metrics, and checksums under `experiments/EXP-75-DIALECT-CONDITIONED-DIFFUSION-SAMPLER/`.
