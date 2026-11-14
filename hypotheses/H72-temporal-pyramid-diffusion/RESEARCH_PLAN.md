# Research Plan: Experiment EXP-77 (Temporal-Pyramid Diffusion-Coupled Network - TPD-Net)

## 1. Objective & Architecture Design
- **71-Dimensional Multi-Resolution Acoustic Stream ($X_a \in \mathbb{R}^{T \times 71}$):**
  * 67-dim base HRTD (5ms dynamics, duration ratios, Charsiu SSL)
  * 4-dim 2.5ms micro-burst temporal pyramid features:
    1. $e_{2.5\text{ms}, 1}$: RMS energy in first 40 samples (2.5ms window)
    2. $e_{2.5\text{ms}, 2}$: RMS energy in second 40 samples (2.5ms to 5.0ms window)
    3. $\Delta e_{\text{micro}}$: normalized energy differential between first and second 2.5ms micro-frames
    4. $C_{\text{micro}}$: micro-crest factor $\max |x_{2.5\text{ms}}| / e_{2.5\text{ms}}$
- **47-Dimensional Conditioning Stream ($X_c \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone one-hot + manner/place embedding
  * 2-dim CMUDict forward/backward phonotactic transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$
- **Score-Matching Diffusion Head:**
  * Forward diffusion process: 50 steps with cosine schedule.
  * Reverse denoising process: 4-step DDIM deterministic posterior sampling.
- **13-Dimensional Triad Gap Stream ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Triad category conditioning (pause, cluster, coda).

## 2. Experimental Steps
1. Construct 71-dim temporal pyramid acoustic sequences and 47-dim IPTM prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train TPD-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and multi-task score-matching loss + diagnostic loss.
3. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H72 falsification criteria.
6. Save artifacts, weights, metrics, and checksums under `experiments/EXP-77-TEMPORAL-PYRAMID-DIFFUSION-COUPLED-NET/`.
