# Research Plan: Experiment EXP-85 (Adaptive-Duration Warped Aligner Network - ADWA-Net)

## 1. Objective & Mathematical Architecture
- **Dynamic Gaussian Attention Bandwidth Modulation:**
  $$\sigma_k = \sigma_0 \cdot \text{Softplus}\left(w_{\sigma} \cdot \log(r_{\text{dur}}(k) + 1e-4) + b_{\sigma}\right)$$
  where:
  * $r_{\text{dur}}(k) = d(k) / \bar{d}_{\text{utt}}$ is the normalized duration ratio of phoneme $k$.
  * Local Gaussian attention mask:
    $$M_{i,j} = -\frac{(i - j)^2}{2 \sigma_i^2}$$
    adapts attention span dynamically to prevent energy leakage from prolonged vowels into short consonants while ensuring complete receptive field coverage on extended vowels.
- **Acoustic Stream ($X_a \in \mathbb{R}^{T \times 67}$):**
  * 67-dim base HRTD (5ms plosive dynamics, prosody, duration ratios, Charsiu SSL)
- **Prompt Stream ($X_p \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone + articulatory representation
  * 2-dim CMUDict forward/backward transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$ from TRAIN
- **Bidirectional Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$):**
  * Followed by 2-layer BiLSTM ($hidden=96$) and diagnostic heads.
- **13-Dimensional Triad Decoupled Gap Gating ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Intra-cluster, coda, and pause transitions.

## 2. Experimental Steps
1. Implement `AdaptiveDurationMonotonicLocalAttention` supporting slot-varying Gaussian bandwidth $\sigma_i$.
2. Construct 67-dim acoustic sequences and 47-dim IPTM prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train ADWA-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and class-weighted multi-task loss.
4. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Evaluate on `dev_search` and `dev_replication`.
6. Audit against H80 falsification criteria.
7. Save artifacts, weights, metrics, and checksums under `experiments/EXP-85-ADAPTIVE-DURATION-WARPED-ALIGNER/`.
