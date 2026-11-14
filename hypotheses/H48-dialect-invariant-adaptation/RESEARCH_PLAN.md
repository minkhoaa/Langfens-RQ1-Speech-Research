# Research Plan: Experiment EXP-53 (Dialect-Invariant Acoustic Adapter Network - DIAA-Net)

## 1. Objective & Model Architecture
Construct, train, and benchmark **DIAA-Net** augmenting HRTD-Net with utterance-level dialect conditioning:
- **Utterance Dialect Encoder ($e_{\text{dialect}} \in \mathbb{R}^{16}$):**
  - Input: Utterance-level mean and std of 5ms burst energy, spectral centroid, spectral tilt, zero-crossing rate, duration ratios, and sil-contrast ($16$ acoustic features).
  - Architecture: 2-layer MLP ($16 \rightarrow 32 \rightarrow 16$, LayerNorm + GELU).
- **FiLM Feature Modulation:**
  - $\gamma(e_{\text{dialect}}) = 1.0 + \tanh(W_{\gamma} e_{\text{dialect}})$
  - $\beta(e_{\text{dialect}}) = W_{\beta} e_{\text{dialect}}$
  - Injected post-LayerNorm before Monotonic Local Gaussian Attention and Decoupled Diagnosis Heads.
- **Base Architecture:**
  - HRTD-Net (67-dim input, 5ms sub-frame micro-dynamics, 15% dynamic sequence token masking).

## 2. Experimental Steps
1. Compute utterance-level acoustic dialect vectors from raw sequences for `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Warm-start core parameters from `HRTD-Net` (EXP-48) and train DIAA-Net for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$).
3. Calibrate operating thresholds on `DEV_CAL`.
4. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Compute per-L1 breakdown across all 6 L1 cohorts (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese) and calculate $\sigma_{\text{L1}}$.
6. Audit against H48 falsification criteria.
