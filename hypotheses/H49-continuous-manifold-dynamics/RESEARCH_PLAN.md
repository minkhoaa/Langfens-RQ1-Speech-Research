# Research Plan: Experiment EXP-54 (Continuous Latent Manifold Dynamics Network - CLMD-Net)

## 1. Objective & Loss Formulation
Design, train, and benchmark **CLMD-Net**, incorporating continuous articulatory manifold regularization into HRTD-Net:
- **Base Architecture:**
  - 67-dimensional input (51 base + 4 tempo + 8 prosodic/phonotactic + 4 micro-temporal 5ms dynamics).
  - 13-dimensional gap input.
  - 15% dynamic sequence token masking with auxiliary phoneme reconstruction.
  - Monotonic Local Gaussian Attention ($\sigma = 3.0$).
  - Decoupled Hierarchical Vowel and Consonant diagnosis heads.
- **Continuous Manifold Regularizer ($\mathcal{L}_{\text{manifold}}$):**
  - Temporal Smoothness: $\mathcal{L}_{\text{smooth}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \| h_{t+1} - h_t \|_2^2$ (penalizes internal latent chatter).
  - Boundary Margin: $\mathcal{L}_{\text{boundary}} = \frac{1}{N-1} \sum_{k=1}^{N-1} \text{ReLU}(1.0 - \| h_{k, \text{end}} - h_{k+1, \text{start}} \|_2)$ (enforces crisp transitions across phone slots).
  - Total Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + 0.10 \mathcal{L}_{\text{smooth}} + 0.10 \mathcal{L}_{\text{boundary}} + 0.30 \mathcal{L}_{\text{manner}} + 0.40 \mathcal{L}_{\text{recon}} + 0.50 \mathcal{L}_{\text{gap}}$.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features from `EXP-48` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train CLMD-Net on GPU (NVIDIA GTX 1650) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$).
3. Calibrate operating thresholds on `DEV_CAL`.
4. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H49 falsification criteria.
