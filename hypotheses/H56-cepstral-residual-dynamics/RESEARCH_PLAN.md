# Research Plan: Experiment EXP-61 (Ultra-High-Resolution Cepstral Residual Network - UCR-Net)

## 1. Technical Formulation & Feature Extraction
Design, implement, train, and benchmark **UCR-Net**:
- **Acoustic Cepstral Residual Extraction:**
  - Compute 13-dimensional MFCCs directly from 16kHz audio waveforms using 25ms window, 10ms hop.
  - Compute empirical canonical phone centroids $\bar{C}_{\text{canonical}}(p) \in \mathbb{R}^8$ (first 8 coefficients excluding $C_0$) strictly on `TRAIN` using canonical prompt text.
  - For every phone slot $k$, calculate the residual deviation:
    $$\Delta_{\text{cep}}(k) = \text{MFCC}_{1-8}(k) - \bar{C}_{\text{canonical}}(p_k)$$
  - Normalize residuals using global standard deviations from `TRAIN`.
- **Stream A (Acoustic Stream, 75 dims):**
  - 67 base features (51 base + 4 tempo + 8 prosodic/phonotactic + 4 micro-dynamics 5ms) + 8 normalized cepstral residuals $\Delta_{\text{cep}} = 75$ dims.
- **Stream P (Canonical Prompt Stream, 42 dims):**
  - Canonical phoneme one-hot (39 dims) + Articulatory priors (manner, place, voicing).
- **Core Architecture:**
  - DSCA-Net backbone: Bidirectional Multi-Head Cross-Attention (MHCA, 4 heads) + Monotonic Local Gaussian Attention ($\sigma = 3.0$) + Decoupled Vowel/Consonant Heads.

## 2. Experimental Steps
1. Compute canonical phone MFCC centroids strictly on `train.jsonl` waveforms.
2. Augment sequence features to 75 dimensions across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train UCR-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$).
4. Calibrate decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H56 falsification criteria.
