# Research Plan: Experiment EXP-56 (Conformer High-Resolution Temporal Dynamics Network - CHTD-Net)

## 1. Objective & Model Architecture
Design, implement, train, and benchmark **CHTD-Net**:
- **Stream A (Acoustic Stream, 67 dims):**
  - Linear projection ($67 \rightarrow 128$) + Positional Embedding.
  - 2-block Conformer encoder:
    * Half-step Feedforward 1 ($128 \rightarrow 256 \rightarrow 128$, GELU, dropout 0.15)
    * Multi-Head Self-Attention (4 heads, $d_{\text{model}}=128$, dropout 0.15)
    * Depthwise Convolution (kernel size 5, padding 2, BatchNorm, GELU, pointwise conv)
    * Half-step Feedforward 2 ($128 \rightarrow 256 \rightarrow 128$, LayerNorm)
- **Stream P (Canonical Prompt Stream, 42 dims):**
  - Canonical phoneme embedding (39 dims) + Articulatory priors (manner, place, voicing).
  - Encoder: 2-layer MLP ($42 \rightarrow 64 \rightarrow 128$).
- **Bidirectional Cross-Attention:**
  - $A_{\text{guided}} = \text{MHCA}(A, P, P)$ and $P_{\text{guided}} = \text{MHCA}(P, A, A)$.
  - Monotonic Gaussian Attention bias ($\sigma = 3.0$).
- **Decoupled Hierarchical Heads:**
  - Vowel diagnosis head and consonant diagnosis head.
  - Gap insertion network.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features from `EXP-48` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Ensure strict zero-leakage prompt stream extraction using `get_canonical_articulatory_features`.
3. Train CHTD-Net on GPU (GTX 1650) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), multi-task articulatory loss, and focal cross-entropy loss.
4. Calibrate decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H51 falsification criteria.
