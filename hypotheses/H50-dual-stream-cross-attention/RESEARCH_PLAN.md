# Research Plan: Experiment EXP-55 (Dual-Stream Audio-Prompt Cross-Attention Network - DSCA-Net)

## 1. Objective & Model Architecture
Design, implement, train, and benchmark **DSCA-Net** with decoupled acoustic and prompt streams:
- **Stream A (Acoustic Stream, 67 dims):**
  - Continuous 5ms sub-frame micro-dynamics + prosodic energy + Wav2Vec2 acoustic posteriors.
  - Encoder: 2-layer BiLSTM (96 hidden units) + Linear projection $\rightarrow 128$ dimensions.
- **Stream P (Prompt Stream, 39 + 3 dims = 42 dims):**
  - Canonical phoneme embedding (39 dims) + Articulatory priors (manner, place, voicing).
  - Encoder: 2-layer MLP ($42 \rightarrow 64 \rightarrow 128$) with LayerNorm and GELU.
- **Bidirectional Multi-Head Cross-Attention (MHCA):**
  - Number of heads: 4, dimension: 128.
  - Direction 1: Acoustic query attends to prompt keys/values: $A_{\text{cross}} = \text{MHCA}(A, P, P)$.
  - Direction 2: Prompt query attends to acoustic keys/values: $P_{\text{cross}} = \text{MHCA}(P, A, A)$.
  - Monotonic Gaussian Attention bias ($\sigma = 3.0$) applied to alignment matrices.
- **Fusion & Diagnosis:**
  - $F = \text{LayerNorm}(\text{Linear}([A + A_{\text{cross}}, P + P_{\text{cross}}], 128))$.
  - Decoupled Hierarchical Vowel and Consonant diagnosis heads.
  - Gap Insertion network.

## 2. Experimental Steps
1. Re-use 5ms sub-frame sequence features from `EXP-48` across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Extract canonical prompt phone representations and articulatory indices.
3. Train DSCA-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), multi-task articulatory loss, and focal cross-entropy loss.
4. Calibrate operating thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H50 falsification criteria.
