# Research Plan: Experiment EXP-62 (Domain-Adaptive Multi-Head Cross-Attention Network - DAMH-Net)

## 1. Technical Formulation & Architecture
Design, implement, train, and benchmark **DAMH-Net**:
- **Acoustic Stream (70 dims):**
  - 67 base features (51 base + 4 tempo + 8 prosodic/phonotactic + 4 micro-dynamics 5ms).
  - 3 Vocal Tract Length Normalization features ($\alpha_{\text{vtln}}, C_{\text{rel}}, \log \alpha_{\text{vtln}}$) computed relative to canonical adult reference $C_{\text{ref}} = 1850$ Hz.
  - Linear projection ($70 \rightarrow 128$) + 2-layer BiLSTM (96 hidden units).
- **Prompt Stream (42 dims):**
  - Canonical phoneme one-hot (39 dims) + Articulatory priors (manner, place, voicing).
  - Linear projection ($42 \rightarrow 64 \rightarrow 128$).
- **Bidirectional Multi-Head Cross-Attention (MHCA):**
  - $A_{\text{guided}} = \text{MHCA}(A, P, P)$ and $P_{\text{guided}} = \text{MHCA}(P, A, A)$.
  - Monotonic Gaussian Attention bias ($\sigma = 3.0$).
- **Decoupled Hierarchical Heads:**
  - Vowel diagnosis head and consonant diagnosis head.
  - Gap insertion network.

## 2. Experimental Steps
1. Compute VTLN features across `train`, `dev_cal`, `dev_search`, and `dev_replication` (L2-ARCTIC), as well as `speechocean762.tar.gz` (SpeechOcean762-Test).
2. Train DAMH-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$).
3. Calibrate operating thresholds on `DEV_CAL`.
4. Evaluate in-domain on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Evaluate zero-shot out-of-domain on all 4,536 phonemes of `SpeechOcean762-Test`.
6. Audit against H57 falsification criteria.
