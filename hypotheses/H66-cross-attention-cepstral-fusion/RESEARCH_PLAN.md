# Research Plan: Experiment EXP-71 (Cross-Attention Spectral-Residual Network - CASR-Net)

## 1. Model Architecture
- **Acoustic Stream ($X_a \in \mathbb{R}^{T \times 75}$):**
  * 67-dim base acoustic feature (Charsiu SSL + 5ms sub-frame plosive release bursts + energy dynamics + duration ratio + formant centroid warp)
  * 8-dim spectral cepstral residuals $\Delta_{\text{cep}} = |c_{\text{learner}} - \bar{c}_{\text{canonical}}|$ computed using canonical MFCC centroids fit strictly on `train.jsonl`
- **Prompt Stream ($X_p \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone one-hot + phonetic manner/place embedding
  * 2-dim CMUDict phonotactic n-gram transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p) = [P(c \mid p), P(s \mid p), P(o \mid p)]$ from TRAIN
- **Gap Stream ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * 13-dim gap acoustic representation with triad category conditioning (pause, cluster, coda)
- **Cross-Modal Attention:**
  * Bidirectional Multi-Head Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$) aligning $X_a$ and $X_p$
  * 2-layer BiLSTM ($hidden=64$) processing concatenated cross-attended representations
  * Manner-gated target diagnostic head (CORRECT, SUBSTITUTION, OMISSION)
  * Triad-gated gap insertion detection head (INSERTION)

## 2. Training & Evaluation Pipeline
1. Compute canonical phone cepstral profiles strictly on `train.jsonl`.
2. Compute empirical interlanguage error transfer priors strictly on `train.jsonl`.
3. Construct 75-dim acoustic sequences and 47-dim prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
4. Train CASR-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and class-weighted multi-task loss.
5. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Freeze operating policy and evaluate on `dev_search` and `dev_replication`.
7. Audit against H66 falsification criteria.
8. Save artifacts, weights, metrics, and checksums under `experiments/EXP-71-CROSS-ATTENTION-SPECTRAL-RESIDUAL-NET/`.
