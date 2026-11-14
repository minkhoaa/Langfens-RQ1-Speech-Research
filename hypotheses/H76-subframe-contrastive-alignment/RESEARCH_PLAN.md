# Research Plan: Experiment EXP-81 (Subframe Temporal Contrastive Aligner Network - STCA-Net)

## 1. Objective & Mathematical Architecture
- **Multi-Resolution Temporal Pyramid Stream ($X_a \in \mathbb{R}^{T \times 72}$):**
  * 67-dim base HRTD (5ms plosive dynamics, Charsiu SSL)
  * 5-dim temporal pyramid (Level 0: 2.5ms micro-burst crest factor to Level 3: 20ms context)
- **Prompt Stream with Interlanguage Priors ($X_p \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone + articulatory representation
  * 2-dim CMUDict forward/backward phonotactic transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$ from TRAIN
- **Temporal Contrastive Boundary Alignment Head:**
  * Bidirectional Multi-Head Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$)
  * 2-layer BiLSTM ($hidden=96$)
  * Boundary sharpening loss penalizing fuzzy phonetic transitions:
    $$\mathcal{L}_{\text{bound}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \text{ReLU}\left(1.5 - \|z_{t+1} - z_t\|_2^2\right) \cdot \mathbb{I}[p_t \neq p_{t+1}]$$
    $$\mathcal{L}_{\text{smooth}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|z_{t+1} - z_t\|_2^2 \cdot \mathbb{I}[p_t = p_{t+1}]$$
- **13-Dimensional Triad Gap Stream ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Triad category conditioning (pause, cluster, coda).

## 2. Experimental Steps
1. Construct 72-dim acoustic sequences and 47-dim prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train STCA-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and multi-task loss $\mathcal{L} = \mathcal{L}_{\text{diag}} + 0.5 \mathcal{L}_{\text{gap}} + 0.1 \mathcal{L}_{\text{bound}} + 0.05 \mathcal{L}_{\text{smooth}}$.
3. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H76 falsification criteria.
6. Save artifacts, weights, metrics, and checksums under `experiments/EXP-81-SUBFRAME-TEMPORAL-CONTRASTIVE-ALIGNER/`.
