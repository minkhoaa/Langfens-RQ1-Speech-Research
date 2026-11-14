# Research Plan: Experiment EXP-74 (VTLN-Warped Cross-Attention Spectral-Residual Network - V-CASR-Net)

## 1. Objective & Mathematical Formulation
1. **Speaker VTLN Factor Calculation:**
   Extract spectral centroid $\bar{F}_{\text{spk}}$ strictly from unlabelled waveforms for each speaker in the dataset.
   $$\alpha_{\text{spk}} = \frac{\bar{F}_{\text{spk}}}{\bar{F}_{\text{train\_corpus}}}$$
2. **VTLN-Normalized Cepstral Residuals:**
   Extract canonical phone MFCC centroids $\bar{c}(p)$ strictly from `train.jsonl`.
   $$\Delta_{\text{cep, norm}}(k) = \frac{|c_{\text{learner}}(k) - \bar{c}(p_k)|}{\alpha_{\text{spk}}}$$
3. **76-Dimensional Acoustic Stream:**
   Concatenate 67-dim HRTD acoustic vector, 8-dim normalized cepstral residuals, and 1-dim $\alpha_{\text{spk}}$:
   $$X_a = [X_{\text{hrtd}}, \Delta_{\text{cep, norm}}, \alpha_{\text{spk}}] \in \mathbb{R}^{T \times 76}$$
4. **47-Dimensional Prompt Stream:**
   $$X_p = [X_{\text{canonical}}, X_{\text{phonotactic}}, \pi(p)] \in \mathbb{R}^{T \times 47}$$
5. **Bidirectional Multi-Head Cross-Attention (MHCA):**
   Align $X_a$ and $X_p$ via 4-head cross-attention ($d_{\text{model}}=64$) followed by 2-layer BiLSTM and manner/triad-gap gating.

## 2. Experimental Steps
1. Compute speaker VTLN factors and canonical MFCC centroids strictly on `train.jsonl`.
2. Construct 76-dim acoustic sequences and 47-dim prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train V-CASR-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and class-weighted multi-task loss.
4. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Evaluate on `dev_search` and `dev_replication`.
6. Audit against H69 falsification criteria.
7. Save artifacts, weights, metrics, and checksums under `experiments/EXP-74-VTLN-WARPED-CEPSTRAL-ADAPTER/`.
