# Research Plan: Experiment EXP-91 (Boundary-Aware Phonotactic Posterior Mapper Network - BAPM-Net)

## 1. Objective & Architecture Design
- **Acoustic Boundary Certainty Estimator ($\alpha_{\text{gate}}(k) \in [0.1, 1.0]$):**
  $$\alpha_{\text{gate}}(k) = \sigma\left(W_c \cdot [r_{\text{dur}}(k), \log(r_{\text{dur}}(k) + 1e-4), C_{2.5\text{ms}}(k), \Delta e_{2.5\text{ms}}(k), \text{tilt}(k)] + b_c\right)$$
  Computes continuous acoustic boundary uncertainty from slot duration ratios and 2.5ms micro-burst crest dynamics.
- **Adaptive Prompt Prior Modulator:**
  $$X_p(k) = [X_{\text{canonical}}(k), \alpha_{\text{gate}}(k) \cdot X_{\text{phonotactic}}(k), \alpha_{\text{gate}}(k) \cdot \pi(p_k)]$$
  Disables prior hallucination on clear steady-state vowels while engaging strong interlanguage prior disambiguation on rapid consonant transitions.
- **Acoustic Stream ($X_a \in \mathbb{R}^{T \times 72}$):**
  * 67-dim base HRTD (5ms plosive dynamics, prosody, Charsiu SSL)
  * 5-dim temporal pyramid (Level 0: 2.5ms micro-burst crest factor to Level 3: 20ms context)
- **Bidirectional Audio-Prompt Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$):**
  * Followed by 2-layer BiLSTM ($hidden=96$) and diagnostic heads.
- **13-Dimensional Triad Decoupled Gap Gating ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Intra-cluster, coda, and pause transitions.

## 2. Experimental Steps
1. Construct 72-dim temporal pyramid acoustic sequences and 47-dim adaptive prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train BAPM-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and class-weighted multi-task loss.
3. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H86 falsification criteria.
6. Save artifacts, weights, metrics, and checksums under `experiments/EXP-91-BOUNDARY-AWARE-PHONOTACTIC-POSTERIOR-MAPPER/`.
