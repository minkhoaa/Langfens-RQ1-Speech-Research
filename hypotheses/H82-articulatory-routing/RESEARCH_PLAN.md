# Research Plan: Experiment EXP-87 (Phonological-Feature Attention Routing Network - FAR-Net)

## 1. Objective & Mathematical Architecture
- **Articulatory Feature Sub-Spaces:**
  * **Manner Sub-Space ($d_m = 48$):** Projects acoustic features corresponding to plosive release bursts, silence contrast, and RMS energy dynamics into specialized manner representation $h_m$.
  * **Place Sub-Space ($d_p = 48$):** Projects acoustic features corresponding to spectral centroid, formant transitions, and spectral tilt into specialized place representation $h_p$.
  * **Voicing Sub-Space ($d_v = 32$):** Projects acoustic features corresponding to ZCR, low-frequency pitch harmonics, and VOT into specialized voicing representation $h_v$.
- **Gated Mixture-of-Experts (MoE) Routing Layer:**
  $$[g_m, g_p, g_v] = \text{Softmax}\left(W_{\text{gate}} \cdot [h_{\text{acoustic}}, h_{\text{prompt}}] + b_{\text{gate}}\right)$$
  $$h_{\text{routed}} = \text{LayerNorm}\left(W_m (g_m \cdot h_m) + W_p (g_p \cdot h_p) + W_v (g_v \cdot h_v)\right)$$
- **Acoustic Stream ($X_a \in \mathbb{R}^{T \times 67}$):**
  * 67-dim base HRTD sub-frame micro-dynamics
- **Prompt Stream ($X_p \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone + articulatory representation
  * 2-dim CMUDict forward/backward transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$ from TRAIN
- **13-Dimensional Triad Decoupled Gap Gating ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Intra-cluster, coda, and pause transitions.

## 2. Experimental Steps
1. Construct 67-dim acoustic sequences and 47-dim IPTM prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train FAR-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and multi-task loss (diagnostic CE + manner/place/voicing multi-task heads + triad gap BCE).
3. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H82 falsification criteria.
6. Save artifacts, weights, metrics, and checksums under `experiments/EXP-87-PHONOLOGICAL-FEATURE-ATTENTION-ROUTING/`.
