# Research Plan: Experiment EXP-104 (Soft-Gold Adjudication Network - SGAN-Net)

## 1. Methodology
- **Empirical Soft Target Formulation:**
  For each training phoneme $k$ with target identity $p_k$:
  $y_{\text{one-hot}} \in \mathbb{R}^3$ ([1, 0, 0] for CORRECT, [0, 1, 0] for SUBSTITUTION, [0, 0, 1] for OMISSION).
  $\pi(p_k) = [P_c, P_s, P_o]$ (empirical error distribution from `train.jsonl`).
  $$y_{\text{soft}}(k) = 0.90 \cdot y_{\text{one-hot}}(k) + 0.10 \cdot \pi(p_k)$$
- **Loss Function:**
  Target diagnostic loss:
  $$\mathcal{L}_{\text{soft\_kl}} = \sum_{c} y_{\text{soft}, c} \log\left(\frac{y_{\text{soft}, c}}{\hat{p}_c + 1e-12}\right)$$
  Total loss:
  $$\mathcal{L} = \mathcal{L}_{\text{soft\_kl}} + 0.6 \mathcal{L}_{\text{gap}} + 0.25 \mathcal{L}_{\text{recon}} + 0.05 \mathcal{L}_{\text{ortho}}$$
- **Architecture:**
  ADAE-Net backbone: 72-dim acoustic stream, BiLSTM (96 hidden), orthogonal latent spaces (manner: 32, place: 32, voicing: 16, timbre: 48), 47-dim canonical IPTM prompt stream.
- **Evaluation:**
  Frozen `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1` on `dev_cal`, `dev_search`, and `dev_replication`.
- Audit against H99 falsification criteria.
- Save all artifacts in `experiments/EXP-104-MULTI-ANNOTATOR-BLENDED-ADJUDICATION/`.
