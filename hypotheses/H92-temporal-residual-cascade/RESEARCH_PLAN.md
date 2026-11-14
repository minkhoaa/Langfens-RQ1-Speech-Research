# Research Plan: Experiment EXP-97 (Multi-Scale Temporal Residual Cascade - MTRC-Net)

## 1. Methodology
- Acoustic inputs: 72-dimensional temporal pyramid features (67 base HRTD + 5 multi-resolution features: 2.5ms crest, 2.5ms delta, 5ms onset, 20ms context, 40ms context).
- Cascaded Residual Gating:
  * Base projection $z_{10} \in \mathbb{R}^{128}$ from frames.
  * Micro-burst residual $\Delta z_{2.5} \in \mathbb{R}^{128}$ gated by $g_0 \in [0, 1]^{128}$.
  * Macro-context residual $\Delta z_{40} \in \mathbb{R}^{128}$ gated by $g_3 \in [0, 1]^{128}$.
  * Fused $z_{\text{cascade}} = z_{10} + g_0 \odot \Delta z_{2.5} + g_3 \odot \Delta z_{40}$.
- Canonical Prompt Stream: 47 dimensions (IPTM prompt features).
- Train on GPU for 10 epochs.
- Calibrate on `dev_cal`.
- Evaluate on `dev_search` and `dev_replication`.
- Audit against H92 falsification criteria.
- Save all artifacts in `experiments/EXP-97-MULTI-SCALE-TEMPORAL-RESIDUAL-CASCADE/`.
