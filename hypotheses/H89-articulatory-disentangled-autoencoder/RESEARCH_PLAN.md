# Research Plan: Experiment EXP-94 (Articulatory Disentangled Autoencoder - ADAE-Net)

## 1. Architecture Design
- **Acoustic Backbone:** 72-dimensional multi-resolution temporal pyramid acoustic stream.
- **Disentangled Latent Projection:**
  $$z = [z_{\text{manner}} (32), z_{\text{place}} (32), z_{\text{voicing}} (16), z_{\text{timbre}} (48)] \in \mathbb{R}^{128}$$
- **Reconstruction Decoder:** Multi-layer perceptron reconstructing the original 72-dim acoustic input from $z$.
- **Loss Function:**
  $$\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.3 \mathcal{L}_{\text{recon}} + 0.1 \mathcal{L}_{\text{ortho}} + 0.5 \mathcal{L}_{\text{gap}}$$
  where $\mathcal{L}_{\text{ortho}}$ penalizes cross-subspace covariance.
- **Classification Heads:**
  Manner logits from $z_{\text{manner}}$, Place logits from $z_{\text{place}}$, Voicing from $z_{\text{voicing}}$, and joint diagnostic predictions from fused $z$.

## 2. Experimental Execution
1. Implement ADAE-Net in `methods/M94-articulatory-autoencoder-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine scheduling.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H89 falsification criteria.
6. Save artifacts, weights, metrics, and checksums in `experiments/EXP-94-ARTICULATORY-DISENTANGLED-AUTOENCODER/`.
