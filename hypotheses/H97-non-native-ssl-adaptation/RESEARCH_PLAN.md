# Research Plan: Experiment EXP-102 (Parameter-Efficient Low-Rank Acoustic Adapter - PEAR-Net)

## 1. Architecture Design
- **Low-Rank Bottleneck:**
  $W_{\text{down}} \in \mathbb{R}^{8 \times 72}$, $W_{\text{up}} \in \mathbb{R}^{72 \times 8}$.
  Initialized with Kaiming uniform for $W_{\text{down}}$ and zero initialization for $W_{\text{up}}$ (guaranteeing exact identity mapping at initialization: $\Delta x_a = 0$).
  Scaling factor $\alpha = 16.0$, so $\frac{\alpha}{r} = 2.0$.
- **Adapted Input:**
  $$x_{\text{adapted}} = x_a + 2.0 \cdot (W_{\text{up}} W_{\text{down}} x_a) \in \mathbb{R}^{72}$$
- **Acoustic Encoder & Diagnostic Subspaces:**
  ADAE-Net backbone: BiLSTM (96 hidden), projection (128 dims), orthogonal manner (32), place (32), voicing (16), and timbre (48) with Frobenius covariance penalty.
- **Canonical Prompt Encoder:** 47-dimensional IPTM prompt stream.
- **Triad Gap Insertion Head:** 13-dim gap features + 3 gap categories.

## 2. Experimental Execution
1. Implement `methods/M102-ssl-lora-adapter-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW, cosine annealing, and class weighting.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H97 falsification criteria.
6. Save artifacts, weights, metrics, and checksums in `experiments/EXP-102-NON-NATIVE-SELF-SUPERVISED-ADAPTATION/`.
