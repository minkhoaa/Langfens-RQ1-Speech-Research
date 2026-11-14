# Research Plan: Experiment EXP-101 (Vocal Tract Area Function Inversion - VTI-Net)

## 1. Physical Formulation & Feature Extraction
- For each target phone interval $[t_{\text{start}}, t_{\text{end}}]$, extract 16kHz audio waveform segment.
- Apply high-frequency pre-emphasis: $y[n] = x[n] - 0.97 x[n-1]$.
- Compute Levinson-Durbin autocorrelation reflection coefficients $r_k$ for order $K=8$.
- Calculate logarithmic cross-sectional area ratios:
  $$\log A_0 = 0.0$$
  $$\log A_{k+1} = \log A_k + \log\left(\frac{1 - r_k}{1 + r_k + 1e-4}\right), \quad k=0, \dots, 6$$
  $$A_{\text{tract}} = [\log A_0, \log A_1, \dots, \log A_7] \in \mathbb{R}^8$$
- Concatenate with 72-dim temporal pyramid acoustic features -> 80-dim physical-acoustic stream.

## 2. Model Architecture
- **Acoustic Encoder:** 80-dimensional input, BiLSTM (96 hidden), projection to 128 dimensions.
- **Canonical Prompt Encoder:** 47-dimensional IPTM prompt features (with interlanguage error priors).
- **Cross-Attention & Fusion:** Bidirectional MHCA (4 heads) + Monotonic Local Attention ($\sigma=3.0$).
- **Disentangled Latent Spaces:** Orthogonal manner (32), place (32), voicing (16), and timbre (48) with Frobenius covariance penalty.
- **Reconstruction Decoder:** 128 -> 80 dimensions (reconstructing both acoustics and vocal tract area functions).

## 3. Experimental Execution
1. Implement `methods/M101-vocal-tract-inversion-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine learning rate decay.
3. Calibrate on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H96 falsification criteria.
6. Save artifacts, weights, metrics, and checksums in `experiments/EXP-101-VOCAL-TRACT-AREA-FUNCTION-INVERSION/`.
