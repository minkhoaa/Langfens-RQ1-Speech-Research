# Research Plan: Experiment EXP-108 (Formant-Bandwidth Normalized Spectral Residuals - FBNR-Net)

## 1. Physical Formulation & Feature Extraction
- **Formant Bandwidth Estimation:**
  For each phoneme interval $[t_s, t_e]$ in 16kHz audio:
  Compute LPC-12 autocorrelation polynomial roots.
  Identify complex conjugate pole pairs corresponding to $F_1 \in [250, 1100]$Hz and $F_2 \in [900, 2800]$Hz.
  Calculate pole radii $|r_1|, |r_2|$ and bandwidths:
  $$B_1 = -\frac{16000}{\pi} \ln(|r_1| + 1e-6), \quad B_2 = -\frac{16000}{\pi} \ln(|r_2| + 1e-6)$$
  Compute normalized bandwidth factor $b_{\text{scale}} = \sqrt{B_1 \cdot B_2} / 100.0$.
- **Bandwidth-Normalized Cepstral Residuals:**
  $$\tilde{\Delta}_{\text{cep}}(k) = \frac{\Delta_{\text{mfcc}, 1:8}(k)}{b_{\text{scale}}(k) + 0.10} \in \mathbb{R}^8$$
- **Acoustic Stream:**
  Concatenate 72-dim temporal pyramid features + 8-dim bandwidth-normalized residuals -> 80 dims.
- **Encoder & Latent Spaces:**
  ADAE-Net backbone: BiLSTM (96 hidden), projection (128 dims), orthogonal manner (32), place (32), voicing (16), and timbre (48) with Frobenius covariance penalty.
- **Canonical Prompt Encoder:** 47-dimensional IPTM prompt stream.

## 2. Experimental Execution
1. Implement `methods/M108-spectral-residual-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine scheduling.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H103 falsification criteria.
6. Save all artifacts in `experiments/EXP-108-CROSS-ATTENTION-SPECTRAL-NORMALIZED-RESIDUALS/`.
