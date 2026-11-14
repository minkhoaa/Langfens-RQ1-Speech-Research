# Research Plan: Hypothesis H37 / Experiment EXP-42

## 1. Objective
Design, extract, train, calibrate, and evaluate the **Vocal Tract Length Normalized Cascaded Network (VTLN-HTC-Net)** under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **VTLN Feature Computation:**
   - For each utterance:
     * Compute frame-level spectral centroid using short-time Fourier transform (STFT):
       $C_t = \frac{\sum_k f_k |X_t(k)|}{\sum_k |X_t(k)| + \epsilon}$.
     * Average over voiced frames to obtain speaker-utterance spectral centroid $\bar{C}_{\text{utt}}$.
     * Compute reference adult centroid $\bar{C}_{\text{ref}} = 1,850\text{ Hz}$.
     * Compute VTLN warp factor: $\alpha_{\text{vtln}} = \bar{C}_{\text{ref}} / (\bar{C}_{\text{utt}} + 1\text{e}-6)$.
     * Compute normalized relative centroid: $C_{\text{rel}} = \bar{C}_{\text{utt}} / \bar{C}_{\text{ref}}$.
     * Compute sub-band spectral ratio (high frequency power $>2500$ Hz vs $<1000$ Hz).
   - Append these 3 VTLN features to the 63-dim master features $\to$ 66-dimensional target vector.
2. **Model Architecture (VTLN-HTC-Net):**
   - $D_t = 66, D_g = 11$.
   - Input projection ($66 \to 128$) + 2-layer BiLSTM ($hidden=96$) + Monotonic Local Attention ($\sigma=3.0$).
   - Dual cascaded heads: `vowel_head` (formant substitutions) and `cons_head` (plosive omissions).
3. **Training & Calibration:**
   - Train on `TRAIN` for 30 epochs with AdamW and Cosine Annealing on GPU.
   - Calibrate thresholds on `DEV_CAL`.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log metrics, PR curves, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save model checkpoints, raw predictions, metrics, logs, and SHA-256 hashes under `experiments/EXP-42-AGE-INVARIANT-FRONTEND-NORMALIZATION/`.
