# Research Plan: Hypothesis H23 / Experiment EXP-28

## 1. Objective
Design, extract, train, calibrate, and evaluate the **Acoustic Prosodic-Energy Diagnostic Network (APED-Net)** under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Prosodic Feature Extraction:**
   - For each utterance, load raw 16kHz audio waveform.
   - For each target phone slot (duration $T_{\text{frames}}$ frames at 10ms frame rate = $[t_{\text{start}}, t_{\text{end}}]$):
     * Compute slot RMS energy: $E_{\text{slot}} = \sqrt{\frac{1}{M}\sum_{m=1}^M s^2(m)}$.
     * Compute utterance mean RMS energy: $\bar{E}_{\text{utt}}$.
     * Compute relative energy: $E_{\text{rel}} = E_{\text{slot}} / (\bar{E}_{\text{utt}} + 10^{-6})$.
     * Compute Zero-Crossing Rate (ZCR).
     * Compute Spectral Centroid over the slot.
     * Compute Energy Decay (ratio of energy in the second half of the slot to the first half).
   - Concatenate these 5 prosodic-energy features with the 55-dimensional sequence features to form 60-dimensional target vectors.
   - For gap slots: compute gap relative RMS energy and gap ZCR, appending 2 features to the 6-dim gap vector (8-dim total).
2. **Model Architecture (APED-Net):**
   - 60-dim target input projection + 2-layer BiLSTM ($hidden=96$) + Monotonic Local Attention ($\sigma=3.0$).
   - Multi-task heads: Manner (8-class), Place (9-class), Voicing (3-class), Target Diagnosis (3-class: Correct, Substitution, Omission).
   - Gap network: 8-dim gap input + left/right contextual representations -> 1-dim insertion logit.
3. **Training & Calibration:**
   - Train on `TRAIN` (1800 utterances) for 30 epochs with AdamW (lr=8e-4) and Cosine Annealing.
   - Calibrate thresholds on `DEV_CAL` (300 utterances) via fine grid search.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).
   - Log all metrics, PR curves, ECE, FWR, and Utility score.
5. **Ledger & Artifact Audit:**
   - Save model checkpoints, raw predictions, metrics, logs, and SHA-256 hashes.
