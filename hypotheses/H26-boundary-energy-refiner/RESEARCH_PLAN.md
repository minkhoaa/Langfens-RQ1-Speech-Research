# Research Plan: Hypothesis H26 / Experiment EXP-31

## 1. Objective
Design, extract, train, calibrate, and evaluate the **Boundary Energy Refinement Network (BER-Net)** under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Sub-Phonemic Boundary Feature Extraction:**
   - For each target phone slot, load the 16kHz audio segment.
   - Compute:
     * `crest_factor`: $\max |s(t)| / (\text{RMS} + 1e-6)$ (identifying plosive release spikes).
     * `edge_energy_ratio`: energy in the final 20ms of the slot vs initial 20ms (identifying release bursts vs occlusion closures).
     * `edge_zcr_diff`: ZCR in the final 20ms minus ZCR in the initial 20ms (identifying frication / burst release).
   - Append these 3 features to the 60-dim APED features to form a 63-dimensional target vector.
   - For gaps: compute gap crest factor and append to form a 10-dimensional gap vector.
2. **Model Architecture (BER-Net):**
   - Input: $D_t = 63, D_g = 10$.
   - Projection: $63 \to 128$.
   - 2-layer BiLSTM ($hidden=96 \to 192$).
   - Fusion: $192 + 63 \to 128$.
   - Monotonic Local Gaussian Attention ($\sigma=3.0$).
   - Multi-task heads: Manner (8-class), Place (9-class), Voicing (3-class), Target Diagnosis (3-class).
   - Gap network: $10 + (128 \times 2) \to 64 \to 32 \to 1$.
3. **Training & Calibration:**
   - Train on `TRAIN` for 30 epochs with AdamW (lr=8e-4) and Cosine Annealing.
   - Calibrate operating thresholds on `DEV_CAL`.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log metrics, PR curves, ECE, FWR, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save model checkpoints, raw predictions, metrics, logs, and SHA-256 hashes under `experiments/EXP-31-ENDLESS-SEARCH-BOUNDARY-REFINER/`.
