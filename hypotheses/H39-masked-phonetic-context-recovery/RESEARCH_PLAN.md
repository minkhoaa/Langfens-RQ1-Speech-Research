# Research Plan: Hypothesis H39 / Experiment EXP-44

## 1. Objective
Design, train, calibrate, and evaluate the **Dynamic Masked Context Training Network (DMCT-Net)** under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Dynamic Masking Implementation:**
   - Initialize trainable mask token $\mathbf{e}_{\text{mask}} \in \mathbb{R}^{63}$.
   - During training, for each utterance batch, randomly sample $15\%$ of active phoneme slots and replace their feature vectors with $\mathbf{e}_{\text{mask}}$.
   - Add an auxiliary reconstruction head: $\text{Linear}(128 \to 39)$ predicting the masked canonical phoneme identity.
   - Total Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vowel}} + \mathcal{L}_{\text{cons}} + 0.20 \mathcal{L}_{\text{recon}} + 0.15 \mathcal{L}_{\text{manner}} + 0.15 \mathcal{L}_{\text{place}} + 0.80 \mathcal{L}_{\text{gap}}$.
2. **Model Training:**
   - Train on `TRAIN` for 30 epochs with AdamW and Cosine Annealing on GPU.
3. **Threshold Calibration on DEV_CAL:**
   - Calibrate thresholds on `DEV_CAL` without masking ($p_{\text{mask}} = 0$).
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` without masking.
   - Log metrics, PR curves, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save checkpoint, predictions, metrics, and SHA-256 manifest under `experiments/EXP-44-SELF-SUPERVISED-DYNAMIC-MASKING/`.
