# Research Plan: Hypothesis H33 / Experiment EXP-38

## 1. Objective
Design, train, calibrate, and evaluate the **Hierarchical Two-Stage Cascaded Network (HTC-Net)** under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Architecture (HTC-Net):**
   - Shared representation backbone:
     * 63-dim input projection ($D \to 128$)
     * 2-layer BiLSTM ($hidden=96 \to 192$)
     * Monotonic Local Gaussian Attention ($\sigma=3.0$)
     * Shared articulatory auxiliary heads (Manner, Place, Voicing)
   - Specialized Diagnostic Heads:
     * `vowel_head`: $\text{Linear}(148 \to 64) \to \text{GELU} \to \text{Dropout} \to \text{Linear}(64 \to 3)$
     * `cons_head`: $\text{Linear}(148 \to 64) \to \text{GELU} \to \text{Dropout} \to \text{Linear}(64 \to 3)$
   - Routing Logic:
     * Target phone identity is known at inference from the prompt text (canonical target $p_t$).
     * If $p_t \in \text{VOWEL}$, prediction is emitted by `vowel_head`.
     * Otherwise, prediction is emitted by `cons_head`.
   - Inter-phonemic gap network:
     * 11-dim gap input + left/right contextual representations $\to 64 \to 32 \to 1$.
2. **Training & Dual Focal Loss:**
   - During training, loss is routed through the corresponding head according to the gold target phone's canonical category.
   - 30 epochs with AdamW (lr=8e-4) and Cosine Annealing on GPU.
3. **Threshold Calibration on DEV_CAL:**
   - Jointly calibrate $(\theta_{\text{subs\_vowel}}, \theta_{\text{omis\_vowel}}, \theta_{\text{subs\_cons}}, \theta_{\text{omis\_cons}}, \theta_{\text{ins}})$.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log metrics, PR curves, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save checkpoint, predictions, metrics, and SHA-256 manifest under `experiments/EXP-38-ENDLESS-SEARCH-HIERARCHICAL-REFINEMENT/`.
