# Research Plan: Hypothesis H19 (JSA-CTC)

1. **Architecture Formulation:**
   - Multi-Head Cross-Attention Layer:
     - Target query $\mathbf{Q} \in \mathbb{R}^{B \times N \times 128}$
     - Frame keys and values $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times T \times 128}$
     - Soft attention pooling with monotonic positional bias: $\mathbf{H}_{\text{attended}} \in \mathbb{R}^{B \times N \times 128}$
   - Fusion with 55-dim sequence features and articulatory multi-task heads.
   - Gap Insertion Head conditioned on attended boundary transitions.
2. **Training & Calibration:**
   - 30 epochs on `TRAIN` with AdamW and cosine decay.
   - Pareto calibration on `DEV_CAL`.
   - Evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
3. **Packaging & Ledger:**
   - Save artifacts in `experiments/EXP-24-FINAL-ROADMAP-EXPANSION/`.
