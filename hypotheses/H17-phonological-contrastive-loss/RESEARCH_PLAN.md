# Research Plan: Hypothesis H17 (PSC-Net)

1. **Architecture Formulation:**
   - Base Encoder: 2-layer BiLSTM + 55-dim adaptive duration features.
   - Normalized embedding layer $\mathbf{z}_i = \mathbf{h}_i / \|\mathbf{h}_i\|_2 \in \mathbb{S}^{127}$.
   - Learnable or normalized class prototypes $\mathbf{C} \in \mathbb{R}^{3 \times 128}$ (for CORRECT, SUBSTITUTION, OMISSION).
   - Supervised margin loss with $m = 0.35$.
2. **Multi-Task Objective:**
   $$\mathcal{L} = \mathcal{L}_{\text{diag}} + 0.20 \mathcal{L}_{\text{margin}} + 0.20 \mathcal{L}_{\text{manner}} + 0.20 \mathcal{L}_{\text{place}} + 0.10 \mathcal{L}_{\text{voicing}} + 0.50 \mathcal{L}_{\text{gap}}$$
3. **Training & Calibration:**
   - 30 epochs on `TRAIN` with AdamW (lr=8e-4).
   - Pareto calibration on `DEV_CAL`.
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
4. **Packaging & Ledger:**
   - Save artifacts in `experiments/EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN/`.
