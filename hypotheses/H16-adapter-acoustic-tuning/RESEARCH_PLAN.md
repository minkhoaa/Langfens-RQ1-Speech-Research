# Research Plan: Hypothesis H16 (NNA-Adapter)

1. **Adapter Architecture:**
   - 2-layer residual bottleneck adapter:
     - Linear(55, 128) + LayerNorm + GELU + Dropout(0.15)
     - Linear(128, 55)
     - Residual skip: $\mathbf{x}_{\text{adapted}} = \mathbf{x} + \alpha_{\text{res}} \cdot \text{Adapter}(\mathbf{x})$
   - Feeds directly into the Hierarchical Articulatory Network (HM-ADN) encoder.
2. **Training Protocol:**
   - Joint optimization of adapter and sequence heads for 30 epochs on `TRAIN` (1,800 utterances).
   - Multi-task articulatory loss: $\mathcal{L}_{\text{diag}} + 0.25 \mathcal{L}_{\text{m}} + 0.25 \mathcal{L}_{\text{pl}} + 0.15 \mathcal{L}_{\text{v}} + 0.5 \mathcal{L}_{\text{gap}}$.
   - AdamW with learning rate 8e-4 and cosine decay.
3. **Calibration & Evaluation:**
   - Pareto calibration on `DEV_CAL`.
   - Primary locked evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
4. **Packaging & Ledger:**
   - Save artifacts in `experiments/EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT/`.
