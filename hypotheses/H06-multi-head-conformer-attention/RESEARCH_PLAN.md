# Research Plan: Hypothesis H06 (Conformer Scorer)

1. **Architecture Definition:**
   - 2-layer Conformer Encoder with 4 attention heads and 1D depthwise convolution (kernel=5, padding=2).
   - Residual connection from input features to output layer.
   - Context-fused gap classifier for insertions.
2. **Training Protocol:**
   - Sequence batching with masking.
   - Focal Loss with $\alpha = [1.0, 1.8, 2.2]$ and $\gamma = 1.5$.
   - 30 epochs with AdamW (lr=8e-4, weight_decay=1e-4) and cosine decay.
3. **Calibration & Evaluation:**
   - Calibrate on `DEV_CAL` via Pareto multi-objective search.
   - Evaluate on `DEV_SEARCH` with primary locked evaluator.
