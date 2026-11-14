# Research Plan: Hypothesis H04 (BiLSTM-CPN)

1. **Utterance Sequence Data Preparation:**
   - Group token-level features by `utterance_id` to form variable-length sequence tensors:
     - Target phone sequence $\mathbf{X}_{\text{target}} \in \mathbb{R}^{B \times N_{\max} \times 51}$
     - Target label sequence $\mathbf{Y}_{\text{target}} \in \mathbb{Z}^{B \times N_{\max}}$
     - Gap sequence $\mathbf{X}_{\text{gap}} \in \mathbb{R}^{B \times (N_{\max}+1) \times 6}$
     - Gap label sequence $\mathbf{Y}_{\text{gap}} \in \mathbb{Z}^{B \times (N_{\max}+1)}$
     - Sequence mask to handle variable utterance lengths.
2. **Model Architecture (`BiLSTM_CPN`):**
   - Target branch:
     - Input projection: Linear(51, 128) + LayerNorm + GELU
     - Bidirectional LSTM: 2 layers, hidden size 128 (bidirectional output 256)
     - Residual fusion: Linear(256 + 51, 128) + LayerNorm + GELU
     - Classifier head: Linear(128, 3) (CORRECT, SUBSTITUTION, OMISSION)
   - Gap branch:
     - BiLSTM output fusion: at gap $j$, concatenate gap features (6 dims) with surrounding phone representations $\mathbf{h}_{j-1}$ and $\mathbf{h}_j$ (128 dims each) $\implies$ total 262 dims.
     - Gap classifier head: Linear(262, 64) + GELU + Linear(64, 1)
3. **Training Protocol:**
   - Sequence-masked focal loss with class weighting $\alpha = [1.0, 1.5, 2.0]$.
   - Batch size 32 utterances (with padding).
   - Optimizer: AdamW, lr=1e-3, cosine decay over 25 epochs.
4. **Calibration & Evaluation:**
   - Calibrate on `DEV_CAL` (300 utterances) with Pareto threshold sweep.
   - Evaluate on `DEV_SEARCH` (599 utterances) with primary locked evaluator.
