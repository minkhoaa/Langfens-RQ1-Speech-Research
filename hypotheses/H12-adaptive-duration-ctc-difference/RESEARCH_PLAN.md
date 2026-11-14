# Research Plan: Hypothesis H12 (SRAD-Net)

1. **Feature Augmentation:**
   - For each utterance sequence, compute mean duration $\bar{D} = \frac{1}{N} \sum_{i=1}^N d_i$.
   - Compute:
     1. $r_i = d_i / (\bar{D} + 1e-6)$
     2. $\log(r_i + 1e-4)$
     3. $\mathbb{I}(r_i < 0.35)$
     4. Silence contrast $s_i - (1.0 - p_i)$
   - Concatenate onto 51-dim feature vector $\implies 55$ input dims.
2. **Architecture:**
   - `AdaptiveHierarchicalNetwork` (55 input dims, shared 2-layer BiLSTM + articulatory auxiliary heads + diagnostic fusion head).
3. **Training & Calibration:**
   - 30 epochs on `TRAIN` sequences with multi-task loss.
   - Pareto calibration on `DEV_CAL`.
   - Primary locked evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
4. **Artifacts & Ledger:**
   - Persist predictions, metrics, checkpoints, and hashes in `experiments/EXP-17-ADAPTIVE-DURATION-CTC/`.
