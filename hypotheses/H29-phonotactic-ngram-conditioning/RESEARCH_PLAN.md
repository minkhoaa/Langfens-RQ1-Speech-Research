# Research Plan: Hypothesis H29 / Experiment EXP-34

## 1. Objective
Extract an English phonotactic bigram/trigram transition prior from CMUDict, train the **Phonotactic N-Gram Conditioned Diagnostic Network (PNC-Net)** on `TRAIN`, calibrate on `DEV_CAL`, and evaluate on `DEV_SEARCH` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Phonotactic Transition Model:**
   - Parse canonical pronunciations from `cmudict.dict` (already held original source) and `TRAIN` split.
   - Compute smoothed phone bigram matrix $P(p_t \mid p_{t-1})$ and backward matrix $P(p_t \mid p_{t+1})$.
   - Compute cluster consonant transition probabilities for inter-phonemic gaps.
2. **Feature Formulation:**
   - For each target phone slot:
     * Forward transition log-probability: $\log P(p_t \mid p_{t-1})$.
     * Backward transition log-probability: $\log P(p_t \mid p_{t+1})$.
     * Positional flag: is onset (slot 0) or coda (last slot).
   - For each inter-phonemic gap:
     * Cluster transition probability: $\log P(p_{t+1} \mid p_t)$.
     * Is consonant-consonant cluster flag.
3. **Model Architecture (PNC-Net):**
   - Target input: $60 + 3 = 63$-dim vector.
   - Gap input: $9 + 2 = 11$-dim vector.
   - BiLSTM + Monotonic Local Attention + Articulatory Multi-Task Heads.
4. **Training & Evaluation:**
   - Train on `TRAIN` (30 epochs).
   - Calibrate thresholds on `DEV_CAL`.
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log metrics, PR curves, ECE, FWR, and overall Utility.
