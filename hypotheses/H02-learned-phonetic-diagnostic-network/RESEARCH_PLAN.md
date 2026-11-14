# Research Plan: Hypothesis H02

1. **Feature Extraction Pipeline:**
   - Write a caching script `tools/extract_acoustic_features.py` using `charsiu/en_w2v2_fc_10ms`.
   - Extract rich acoustic-phonetic vectors for all tokens in `TRAIN`, `DEV_CAL`, and `DEV_SEARCH`.
   - Feature dimensions per target phone:
     - Phone identity one-hot (39 dims)
     - Canonical logit & posterior
     - Top competitor logit & posterior
     - Pre-softmax margin ($\ell_{\text{target}} - \ell_{\text{comp}}$)
     - Silence logit & posterior
     - Duration in frames & log duration
     - Frame posterior entropy
     - Context features: preceding and succeeding phone margins
   - Feature dimensions per gap:
     - Gap duration in frames
     - Max non-silence phone probability
     - Max non-silence phone logit
     - Silence probability in gap
     - Preceding and following target phone margins
2. **Architecture:**
   - Multi-Layer Perceptron (MLP) with LayerNorm, GELU, and Residual connections.
   - Separate heads for Target Phone Classification (3 classes: CORRECT, SUBSTITUTION, OMISSION) and Gap Insertion Detection (Binary: 0, 1).
   - Loss: Focal Loss with $\gamma = 2.0$ and inverse class frequency weighting $\alpha_c$.
3. **Training & Calibration:**
   - Train on `TRAIN` (1,800 utterances, ~60k phones).
   - Calibrate thresholds on `DEV_CAL` (300 utterances, 2 speakers: ERMS, THV) to satisfy FWR guardrail.
   - Evaluate on `DEV_SEARCH` (599 utterances, 4 speakers: YBAA, NCC, RRBI, YDCK).
4. **Experiment Execution:**
   - Run `EXP-04-LPDN` and evaluate with primary locked evaluator.
