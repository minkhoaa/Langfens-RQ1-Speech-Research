# Data Leakage & Evaluation Integrity Audit

This protocol guarantees strict zero-leakage standards across all training, development, calibration, and evaluation operations.

---

## 1. Zero-Leakage Invariants

1. **Speaker Disjointness:** No speaker appearing in `TRAIN` or `DEV_SEARCH` or `DEV_CAL` may ever appear in `DEV_REPLICATION` or `TEST_SEALED`.
2. **Feature Leakage Prevention:**
   - Normalization statistics (mean, variance, mel-spectrogram baselines) must be fitted strictly on the `TRAIN` split only.
   - Foundation model embeddings must be computed purely causally or per-utterance without batch-wide cross-utterance normalization.
3. **Prompt & Utterance Independence:**
   - Utterances are strictly partitioned by speaker groups. Prompts spoken across multiple speakers must not share any speaker-dependent acoustic representations.
4. **Oracle Label Prohibition at Inference:**
   - Evaluator verifies that the inference function receives strictly: `(audio_waveform, prompt_text)`.
   - Any access to ground truth labels, TextGrids, timestamps, or phone error annotations during prediction immediately raises an `ORACLE_LEAKAGE_EXCEPTION` and invalidates the run.
5. **Threshold & Calibration Separation:**
   - Operating thresholds ($\tau_{\text{INS}}$, class prior weights $w_c$) must be calibrated solely on `DEV_CAL`.
   - Never fit or select thresholds on `DEV_REPLICATION` or `TEST_SEALED`.

---

## 2. Leakage Test Suite

Automated verification tests in `evaluation/tests/test_leakage.py`:
- `test_speaker_disjointness`: Asserts intersection of speaker IDs across partitions is strictly empty.
- `test_inference_signature`: Introspects model `predict()` signatures to verify no reference realization or alignment inputs exist.
- `test_mismatched_audio_control`: Feeds randomized/silent audio and asserts predictions collapse to uninformed prior rather than tracking ground truth.
