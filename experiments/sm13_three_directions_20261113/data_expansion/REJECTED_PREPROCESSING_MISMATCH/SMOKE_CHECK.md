# Data expansion extraction smoke

This is an interface smoke only; it is not a benchmark result and writes no model or prediction artifact.

- Source: `HQTV_arctic_a0362`, one of the deterministic 50 nonmanifest utterances selected for fold 0 (HQTV is fit-only in fold 0).
- Prompt: `Nope not the slightest idea`.
- Processing: 44.1 kHz audio resampled to 16 kHz; CMUdict prompt phones; frozen CharSiu prompt-phone alignment; frozen Wav2Vec2-base-960h hidden states 0–12.
- Output: 19 target rows, shape `[19, 13, 2304]`, as required by the shared scalar-mixing SM13 target head.
- Runtime: CPU only, PyTorch threads capped at 2. This smoke did not train or calibrate any model and did not inspect labels.
- Command pattern: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY' ...` (inline one-utterance extraction calling `speaker_candidates`, `align_centers`, and `extract_target_windows`).

The full per-fold extraction and candidate benchmark remain subject to the shared exclusive GPU lock.
