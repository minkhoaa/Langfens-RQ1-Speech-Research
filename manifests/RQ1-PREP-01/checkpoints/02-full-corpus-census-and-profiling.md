# LANGFENS — RQ1-PREP-01 / 02-Full Corpus Census and Profiling

**Status:** COMPLETED

**Recorded at:** 2026-10-23T15:15:00+07:00

**Evidence source:** Commits `f84581f`, `98e1072`, `64a1c4d`, `manifests/RQ1-PREP-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Run the full 3,599-utterance census across L2-ARCTIC to reconcile label totals against RQ1-STD-01R, resolve PyTorch DataLoader worker contention, and profile end-to-end CPU audio throughput.

## 2. Inputs & Dependencies
- Implemented package: `src/langfens_prep/`.
- Frozen splits: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`.
- Validator and benchmark scripts: `src/langfens_prep/validator.py`, `src/langfens_prep/benchmark.py`.

## 3. Work Performed
1. Executed full census scan over 3,599 TextGrids and verified invariant prompt denominator.
2. Profiled multi-worker DataLoader execution, identifying process hanging due to shared unpicklable file handles across subprocess forks.
3. Patched `audio_loader.py` and `dataset.py` with lazy per-worker archive handle instantiation (commit `64a1c4d`).
4. Ran CPU micro-benchmarking measuring I/O latency, resampling overhead, and collation speed across batch sizes.

## 4. Results & Verification
- Acceptance gate: `CENSUS_VERIFICATION: PASS` and `BENCHMARK: PASS`.
- Census invariant: 119,151 canonical tokens, 101,633 correct, 17,518 error ($\Delta = 0$).
- Multiprocessing: 4-worker DataLoader loads batches cleanly without deadlocks.
- CPU throughput: 251.8 utterances/sec (3.97 ms/utt).
- Documented in [DATA_VALIDATION_REPORT.md](../DATA_VALIDATION_REPORT.md) and [AUDIO_BENCHMARK_REPORT.md](../AUDIO_BENCHMARK_REPORT.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-PREP-01/DATA_VALIDATION_REPORT.md`, `DATA_CONTRACT_VALIDATION.json`, `AUDIO_BENCHMARK_REPORT.md`, `benchmark_results.json`.
- Bugfix commit: `64a1c4dfix(dataset): instantiate archive readers lazily inside workers to support multiprocessing`.

## 6. Restart / Recovery Instructions
- Re-run census check:
  ```bash
  python -m langfens_prep.validator
  ```
- Fast read-only validation.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: None for data preparation.
- Prohibited actions: Never use non-lazy archive handles in multiprocessing.
- Next phase: [RQ1-TRN-00](../../RQ1-TRN-00/PROCESS_CHECKPOINT.md).
