# LANGFENS — RQ1-PREP-01 / Audio Preparation and Dataset Validation

**Status:** COMPLETED

**Recorded at:** 2026-10-23T15:20:00+07:00

**Evidence source:** Commits `085bfcd`, `7aa50f7`, `b3a8dae`, `312132b`, `f84581f`, `98e1072`, `83bbc39`, `64a1c4d`, `manifests/RQ1-PREP-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Implement the modular data preparation software package (`src/langfens_prep/`), perform full 3,599-utterance corpus census validation asserting zero label variance, benchmark host CPU audio ingestion throughput, and ensure non-destructive pipeline operation.

## 2. Inputs & Dependencies
- Source branch: `main` (commit `64a1c4d`).
- Frozen splits: `manifests/RQ1-SPLIT-01/` (SHA-256 `7d6bfdf6...`, `6056d56d...`, `f829ec96...`).
- Raw archives: `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`, `datasets/SpeechOcean762/original/speechocean762.tar.gz`.
- Reference package: `src/langfens_prep/`.

## 3. Work Performed
1. Implemented `src/langfens_prep/` modules: `provenance`, `phoneme_normalizer`, `textgrid_parser`, `speechocean_parser`, `audio_loader`, `dataset`, `collate`, `validator`, and `benchmark`.
2. Implemented polyphase sinc bandlimited resampling (44.1 kHz $\to$ 16 kHz) and safe in-memory audio extraction.
3. Executed full census validation over all 3,599 gold Praat TextGrids of L2-ARCTIC v5.0.
4. Resolved PyTorch DataLoader multiprocessing worker contention by refactoring archive readers for lazy instantiation inside worker processes (commit `64a1c4d`).
5. Conducted empirical CPU micro-benchmarks measuring end-to-end audio ingestion throughput.

## 4. Results & Verification
- Acceptance gate: `RQ1-PREP-01: PASS`.
- Full corpus census: Exactly 119,151 canonical phonemes (101,633 correct, 17,518 error) with zero unexplained variance against `RQ1-STD-01R`.
- SpeechOcean762: All 245 unmanifested extra WAVs strictly excluded from supervised ingestion.
- Empirical CPU benchmark: **251.8 utterances/sec** (3.97 ms/utt) on host CPU, establishing verified data ingestion throughput.
- Documented in [DATA_PREPARATION_REPORT.md](DATA_PREPARATION_REPORT.md) and [DATA_VALIDATION_REPORT.md](DATA_VALIDATION_REPORT.md).

## 5. Output Artifacts
- Source code: `src/langfens_prep/` (10 modules, commit `64a1c4d`).
- Validation records: `manifests/RQ1-PREP-01/DATA_VALIDATION_REPORT.md`, `DATA_CONTRACT_VALIDATION.json`.
- Benchmark records: `manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md`, `benchmark_results.json`.
- Test suite: `tests/test_provenance.py`, `tests/test_phoneme_normalizer.py`, `tests/test_textgrid_parser.py`, `tests/test_dataset.py`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 reference environment (`.venv`).
- Run full census validator:
  ```bash
  python -m langfens_prep.validator
  ```
- Run CPU micro-benchmarking:
  ```bash
  python -m langfens_prep.benchmark
  ```
- Lightweight read-only operations; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Model training architecture and contracts deferred to RQ1-TRN.
- Prohibited actions: Never resample or extract raw dataset archives on disk.
- Next phase: [RQ1-TRN-00](../RQ1-TRN-00/PROCESS_CHECKPOINT.md).
