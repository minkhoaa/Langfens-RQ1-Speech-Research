# LANGFENS — RQ1-PREP-01 / 01-Ingestion Pipeline and Parsers

**Status:** COMPLETED

**Recorded at:** 2026-10-23T15:00:00+07:00

**Evidence source:** Commits `7aa50f7`, `b3a8dae`, `312132b`, `manifests/RQ1-PREP-01/DATA_PREPARATION_REPORT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Implement the core audio ingestion, parsing, normalization, and PyTorch dataset modules under `src/langfens_prep/` with test-label blindness and safe in-memory audio extraction.

## 2. Inputs & Dependencies
- Specifications: `manifests/RQ1-STD-01/` (label schema, phoneme mapping, audio prep).
- Frozen splits: `manifests/RQ1-SPLIT-01/`.
- Raw dataset archives.

## 3. Work Performed
1. Implemented `textgrid_parser.py` parsing intervals into correct, substitution, deletion, and addition categories.
2. Implemented `phoneme_normalizer.py` folding 39-ARPAbet symbols, handling stress markers and AX schwa.
3. Implemented `audio_loader.py` featuring polyphase sinc resampling (16 kHz) directly from ZIP streams.
4. Implemented `dataset.py` and `collate.py` supporting variable-length batch collation and test-label masking.

## 4. Results & Verification
- Acceptance gate: Unit test suite passed (`tests/test_textgrid_parser.py`, `tests/test_audio_loader.py`).
- Audio loaded with 100% bit-exact 16 kHz float32 normalization.
- Variable-length collator correctly applies `-100` padding and 0-blank padding.

## 5. Output Artifacts
- Source code: `src/langfens_prep/phoneme_normalizer.py`, `textgrid_parser.py`, `audio_loader.py`, `dataset.py`, `collate.py`.

## 6. Restart / Recovery Instructions
- Run unit test suite:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_textgrid_parser.py tests/test_phoneme_normalizer.py
  ```
- Lightweight CPU tests; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Full census execution and worker multiprocessing concurrency fix.
- Prohibited actions: Never leak evaluation labels into dataset items during test mode.
- Next step: [02-full-corpus-census-and-profiling.md](02-full-corpus-census-and-profiling.md).
