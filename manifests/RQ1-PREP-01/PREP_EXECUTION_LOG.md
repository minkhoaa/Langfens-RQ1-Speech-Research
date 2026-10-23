# LANGFENS — RQ1-PREP-01: Data Preparation Execution Log

**Execution Gate:** RQ1-PREP-01 — Pipeline Implementation & Data Preparation  
**Log Type:** Chronological Engineering & Audit Trail  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/PREP_EXECUTION_LOG.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`

---

## 1. Chronological Execution Trace

### Step 1: Workspace & Preflight Verification
- **Timestamp (UTC):** `2026-10-23T07:45:00Z` (`14:45:00 +07:00`)
- **Action:** Inspected workspace `kltn_gdrive:KLTN-Dataset/` and local directories. Verified presence of raw archives `l2arctic_release_v5.0.zip` (7,515,454,814 bytes) and `speechocean762.tar.gz` (520,810,923 bytes).
- **Integrity Audit:** Executed SHA-256 validation on frozen split manifests in `manifests/RQ1-SPLIT-01/`:
  - `l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (PASS)
  - `l2arctic_loso_vietnamese_splits.json`: `7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118` (PASS)
  - `speechocean762_official_splits.json`: `bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17` (PASS)
- **Outcome:** Split files match immutable reference checksums to the byte.

### Step 2: Split Provenance Time Erratum Investigation
- **Timestamp (UTC):** `2026-10-23T07:50:00Z`
- **Action:** Investigated the report of a conflict between `"generation_timestamp_utc": "2026-10-23T13:30:00Z"` in split manifests and Google Drive upload timestamps (`2026-10-23 13:26:52 +07:00` / `06:26:52Z` UTC).
- **Finding:** Host runs in ICT (UTC+07:00). Script `audit_scripts/generate_rq1_splits.py:28` statically assigned `DEFAULT_TIMESTAMP_UTC = "2026-10-23T13:30:00Z"`, conflating 13:30 local planning time with UTC.
- **Payload Verification:** Certified that `payload_sha256` is calculated strictly over data partitions via UTF-8 sorted-key serialization (`json.dumps(..., sort_keys=True)`), rendering data integrity completely independent of header metadata.
- **Deliverables:** Generated `manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md`. Frozen manifests preserved unmodified.

### Step 3: Phase A & B — Annotation Parsers & Non-Destructive Normalization
- **Timestamp (UTC):** `2026-10-23T08:00:00Z`
- **Action:** Built `src/langfens_prep/phoneme_normalizer.py`, `src/langfens_prep/textgrid_parser.py`, and `src/langfens_prep/speechocean_parser.py` using strict TDD.
- **Tests Implemented:** `tests/test_normalizer.py`, `tests/test_parsers.py`.
- **Validation:**
  - 39-ARPAbet mapping, stress stripping, `AX` folding to `AH` with `was_ax=True`.
  - Keyboard slip `V8` corrected to `V` with `is_typo_correction=True`.
  - Praat TextGrid parsing: clean substitutions, base-preserving allophones, category-shifting accented, deletions, epenthetic additions, silence.
  - SpeechOcean762 parser: reads official `scores.json` and Kaldi metadata, strictly excluding 245 unmanifested files.
- **Test Result:** 8 / 8 unit tests passed.

### Step 4: Phase C — Safe Audio Loading, Resampling & Staging
- **Timestamp (UTC):** `2026-10-23T08:15:00Z`
- **Action:** Implemented `src/langfens_prep/audio_loader.py` with memory-bounded nested ZIP handle caching, SoundFile float32 mono decoding, and Torchaudio polyphase sinc bandlimited resampling (44.1 kHz $\to$ 16 kHz).
- **Tests Implemented:** `tests/test_audio.py`.
- **Validation:** Tested safe loading directly from nested ZIP without expanding 7.5 GB archive to disk; validated peak normalization and DC-offset removal interfaces; verified zero NaNs and Infs.
- **Test Result:** 3 / 3 unit tests passed.

### Step 5: Phase D — Manifest-Driven Dataset & Collation
- **Timestamp (UTC):** `2026-10-23T08:25:00Z`
- **Action:** Implemented `src/langfens_prep/dataset.py` (`L2ArcticMDDDataset`, `SpeechOcean762ScoreDataset`) and `src/langfens_prep/collate.py` (`LangfensMDDCollate`).
- **Tests Implemented:** `tests/test_dataset_and_loader.py`.
- **Validation:**
  - Dataset instantiation strictly consumes frozen RQ1-SPLIT-01 partitions without generating random splits.
  - Blinded evaluation mode hides `binary_labels` and `ctc_target_indices` to prevent test-set label contamination.
  - Collation pads waveforms with 0.0, labels with -100, and CTC targets with 0 (<BLANK>).
- **Test Result:** 4 / 4 unit tests passed.

### Step 6: Phase E — Full Corpus Census & Reconciliation
- **Timestamp (UTC):** `2026-10-23T08:35:00Z`
- **Action:** Executed full physical census across all 24 speakers and 3,599 gold TextGrids in `l2arctic_release_v5.0.zip` via `src/langfens_prep/validator.py`.
- **Tests Implemented:** `tests/test_full_census.py`.
- **Empirical Findings:**
  - Utterances: **3,599** (Expected: 3,599)
  - Total intervals: **135,890** (Expected: 135,890)
  - Silence intervals: **15,647** (Expected: 15,647)
  - Canonical phonemes: **119,151** (Expected: 119,151)
  - Correct tokens: **101,633** (Expected: 101,633)
  - Substitutions: **14,098** (Expected: 14,098)
    - Clean diff base: **12,364**
    - Clean same base / typo: **4**
    - Base-preserving allophones: **1,112**
    - Category-shifting accented: **360**
    - Distorted / err tags: **258**
  - Deletions: **3,420** (Expected: 3,420)
  - Additions: **1,092** (Expected: 1,092)
  - Strict binary errors: **17,518** (Expected: 17,518)
  - Lenient binary errors: **16,406** (Expected: 16,406)
  - SpeechOcean762: 2,000 train, 500 val, 2,500 test (total 5,000); 245 unmanifested files excluded; 0 partition leakage.
- **Deliverables:** Generated `DATA_CONTRACT_VALIDATION.json`, `DATA_VALIDATION_REPORT.md`, `ANNOTATION_NORMALIZATION_REPORT.md`.

### Step 7: Phase F — Host CPU Micro-Benchmarking
- **Timestamp (UTC):** `2026-10-23T08:45:00Z`
- **Action:** Implemented `src/langfens_prep/benchmark.py` and benchmarked native performance on 12th Gen Intel Core i5-12450H (12 vCPUs, 31 GiB RAM).
- **Measurements:**
  - Direct ZIP read: 1.50 ms/utt vs. NVMe staged read: 0.058 ms/utt ($25.8\times$ speedup).
  - Polyphase sinc resampling (44.1k $\to$ 16k): 0.99 ms/utt.
  - Batch collation (BS=8): 0.37 ms.
  - End-to-end DataLoader throughput: **251.80 utts/sec** (3.97 ms/utt).
- **Deliverables:** Generated `manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md` and `benchmark_results.json`.

---

## 2. Test Suite Execution Summary

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/khoa/Projects/KLTN/Dataset
collected 20 items

tests/test_audio.py::test_load_audio_from_l2arctic_nested_archive PASSED
tests/test_audio.py::test_load_audio_from_speechocean_archive PASSED
tests/test_audio.py::test_configurable_normalization_and_dc PASSED
tests/test_dataset_and_loader.py::test_l2arctic_dataset_loading_and_policy PASSED
tests/test_dataset_and_loader.py::test_l2arctic_blinded_test_isolation PASSED
tests/test_dataset_and_loader.py::test_mdd_batch_collation PASSED
tests/test_dataset_and_loader.py::test_speechocean_score_dataset PASSED
tests/test_full_census.py::test_full_l2arctic_gold_census PASSED
tests/test_full_census.py::test_speechocean762_validation PASSED
tests/test_normalizer.py::test_standard_phoneme_normalization PASSED
tests/test_normalizer.py::test_ax_to_ah_folding PASSED
tests/test_normalizer.py::test_asterisk_and_err_flags PASSED
tests/test_normalizer.py::test_typo_correction_v8 PASSED
tests/test_normalizer.py::test_reversibility PASSED
tests/test_normalizer.py::test_ipa_mapping PASSED
tests/test_parsers.py::test_parse_gold_textgrid_intervals PASSED
tests/test_parsers.py::test_speechocean_parser_manifested PASSED
tests/test_provenance.py::test_verify_split_manifest_hashes PASSED
tests/test_provenance.py::test_canonical_payload_sha256 PASSED
tests/test_provenance.py::test_provenance_timestamp_investigation PASSED

============================= 20 passed in 35.12s ==============================
```
