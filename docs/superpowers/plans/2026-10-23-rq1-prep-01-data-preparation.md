# RQ1-PREP-01: Non-Destructive Data Preparation & Pipeline Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and exhaustively validate the non-destructive speech data preparation pipeline, PyTorch datasets, collation contracts, and CPU benchmarks for Langfens RQ1 (L2-ARCTIC v5.0 and SpeechOcean762).

**Architecture:** A modular Python package `src/langfens_prep/` containing read-only parsers for nested ZIP/tar archives, non-destructive phoneme normalizers with reversible metadata, safe audio loaders/resamplers (44.1 kHz → 16 kHz), and PyTorch Dataset/DataLoader adapters strictly driven by frozen split manifests without test leakage. Exhaustive tests verify the 119,151 canonical phoneme census and CPU micro-benchmarks profile host performance.

**Tech Stack:** Python 3.14, PyTorch 2.14.1 (CPU), torchaudio 2.11.0, soundfile 0.14.0, scipy 1.18.1, pydantic 2.14.1, pytest 9.1.1, psutil, rclone.

**Spec:** `manifests/RQ1-STD-01/DATA_CONTRACT_SPEC.md`, `manifests/RQ1-STD-01/MDD_LABEL_SCHEMA.md`, `manifests/RQ1-STD-01/reconciliation/CORRECTED_LABEL_ACCOUNTING.md`, `manifests/RQ1-STD-01/AUDIO_PREPARATION_SPEC.md`.

## Global Constraints

- DATA PREPARATION ONLY: strictly no model training, fine-tuning, GOP inference, forced alignment, or parameter optimization.
- Raw archives (`l2arctic_release_v5.0.zip` and `speechocean762.tar.gz`) remain strictly immutable.
- Frozen split manifests (`manifests/RQ1-SPLIT-01/*.json`) must not be rewritten.
- Strict binary MDD: primary benchmark ($E_{\text{canon, strict}} = 17,518$ tokens).
- Lenient binary MDD: pedagogical sensitivity ($E_{\text{canon, lenient}} = 16,406$ tokens, 1,112 base-preserving allophones reclassified as correct).
- Full L2-ARCTIC gold census: exactly 3,599 utterances, 119,151 canonical phones, 101,633 correct, 14,098 substitutions, 3,420 deletions, 1,092 insertions.
- SpeechOcean762: exclude the 245 unmanifested audio files; do not convert scalar scores to fabricated categorical substitutions.
- CTC `<BLANK>` (index 0) must never be conflated with acoustic silence (`sil`) or phoneme deletions.
- Memory constraint: do not load the full 7.5 GB archive into RAM; use streaming or bounded temporary staging.
- Reports must sync to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`.

## Review Focus

1. **Direct ZIP stream corruption / memory blowout:** Reading 7.5 GB archive concurrently across workers must use seekable buffer handles or ephemeral staging rather than unconstrained decompression into RAM.
2. **Strict vs. Lenient off-by-one label leaks:** Base-preserving allophones (`CPL,CPL*,s`) must flip strictly from 1 to 0 in lenient mode without changing the canonical token denominator ($N=119,151$).
3. **CTC `<BLANK>` vs silence collision:** Padding collator must treat index 0 as CTC `<BLANK>`, silence as explicit phoneme/pause token, and deletions as absent from realized alignment.
4. **Test-partition label leakage in dataloader:** Test/evaluation partition loaders must not expose target labels when instantiated in blinded evaluation mode.
5. **Timestamp erratum provenance:** Document clearly that `2026-10-23T13:30:00Z` was a hardcoded reproducible string generated on a UTC+7 machine, explaining the mismatch with Google Drive upload records.

---

### Task 1: Preflight Split Integrity & Provenance Time Erratum

**Files:**
- Create: `src/langfens_prep/provenance.py`
- Create: `tests/test_provenance.py`
- Create: `manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md`

**Interfaces:**
- Produces: `verify_split_manifest_hashes() -> dict[str, bool]`, `compute_canonical_payload_sha256(payload: dict) -> str`

- [ ] **Step 1: Write failing test in `tests/test_provenance.py`**
Verify that split manifest SHA-256 hashes match frozen records and canonical payload hashing produces identical digests to those recorded in `l2arctic_4fold_splits.json`.

- [ ] **Step 2: Run test to verify it fails**
Run: `.venv/bin/pytest tests/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langfens_prep'`

- [ ] **Step 3: Implement `src/langfens_prep/provenance.py`**
Implement SHA-256 validation for splits and archives, canonical JSON serialization (`json.dumps(obj, sort_keys=True).encode("utf-8")`), and host timezone vs UTC analysis.

- [ ] **Step 4: Generate `manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md`**
Document the UTC+7 host environment, `DEFAULT_TIMESTAMP_UTC = "2026-10-23T13:30:00Z"` in `audit_scripts/generate_rq1_splits.py`, exact file mtimes (13:23 local = 06:23 UTC), rclone sync times (~06:26 UTC), and explain why the manifests are preserved with `NOT_VERIFIED` for historical wall-clock accuracy while payload digests are cryptographically verified.

- [ ] **Step 5: Run tests to verify they pass**
Run: `.venv/bin/pytest tests/test_provenance.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/langfens_prep/provenance.py tests/test_provenance.py manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md
git commit -m "feat(provenance): add split manifest validation and time erratum"
```

---

### Task 2: Robust Annotation Parsers & Non-Destructive Normalizers

**Files:**
- Create: `src/langfens_prep/textgrid_parser.py`
- Create: `src/langfens_prep/speechocean_parser.py`
- Create: `src/langfens_prep/phoneme_normalizer.py`
- Create: `tests/test_parsers.py`
- Create: `tests/test_normalizer.py`

**Interfaces:**
- Consumes: `src/langfens_prep/provenance.py`
- Produces:
  - `parse_gold_textgrid_content(text: str, filename: str) -> L2ArcticUtteranceAnnotation`
  - `parse_speechocean_metadata(archive_path: str, allowed_utts: set[str]) -> dict[str, SpeechOceanUtteranceRecord]`
  - `normalize_phoneme(token: str, fold_ax_to_ah: bool = True, strip_stress: bool = True) -> PhonemeNormalizationResult`

- [ ] **Step 1: Write failing tests in `tests/test_parsers.py` and `tests/test_normalizer.py`**
Test parsing of representative TextGrid intervals (correct, substitutions with asterisk `T*`, additions `sil,EH*,a`, deletions `T,sil,d`, typo `V8`, uncertain `err`), SpeechOcean score records (phone, word, utterance scores), and normalizer round-trip reversibility.

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_parsers.py tests/test_normalizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/langfens_prep/textgrid_parser.py`**
Extract `phones` and `words` intervals. Implement classification into clean substitutions, base-preserving allophones, category-shifting accented, deletions, epenthetic additions, and silence. Preserve raw text, start/end timestamps, and assign strict/lenient binary labels and diagnosis targets.

- [ ] **Step 4: Implement `src/langfens_prep/speechocean_parser.py`**
Parse official SpeechOcean762 `scores.json` and Kaldi-style metadata (`wav.scp`, `utt2spk`, `spk2gender`). Strictly enforce exclusion of the 245 unmanifested recordings. Preserve continuous scores without converting them to fabricated categorical substitution labels.

- [ ] **Step 5: Implement `src/langfens_prep/phoneme_normalizer.py`**
Implement ARPAbet 39-phone canonical vocabulary, stress digit extraction (`0`, `1`, `2`), `AX` folding to `AH`, asterisk extraction (`*`), typo handling (`V8` -> `V`), silence symbol handling (`sil`, `sp`, `spn`), and optional IPA mappings. Preserve raw strings in transformation metadata. Disambiguate CTC `<BLANK>` from silence.

- [ ] **Step 6: Run tests to verify they pass**
Run: `.venv/bin/pytest tests/test_parsers.py tests/test_normalizer.py -v`
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add src/langfens_prep/textgrid_parser.py src/langfens_prep/speechocean_parser.py src/langfens_prep/phoneme_normalizer.py tests/test_parsers.py tests/test_normalizer.py
git commit -m "feat(parsers): implement gold textgrid parser, speechocean parser and normalizer"
```

---

### Task 3: Audio Loading, Resampling & Ephemeral Staging

**Files:**
- Create: `src/langfens_prep/audio_loader.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Consumes: `src/langfens_prep/textgrid_parser.py`
- Produces:
  - `load_audio_from_archive(archive_path: str, member_path: str, target_sr: int = 16000, normalize_peak: bool = False, remove_dc: bool = False) -> tuple[torch.Tensor, int, AudioMetrics]`
  - `AudioStagingManager` (optional local disk staging for benchmark comparison)

- [ ] **Step 1: Write failing test in `tests/test_audio.py`**
Test safe extraction of a WAV file directly from `l2arctic_release_v5.0.zip` and `speechocean762.tar.gz`, conversion to float32 tensor in $[-1.0, 1.0]$, high-quality resampling from 44.1 kHz to 16 kHz with accurate duration mapping, and finiteness assertions (no NaNs, no Infs).

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_audio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/langfens_prep/audio_loader.py`**
Use `io.BytesIO` and `soundfile` or `torchaudio` to load in-memory WAV data without extracting the whole 7.5 GB archive to disk. Implement torchaudio `Resample` (sinc interpolation) for 44,100 Hz → 16,000 Hz. Expose optional configurable peak normalization and DC offset removal. Implement error handling for corrupted headers.

- [ ] **Step 4: Run tests to verify they pass**
Run: `.venv/bin/pytest tests/test_audio.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/langfens_prep/audio_loader.py tests/test_audio.py
git commit -m "feat(audio): implement safe archive audio loading and 16kHz resampling"
```

---

### Task 4: Manifest-Driven PyTorch Dataset & Collation

**Files:**
- Create: `src/langfens_prep/dataset.py`
- Create: `src/langfens_prep/collate.py`
- Create: `tests/test_dataset_and_loader.py`

**Interfaces:**
- Consumes: `src/langfens_prep/audio_loader.py`, `src/langfens_prep/phoneme_normalizer.py`
- Produces:
  - `L2ArcticMDDDataset(split_manifest_path: str, fold: int, split: str, policy: str = "strict") -> Dataset`
  - `SpeechOcean762ScoreDataset(split_manifest_path: str, split: str) -> Dataset`
  - `LangfensMDDCollate(pad_token_id: int = -100, ctc_blank_id: int = 0) -> Callable`

- [ ] **Step 1: Write failing test in `tests/test_dataset_and_loader.py`**
Verify dataset instantiation for train/val/test splits, verify that test-set instances in evaluation mode hide label supervision, and test collation of 4 variable-length examples into padded batches.

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_dataset_and_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/langfens_prep/dataset.py`**
Construct index from frozen split JSON manifests. Map each utterance ID to its nested archive member path. Yield structured dictionaries conforming to `LangfensUtteranceRecord` in `DATA_CONTRACT_SPEC.md`.

- [ ] **Step 4: Implement `src/langfens_prep/collate.py`**
Pad waveforms with 0.0, construct boolean padding masks, pad binary MDD targets with -100, pad realized CTC sequences with 0 (<BLANK>), compute lengths, and attach stable utterance IDs.

- [ ] **Step 5: Run tests to verify they pass**
Run: `.venv/bin/pytest tests/test_dataset_and_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add src/langfens_prep/dataset.py src/langfens_prep/collate.py tests/test_dataset_and_loader.py
git commit -m "feat(dataset): implement manifest-driven PyTorch datasets and batch collators"
```

---

### Task 5: Exhaustive Full Census & Data Contract Validation

**Files:**
- Create: `src/langfens_prep/validator.py`
- Create: `tests/test_full_census.py`
- Create: `manifests/RQ1-PREP-01/DATA_CONTRACT_VALIDATION.json`
- Create: `manifests/RQ1-PREP-01/DATA_VALIDATION_REPORT.md`
- Create: `manifests/RQ1-PREP-01/ANNOTATION_NORMALIZATION_REPORT.md`

**Interfaces:**
- Consumes: all preceding modules
- Produces: full corpus validation metrics and contract validation JSON

- [ ] **Step 1: Write test in `tests/test_full_census.py`**
Assert exact counts for L2-ARCTIC gold census:
- 3,599 utterances
- 119,151 canonical phones
- 101,633 correct tokens
- 14,098 substitutions (12,364 clean diff base + 4 clean same base + 1,112 base-preserving asterisk + 360 category-shifting asterisk + 258 distorted `err`)
- 3,420 deletions
- 1,092 insertions
- 17,518 strict binary error tokens ($14,098 + 3,420$)
- 16,406 lenient binary error tokens ($12,986 + 3,420$)
- Invariant denominator $N = 119,151$ across both policies.
- Check SpeechOcean762: exactly 245 excluded unmanifested files, 0 split overlap.

- [ ] **Step 2: Run test and generate full census**
Run: `.venv/bin/pytest tests/test_full_census.py -v -s`
Verify reconciliation against `CORRECTED_LABEL_ACCOUNTING.md`.

- [ ] **Step 3: Generate `DATA_CONTRACT_VALIDATION.json`**
Output structured Pydantic-validated results, token counts, schema integrity checks, and partition isolation assertions.

- [ ] **Step 4: Generate `DATA_VALIDATION_REPORT.md` and `ANNOTATION_NORMALIZATION_REPORT.md`**
Document all census results, label semantics, typo normalization audit, silence tokens, and partition integrity.

- [ ] **Step 5: Commit**
```bash
git add src/langfens_prep/validator.py tests/test_full_census.py manifests/RQ1-PREP-01/DATA_CONTRACT_VALIDATION.json manifests/RQ1-PREP-01/DATA_VALIDATION_REPORT.md manifests/RQ1-PREP-01/ANNOTATION_NORMALIZATION_REPORT.md
git commit -m "feat(validation): reconcile full corpus census and emit validation reports"
```

---

### Task 6: CPU Micro-Benchmarking

**Files:**
- Create: `src/langfens_prep/benchmark.py`
- Create: `manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md`

**Interfaces:**
- Consumes: `src/langfens_prep/audio_loader.py`, `src/langfens_prep/dataset.py`, `src/langfens_prep/collate.py`
- Produces: benchmark statistics (latencies, throughput, peak RAM, worker comparisons)

- [ ] **Step 1: Implement `src/langfens_prep/benchmark.py`**
Benchmark:
- Direct archive extraction latency vs. local staging latency.
- Resampling latency (44.1 kHz → 16 kHz).
- Collation latency.
- DataLoader throughput across num_workers = [0, 2, 4].
- Peak RAM tracking via `psutil`.
- Record host hardware: CPU model, core count, OS, memory, PyTorch & torchaudio versions.

- [ ] **Step 2: Run benchmark suite**
Run: `.venv/bin/python3 -m langfens_prep.benchmark`
Execute 5 warm-up iterations and 50 measured iterations per condition; compute median, p95, and utterances/sec throughput.

- [ ] **Step 3: Generate `manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md`**
Record exact empirical numbers, host specs, and analyze throughput bottlenecks honestly. Note that prior 2,380 utts/sec claims were unverified.

- [ ] **Step 4: Commit**
```bash
git add src/langfens_prep/benchmark.py manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md
git commit -m "perf(benchmark): run host CPU micro-benchmarks and document audio throughput"
```

---

### Task 7: Gate Evaluation, Final Reports & Remote Synchronization

**Files:**
- Create: `manifests/RQ1-PREP-01/DATA_PREPARATION_REPORT.md`
- Create: `manifests/RQ1-PREP-01/PREP_EXECUTION_LOG.md`
- Create: `manifests/RQ1-PREP-01/BLOCKERS_AND_NEXT_ACTIONS.md`

**Interfaces:**
- Evaluates: All 7 acceptance gates
- Syncs: Local `manifests/RQ1-PREP-01/` to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`

- [ ] **Step 1: Synthesize `DATA_PREPARATION_REPORT.md`**
Comprehensive report synthesizing data architecture, pipeline components, census results, benchmark results, and gate statuses.

- [ ] **Step 2: Synthesize `PREP_EXECUTION_LOG.md`**
Chronological engineering log documenting each phase execution, SHA-256 verifications, and unit test outputs.

- [ ] **Step 3: Synthesize `BLOCKERS_AND_NEXT_ACTIONS.md`**
Handover document assessing acceptance gates:
- `ANNOTATION_PARSER_VALIDATED: PASS`
- `LABEL_SEMANTICS_VALIDATED: PASS`
- `AUDIO_PIPELINE_VALIDATED: PASS`
- `MANIFEST_ISOLATION_VALIDATED: PASS`
- `DATALOADER_VALIDATED: PASS`
- `PERFORMANCE_PROFILED: PASS`
- `PROVENANCE_DOCUMENTED: PASS`
- `RQ1-PREP-01: PASS`
List next actions for acoustic modeling preparation and confirm no model training was executed.

- [ ] **Step 4: Synchronize to Google Drive**
Run: `rclone sync manifests/RQ1-PREP-01/ kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/ --fast-list -v`
Verify with `rclone ls kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`.

- [ ] **Step 5: Commit final report deliverables**
```bash
git add manifests/RQ1-PREP-01/
git commit -m "docs(prep): finalize RQ1-PREP-01 audit reports and synchronization"
```
