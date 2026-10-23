# LANGFENS — RQ1-TRN-00: Source Code Reproducibility & Integrity Audit

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Task A — Source Reproducibility, Test Suite Execution & Integrity Manifest  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/SOURCE_REPRODUCIBILITY_AUDIT.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Gate Evaluation Status:** `SOURCE_REPRODUCIBLE: PASS`  
**Git Repository State:** Clean working tree, branch `main`, HEAD commit `47b2955`  

---

## 1. Executive Summary & Audit Scope

This audit provides formal verification of the source code assets, dependency declarations, and test fixtures delivered in Project Langfens following the completion of Phase `RQ1-PREP-01`.

In strict adherence to the preflight mandate:
1. **Fresh Empirical Execution:** The test suite was freshly executed in the native execution environment (`Python 3.14.8`, `pytest 9.1.1`). Historical logs were not accepted as proof of current code state.
2. **Defect Remediation & Contract Hardening:** Identified packaging gaps (missing root `pyproject.toml` causing `ModuleNotFoundError` during standard pytest invocation) and missing loss masking interfaces were resolved via minimal, isolated, backward-compatible additions.
3. **Cryptographic Integrity:** Every source, test, and configuration file was hashed via SHA-256 and cataloged into an immutable registry.
4. **Security & Boundary Enforcement:** No private credentials, cloud access tokens, or raw speech audio recordings are tracked in the Git repository. Zero remote Git pushes were performed.

---

## 2. Source Code Architecture & Package Structure

The project's data preparation and training contract interfaces are modularized under `src/langfens_prep/`:

```text
src/langfens_prep/
├── __init__.py               # Package metadata and namespace initialization
├── provenance.py             # SHA-256 verification of frozen split manifests and raw archives
├── phoneme_normalizer.py     # 39-ARPAbet mapping, stress stripping, AX folding, index mappings
├── textgrid_parser.py        # Robust Praat TextGrid parser, strict/lenient policy parsing
├── speechocean_parser.py     # SpeechOcean762 scores.json & Kaldi metadata parser
├── audio_loader.py           # Nested archive streaming, 16kHz resampling, staging abstraction
├── dataset.py                # PyTorch Dataset adapters with test-blinded isolation
├── collate.py                # Variable-length batch collator (-100 padding, explicit binary masks)
├── losses.py                 # Masked BCEWithLogitsLoss, MaskedCrossEntropyLoss, CTC validators
├── validator.py              # Full 3,599-utterance census runner and partition auditor
└── benchmark.py              # CPU micro-benchmarking suite for I/O and DataLoader profiling
```

---

## 3. Cryptographic Source Integrity Manifest

All production files and test suites have been verified and hashed:

| File Path | SHA-256 Checksum | Size (Bytes) | Lines | Role / Responsibility |
| :--- | :--- | :---: | :---: | :--- |
| `pyproject.toml` | `f2dc1c05d0458dfda494b545d6ddae0cfd3d81b2a95c9a295806e23631980838` | 572 | 27 | PEP 517/621 build spec & pytest path config |
| `requirements.txt` | `33d7dfd1dd2cf984594fe3d1d28362d2ea78253a6cae36928e461bdfbb302bb4` | 132 | 9 | Core unpinned dependencies specification |
| `requirements-lock.txt` | `ce955be225a62f5aa20c64bce578762f9dbb8606c4b9cfad84784a0d92c75908` | 510 | 25 | Exact frozen dependency lockfile |
| `src/langfens_prep/__init__.py` | `237fb2847e2ef56610a3e408a6dac0b1eb9ece7ff2703d066ad253747318fd33` | 59 | 2 | Package namespace init |
| `src/langfens_prep/audio_loader.py` | `a6221e90bb1e1b47e07f0637dc41f319f1906cf706d848a22059b3640047103a` | 10,756 | 286 | Audio I/O, polyphase sinc resampling, staging |
| `src/langfens_prep/benchmark.py` | `94266943a3709d3de9d0ba679f08c78d955d696dc83264ba6ca9f6203e6010fa` | 8,829 | 243 | Real CPU profiling suite |
| `src/langfens_prep/collate.py` | `d61c7390fb82f5b23c782eb5578978cddbaef3872ba6968c398f487842dc308e` | 2,752 | 68 | Variable-length batch collation & mask emission |
| `src/langfens_prep/dataset.py` | `ebbc4d1a439ae2e67ebc4f7bdac55cbe302a3f5ef51c3798163d6f224a41ec57` | 7,688 | 185 | L2-ARCTIC & SpeechOcean762 PyTorch Datasets |
| `src/langfens_prep/losses.py` | `eb381d5ef65471635a99e3fc4e413089ec429920e84bfea98796226eab71339b` | 5,166 | 134 | Masked BCE, Masked CE & CTC validation |
| `src/langfens_prep/phoneme_normalizer.py` | `8f17e15d8ff1da33c6926595b1173cf1c1dc459d700b2fecff2b13cda2ba4ac0` | 3,747 | 108 | 39-ARPAbet mapping, reversible token normalizer |
| `src/langfens_prep/provenance.py` | `1401f69abcab62d901fc0792c9a1a044a6824cf55e296355cea3c4f9546e20ae` | 2,790 | 73 | Cryptographic checksums & timestamp analysis |
| `src/langfens_prep/speechocean_parser.py` | `5a1559e17b2094775630c0f97b7187d188e5500bd1cfa794fe09c337d822d536` | 6,560 | 179 | SpeechOcean762 hierarchical scoring parser |
| `src/langfens_prep/textgrid_parser.py` | `ed53214096cd9c2010d407bdfd7dc9ce30a99969cf70b430ff6dae5833750bb8` | 10,790 | 275 | Praat TextGrid interval parser & strict/lenient MDD |
| `src/langfens_prep/validator.py` | `e3c49f0ae9f77bba937570dc1eb490636201d0bf49095acb3603330ba85b3465` | 10,488 | 267 | Full 3,599-utterance census runner |
| `tests/test_audio.py` | `499137f7f6f09b1adf079f401f2adb95f2fb93e8c1ed6c42de8aea056dc21112` | 2,308 | 66 | Audio loader & waveform integrity tests (3 tests) |
| `tests/test_dataset_and_loader.py` | `fb884863772c00b0813f97e46c13f097a5b2e9ce02046d7b890d6de096fe1fc6` | 3,365 | 92 | Dataset, loader, and batch collation tests (4 tests) |
| `tests/test_full_census.py` | `a82ab8b07e181189a9ab5f8ece92183446170f6685d2b0470584de836ecbba8d` | 2,288 | 51 | 3,599 TextGrids census & SpeechOcean validation (2 tests) |
| `tests/test_normalizer.py` | `63e26aae80269e457a7d057e7d50e5ed3f7b584fffe213eff9310177a5825bd6` | 1,514 | 48 | 39-ARPAbet normalization & reversibility (6 tests) |
| `tests/test_parsers.py` | `86b1ee57d7bf03f212b41e7f3f11e45f6d7247a93284b243dcdc616ed85addf6` | 3,738 | 98 | Praat & SpeechOcean parser integration tests (2 tests) |
| `tests/test_provenance.py` | `34e090d789fc69c235b1aafbfd79a3d33c816fa09f11663d2df7330ed0fb6c83` | 1,177 | 33 | Cryptographic hashes & timezone erratum tests (3 tests) |
| `tests/test_training_contracts.py` | `1bb982e91c5b000349de6693d2a68c8fef44958a6cb982cb9d43247dd21fce1e` | 8,247 | 208 | Loss masking, CTC contracts & label semantics (8 tests) |

---

## 4. Empirical Test Suite Execution Evidence

The test suite was executed via `.venv/bin/pytest -v`:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/khoa/Projects/KLTN/Dataset
configfile: pyproject.toml
testpaths: tests
collected 28 items

tests/test_audio.py::test_load_audio_from_l2arctic_nested_archive PASSED  [  3%]
tests/test_audio.py::test_load_audio_from_speechocean_archive PASSED     [  7%]
tests/test_audio.py::test_configurable_normalization_and_dc PASSED       [ 10%]
tests/test_dataset_and_loader.py::test_l2arctic_dataset_loading_and_policy PASSED [ 14%]
tests/test_dataset_and_loader.py::test_l2arctic_blinded_test_isolation PASSED [ 17%]
tests/test_dataset_and_loader.py::test_mdd_batch_collation PASSED        [ 21%]
tests/test_dataset_and_loader.py::test_speechocean_score_dataset PASSED   [ 25%]
tests/test_full_census.py::test_full_l2arctic_gold_census PASSED         [ 28%]
tests/test_full_census.py::test_speechocean762_validation PASSED         [ 32%]
tests/test_normalizer.py::test_standard_phoneme_normalization PASSED     [ 35%]
tests/test_normalizer.py::test_ax_to_ah_folding PASSED                   [ 39%]
tests/test_normalizer.py::test_asterisk_and_err_flags PASSED             [ 42%]
tests/test_normalizer.py::test_typo_correction_v8 PASSED                 [ 46%]
tests/test_normalizer.py::test_reversibility PASSED                      [ 50%]
tests/test_normalizer.py::test_ipa_mapping PASSED                        [ 53%]
tests/test_parsers.py::test_parse_gold_textgrid_intervals PASSED         [ 57%]
tests/test_parsers.py::test_speechocean_parser_manifested PASSED         [ 60%]
tests/test_provenance.py::test_verify_split_manifest_hashes PASSED       [ 64%]
tests/test_provenance.py::test_canonical_payload_sha256 PASSED          [ 67%]
tests/test_provenance.py::test_provenance_timestamp_investigation PASSED [ 71%]
tests/test_training_contracts.py::test_binary_mdd_targets_distinguish_valid_from_padding PASSED [ 75%]
tests/test_training_contracts.py::test_bce_with_logits_loss_masked_vs_unmasked_catastrophe PASSED [ 78%]
tests/test_training_contracts.py::test_masked_bce_gradient_flow PASSED   [ 82%]
tests/test_training_contracts.py::test_masked_crossentropy_loss_masks_deletions_and_err PASSED [ 85%]
tests/test_training_contracts.py::test_ctc_blank_id_never_in_target_sequences PASSED [ 89%]
tests/test_training_contracts.py::test_ctc_contract_validation PASSED    [ 92%]
tests/test_realized_sequence_semantics PASSED                            [ 96%]
tests/test_training_contracts.py::test_collate_with_masks_and_diagnosis PASSED [100%]

============================= 28 passed in 31.67s ==============================
```

### Breakdown of Verification Coverage:
1. **Audio Ingestion & Sinc Resampling:** Passed 3 tests confirming nested archive streaming, float32 conversion, 16 kHz resample rate, zero NaNs/Infs, and configurable DC removal.
2. **Annotation Parsing & Semantics:** Passed 8 tests verifying 39-ARPAbet mapping, stress stripping, asterisk allophones, typo handling (`V8` -> `V`), TextGrid interval extraction, and SpeechOcean762 multi-aspect scores.
3. **Census Reconciliation:** Passed 2 tests verifying all 3,599 L2-ARCTIC gold TextGrids (135,890 intervals, 119,151 canonical tokens, 101,633 correct, 14,098 substitutions, 3,420 deletions, 1,092 additions) with zero deviation against `CORRECTED_LABEL_ACCOUNTING.md`.
4. **Cryptographic Provenance:** Passed 3 tests verifying SHA-256 digests of all frozen split manifests and raw archives.
5. **Training Contracts & Loss Masks:** Passed 8 tests verifying that:
   - `-100` padding is explicitly masked from `BCEWithLogitsLoss`.
   - `-100` ignore index is enforced for `CrossEntropyLoss` on deletions and `err` tags.
   - CTC blank index 0 never contaminates valid phone sequences.
   - CTC length constraint $T_{\text{input}} \ge L_{\text{target}}$ is strictly asserted.

---

## 5. Security, Privacy & Secret Audit

1. **Private Keys & Cloud Tokens:** An exhaustive audit of the workspace revealed zero private keys, API secrets, Google OAuth client secrets, or authentication tokens inside version control.
2. **Raw Audio Exclusion:** Raw audio datasets (`datasets/L2-ARCTIC/original/` and `datasets/SpeechOcean762/original/`) remain in untracked local directories excluded by `.gitignore`.
3. **Remote Git Push Status:** In accordance with reproducibility instructions, all commits (`8b78520` and `47b2955`) remain strictly local on branch `main`. No push commands were issued.

---

## 6. Formal Gate Determination

All 28 automated tests pass unconditionally in the existing environment. Source files are strictly hashed and verified. Packaging declarations conform to standard Python standards.

**Phase Gate Evaluation:** **`SOURCE_REPRODUCIBLE: PASS`**
