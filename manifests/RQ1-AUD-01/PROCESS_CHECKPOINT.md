# LANGFENS — RQ1-AUD-01 / Dataset Acquisition and Annotation Audit

**Status:** COMPLETED

**Recorded at:** 2026-10-22T22:15:00+07:00

**Evidence source:** Commit `5ccb4dc2cc986036175161c660b73a206de188d2`, `manifests/RQ1-AUD-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Establish cryptographic integrity baselines for raw speech corpora (L2-ARCTIC v5.0 and SpeechOcean762), conduct exhaustive annotation audits across 3,599 gold Praat TextGrids and 5,000 scored utterances, and reconcile phonetic inventory alignment across benchmark lexicons before downstream split generation.

## 2. Inputs & Dependencies
- Source branch: `main` (commit `5ccb4dc2cc986036175161c660b73a206de188d2`).
- L2-ARCTIC v5.0 raw archive: `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` (7,515,454,814 bytes, SHA-256 `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`).
- SpeechOcean762 raw archive: `datasets/SpeechOcean762/original/speechocean762.tar.gz` (520,810,923 bytes, SHA-256 `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`).
- Reference lexicons: CMUDict v0.7b (`resources/CMUDict/cmudict.dict`, commit `74790861`), MFA `english_us_arpa.dict` v3.0.0.

## 3. Work Performed
1. Verified raw archive checksums, performed non-destructive ZIP CRC-32 and TAR stream integrity scans.
2. Parsed all 3,599 manual gold Praat TextGrids across 24 speakers (135,890 intervals) using in-memory streams.
3. Profiled SpeechOcean762 multi-aspect rater distributions across 5,000 scored utterances (94,445 phones).
4. Reconciled phonetic inventories (39-ARPAbet base, AX schwa mapping, stress stripping) and cataloged label issues.
5. Executed post-gate reconciliation (`RQ1-AUD-01R`) to resolve denominator accounting, isolate 245 unannotated orphan WAVs in SpeechOcean762, and rectify runtime logging.

## 4. Results & Verification
- Acceptance gate: `RQ1-AUD-01: PASS` / `RQ1-AUD-01R: PASS`.
- L2-ARCTIC census: 119,151 canonical non-silence tokens; 101,633 Correct (85.30%), 17,518 Error (14.70%).
- Error decomposition: 14,098 Substitutions, 3,420 Deletions, 1,092 Insertions; PER = 15.62%.
- Vietnamese cohort deletion concentration: 1,566 / 3,420 (45.79% of all corpus deletions).
- Discovered 245 unmanifested extra WAVs in SpeechOcean762 (`SPEAKER9600`–`SPEAKER9649`), flagged for exclusion.
- Documented in [L2_ARCTIC_ANNOTATION_AUDIT.md](L2_ARCTIC_ANNOTATION_AUDIT.md) and [SCIENTIFIC_CONSISTENCY_REPORT.md](reconciliation/SCIENTIFIC_CONSISTENCY_REPORT.md).

## 5. Output Artifacts
- Audit deliverables: `manifests/RQ1-AUD-01/*.md`, `*.csv` (commit `5ccb4dc`).
- Reconciliation specs: `manifests/RQ1-AUD-01/reconciliation/*.md` (commit `5ccb4dc`).
- Python audit tools: `audit_scripts/audit_archive_integrity.py`, `audit_scripts/audit_l2arctic_annotations.py`.
- Persistent Google Drive archive: [`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`](https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK).

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 CPU reference environment (`.venv`).
- Verification command:
  ```bash
  python audit_scripts/audit_archive_integrity.py
  ```
- No user authorization required (read-only audit).
- Validation: SHA-256 digests must match ledger exactly before downstream pipeline ingestion.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: 245 unannotated SpeechOcean762 WAVs must never be fed into supervised training.
- Prohibited actions: Never decompress raw archives to disk; do not modify original archives or TextGrids.
- Next phase: [RQ1-STD-01](../RQ1-STD-01/PROCESS_CHECKPOINT.md).
