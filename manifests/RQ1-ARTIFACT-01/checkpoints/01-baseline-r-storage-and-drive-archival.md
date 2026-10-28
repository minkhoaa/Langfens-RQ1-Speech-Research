# LANGFENS — RQ1-ARTIFACT-01 / 01-Baseline R Storage and Drive Archival

**Status:** COMPLETED

**Recorded at:** 2026-10-27T17:10:00+07:00

**Evidence source:** Commit `a200b73b90a137316817bc42df0ba6e0db2c24c5`, `manifests/RQ1-ARTIFACT-01/STORAGE_VERIFICATION_REPORT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Verify the persistent storage hierarchy, file ancestry, write permissions, and cryptographic checksums of Baseline R artifacts on Google Drive under experiment folder `RQ1-TRN-02A`.

## 2. Inputs & Dependencies
- Google Drive Remote: `kltn_gdrive:KLTN-Dataset` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`).
- Experiment folder ID: `11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`.
- Local verification tools: `rclone`, Python `hashlib`.

## 3. Work Performed
1. Audited folder ancestry: confirmed all training outputs reside strictly under authorized root folder `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`.
2. Verified remote file sizes and cryptographic checksums for `best.pt`, `last.pt`, `history.json`, `EVALUATION_REPORT.json`, and `BASELINE_COMPARISON.json`.
3. Verified authorized fold package `fold1_train_val_only.zip` (761,922,918 bytes, SHA-256 `b750eb3b...`).
4. Uploaded and performed read-back verification of `BASELINE_R_ARTIFACT_MANIFEST.json` (ID: `1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP`).

## 4. Results & Verification
- Acceptance gate: `STORAGE_VERIFICATION: PASS`.
- Checkpoints intact: `best.pt` (412,366,029 B, SHA-256 `a2445c8d...`), `last.pt` (412,367,181 B, SHA-256 `bf46fab3...`).
- Zero duplicate files or test split leaks detected.
- Documented in [STORAGE_VERIFICATION_REPORT.md](../STORAGE_VERIFICATION_REPORT.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-ARTIFACT-01/STORAGE_VERIFICATION_REPORT.md`.
- Remote Manifest: `BASELINE_R_ARTIFACT_MANIFEST.json` on Google Drive.

## 6. Restart / Recovery Instructions
- Inspect Drive storage via rclone:
  ```bash
  rclone lsf kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/
  ```
- Read-only query; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Model export and Hugging Face release.
- Prohibited actions: Never stream training audio directly across Drive mounts.
- Next step: [02-safetensors-export-and-huggingface-release.md](02-safetensors-export-and-huggingface-release.md).
