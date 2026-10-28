# LANGFENS — RQ1-ARTIFACT-01 / Baseline R Archival and Private Hugging Face Release

**Status:** COMPLETED

**Recorded at:** 2026-10-27T17:35:00+07:00

**Evidence source:** Commit `a200b73b90a137316817bc42df0ba6e0db2c24c5`, `manifests/RQ1-ARTIFACT-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Execute the formal archival, serialization, private publication, and three-platform cross-verification of LANGFENS RQ1 Fold 1 Baseline R across Google Drive, Hugging Face Hub, and GitHub with verified byte-level checksums and numerical parity.

## 2. Inputs & Dependencies
- Source branch: `docs/rq1-artifact-01-baseline-release` (commit `a200b73b90a137316817bc42df0ba6e0db2c24c5`).
- Checkpoint: `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`, 412,366,029 bytes).
- Google Drive root: [`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`](https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK).
- Release package script: `audit_scripts/export_huggingface_release.py`.

## 3. Work Performed
1. Verified persistent Google Drive hierarchy under folder ID `11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx` (`RQ1-TRN-02A`), confirming hashes for `best.pt`, `last.pt`, `fold1_train_val_only.zip`, and training logs.
2. Uploaded and read-back verified `BASELINE_R_ARTIFACT_MANIFEST.json` directly on Drive.
3. Exported native lossless FP32 `model.safetensors` (389,650,424 bytes, SHA-256 `a5b9d41f...`).
4. Executed forward pass numerical parity verification on all 300 validation utterances, confirming bit-for-bit equivalence between PyTorch checkpoint and `model.safetensors` (maximum logit difference: 0.000000e+00).
5. Published private model release to Hugging Face Hub at `minkhoaa/langfens-rq1-fold1-baseline-r` (commit `6c31f719...`). Dataset metadata repo recorded as pending upload.

## 4. Results & Verification
- Acceptance gate: `BASELINE_R_RELEASE: PASS`.
- Google Drive archival: Verified present and intact under authorized workspace.
- Numerical parity: Max absolute CTC and Binary logit differences are `0.000000e+00` (100% equivalence).
- Hugging Face repository visibility: Confirmed strictly **PRIVATE** via authenticated API queries.
- Official test sealing: Zero test audio, TextGrids, or labels staged or released.
- Documented in [BASELINE_RELEASE_MANIFEST.md](BASELINE_RELEASE_MANIFEST.md), [STORAGE_VERIFICATION_REPORT.md](STORAGE_VERIFICATION_REPORT.md), and [HUGGINGFACE_RELEASE_REPORT.md](HUGGINGFACE_RELEASE_REPORT.md).

## 5. Output Artifacts
- Manifests: `manifests/RQ1-ARTIFACT-01/BASELINE_RELEASE_MANIFEST.md`, `STORAGE_VERIFICATION_REPORT.md`, `HUGGINGFACE_RELEASE_REPORT.md`.
- Remote Drive Manifest: [`BASELINE_R_ARTIFACT_MANIFEST.json`](https://drive.google.com/file/d/1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP/view) (SHA-256 `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22`).
- Hugging Face Model: [`minkhoaa/langfens-rq1-fold1-baseline-r`](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r) (rev `6c31f7196c2e03a6766c5525833de465a1a9345d`).

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 (`.venv`) with `huggingface_hub` and `safetensors`.
- Verify remote Hugging Face metadata (requires authenticated token):
  ```bash
  python -c "from huggingface_hub import HfApi; info = HfApi().repo_info('minkhoaa/langfens-rq1-fold1-baseline-r'); print('Private:', info.private, 'SHA:', info.sha)"
  ```
- Read-only API verification.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Hugging Face dataset metadata repository upload pending operator sign-off.
- Prohibited actions: Never make Hugging Face model repository public without user authorization; never release sealed test data.
- Next phase: [RQ1-TRN-02B](../RQ1-TRN-02B/PROCESS_CHECKPOINT.md).
