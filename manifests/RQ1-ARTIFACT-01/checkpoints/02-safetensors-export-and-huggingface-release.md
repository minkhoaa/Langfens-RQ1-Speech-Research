# LANGFENS — RQ1-ARTIFACT-01 / 02-Safetensors Export and Hugging Face Release

**Status:** COMPLETED

**Recorded at:** 2026-10-27T17:30:00+07:00

**Evidence source:** Commit `a200b73b90a137316817bc42df0ba6e0db2c24c5`, `manifests/RQ1-ARTIFACT-01/HUGGINGFACE_RELEASE_REPORT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Export native lossless FP32 `model.safetensors` from production `best.pt`, verify bit-for-bit numerical parity across validation utterances, and publish private model repository to Hugging Face Hub.

## 2. Inputs & Dependencies
- Checkpoint: `best.pt` (SHA-256 `a2445c8d...`).
- Exporter script: `audit_scripts/export_huggingface_release.py`.
- Target namespace: `minkhoaa` (authenticated via `huggingface_hub`).

## 3. Work Performed
1. Exported weights to native `model.safetensors` (389,650,424 bytes, SHA-256 `a5b9d41f...`).
2. Ran numerical parity test comparing forward pass outputs of PyTorch checkpoint versus `model.safetensors` on all 300 validation utterances.
3. Created model card (`README.md`), configuration (`config.json`), metadata, and standalone CPU inference loader (`example_inference.py`).
4. Uploaded release package to private Hugging Face repository `minkhoaa/langfens-rq1-fold1-baseline-r` (tag `baseline-r-v0.1.0`, commit `6c31f719...`).
5. Documented Hugging Face dataset metadata repository status as pending upload.

## 4. Results & Verification
- Acceptance gate: `HUGGINGFACE_RELEASE: PASS`.
- Bit-for-bit equivalence: Maximum absolute difference for CTC and Binary logits = `0.000000e+00`.
- Privacy verified: `repo_info.private == True`.
- Documented in [HUGGINGFACE_RELEASE_REPORT.md](../HUGGINGFACE_RELEASE_REPORT.md).

## 5. Output Artifacts
- Hugging Face Model: [`https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r`](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r).
- Report: `manifests/RQ1-ARTIFACT-01/HUGGINGFACE_RELEASE_REPORT.md`.

## 6. Restart / Recovery Instructions
- Verify model download and inference:
  ```bash
  python -c "from huggingface_hub import hf_hub_download; path = hf_hub_download('minkhoaa/langfens-rq1-fold1-baseline-r', 'config.json'); print('Downloaded config:', path)"
  ```
- Requires authenticated Hugging Face token.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Public release pending thesis evaluation; dataset metadata repo upload pending.
- Prohibited actions: Never make repository public without user authorization; never upload raw audio.
- Next phase: [RQ1-TRN-02B](../../RQ1-TRN-02B/PROCESS_CHECKPOINT.md).
