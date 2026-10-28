# LANGFENS — RQ1-TRN-01A / 02-Backbone Integrity and Inference Validity

**Status:** COMPLETED

**Recorded at:** 2026-10-23T20:30:00+07:00

**Evidence source:** Commit `6581027`, `manifests/RQ1-TRN-01A/reconciliation/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Verify that the pretrained Wav2Vec 2.0 backbone remains strictly frozen, reconcile realized phone token indexing with CTC `<BLANK>`, and profile GPU VRAM consumption.

## 2. Inputs & Dependencies
- Implemented model: `src/langfens_prep/model.py`.
- Hugging Face revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Reconciliation tests: `tests/test_reconciliation_regression.py`.

## 3. Work Performed
1. Audited parameter gradients: verified that all 94,396,320 backbone parameters have `requires_grad=False`.
2. Reconciled CTC phone vocabulary: index 0 `<BLANK>`, indices 1..39 mapped to standard ARPAbet symbols.
3. Profiled GPU memory scaling across batch sizes 1, 2, 4 on GTX 1650 (4 GB) and projected to Colab T4 (16 GB).

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01A-R: PASS`.
- Zero gradient flow into Wav2Vec 2.0 transformer layers confirmed.
- Verified VRAM footprint: ~3.2 GB with batch size 2 and AMP fp16.
- Documented in [PRETRAINED_BACKBONE_INTEGRITY.md](../reconciliation/PRETRAINED_BACKBONE_INTEGRITY.md) and [FULL_BACKBONE_GPU_FEASIBILITY.md](../reconciliation/FULL_BACKBONE_GPU_FEASIBILITY.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01A/reconciliation/PRETRAINED_BACKBONE_INTEGRITY.md`, `REALIZED_PHONE_CTC_RECONCILIATION.md`, `FULL_BACKBONE_GPU_FEASIBILITY.md`.
- Metrics: `manifests/RQ1-TRN-01A/reconciliation/gpu_feasibility_measurements.json`.

## 6. Restart / Recovery Instructions
- Run backbone integrity verification:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_reconciliation_regression.py -k test_pretrained_backbone_frozen
  ```
- Fast CPU test.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Acoustic padding validation on variable-length inputs.
- Prohibited actions: Never enable backbone gradients during baseline runs.
- Next step: [03-training-correctness-and-acoustic-padding.md](03-training-correctness-and-acoustic-padding.md).
