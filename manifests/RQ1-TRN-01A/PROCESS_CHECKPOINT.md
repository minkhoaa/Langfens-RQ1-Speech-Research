# LANGFENS — RQ1-TRN-01A / Model Architecture and Training Contract

**Status:** COMPLETED

**Recorded at:** 2026-10-23T22:40:00+07:00

**Evidence source:** Commits `f0ade49`, `6581027`, `5942338`, `manifests/RQ1-TRN-01A/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Implement the multi-task `LangfensWav2Vec2MDD` architecture (joint CTC recognition, cross-attention canonical alignment, binary MDD detection, and multi-class diagnosis), verify frozen pretrained backbone integrity, and resolve acoustic padding and step accounting contracts.

## 2. Inputs & Dependencies
- Source branch: `feat/rq1-trn-01a-training-smoke-test` (commits `f0ade49` through `5942338`).
- Pretrained backbone: `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94,396,320 parameters).
- Model implementation: `src/langfens_prep/model.py`.
- Training contract review: `manifests/RQ1-TRN-00/TRAINING_CONTRACT_REVIEW.md`.

## 3. Work Performed
1. Implemented `LangfensWav2Vec2MDD` combining Wav2Vec2 feature extraction, `CanonicalAcousticCrossAttention` (4 heads, 128 dim), `ctc_head` (40 classes), `binary_head` (MLP $\to 1$ logit), and `diagnosis_head` (39 classes).
2. Executed synthetic smoke tests on CPU and CUDA environments (`f0ade49`).
3. Completed RQ1-TRN-01A-R gate: verified pretrained backbone remains frozen (`requires_grad=False`), validated CTC blank indexing, and measured GPU VRAM profiles (`6581027`).
4. Completed RQ1-TRN-01A-R2 final gate: validated acoustic padding handling, resolved CTC contract violations on short inputs (zero-loss fallback), and reconciled step accounting (`5942338`).

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01A: PASS`, `RQ1-TRN-01A-R: PASS`, `RQ1-TRN-01A-R2: PASS`.
- Pretrained backbone freeze: Exactly 94,396,320 frozen parameters verified; only heads and cross-attention are trainable.
- CTC contract: Enforces length check (`T_frames >= 2 * L_target`); raises warning and emits zero loss if violated.
- Documented in [MODEL_ARCHITECTURE_REPORT.md](MODEL_ARCHITECTURE_REPORT.md), [PRETRAINED_BACKBONE_INTEGRITY.md](reconciliation/PRETRAINED_BACKBONE_INTEGRITY.md), and [LOSS_AND_CTC_CONTRACT_AUDIT.md](reconciliation/final_gate/LOSS_AND_CTC_CONTRACT_AUDIT.md).

## 5. Output Artifacts
- Architecture code: `src/langfens_prep/model.py`.
- Reports: `manifests/RQ1-TRN-01A/MODEL_ARCHITECTURE_REPORT.md`, `CUDA_ENVIRONMENT_REPORT.md`, `TRAINING_CONTRACT_TEST_REPORT.md`.
- Reconciliation reports: `manifests/RQ1-TRN-01A/reconciliation/*.md`, `manifests/RQ1-TRN-01A/reconciliation/final_gate/*.md`.
- Test suites: `tests/test_model.py`, `tests/test_reconciliation_regression.py`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 (`.venv`) or Python 3.11 CUDA (`.venv-cuda`).
- Run architecture and regression tests:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_model.py tests/test_reconciliation_regression.py
  ```
- Fast CPU test execution; zero user authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Real-data training sanity check deferred to RQ1-TRN-01B0.
- Prohibited actions: Never unfreeze the Wav2Vec2 backbone during baseline training; never use gold TextGrid timestamps as model inputs.
- Next phase: [RQ1-TRN-01B0](../RQ1-TRN-01B0/PROCESS_CHECKPOINT.md).
