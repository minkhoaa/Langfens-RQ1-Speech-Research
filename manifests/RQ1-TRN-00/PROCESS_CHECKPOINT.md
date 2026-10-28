# LANGFENS — RQ1-TRN-00 / Baseline Experiment Design and Environment Planning

**Status:** COMPLETED

**Recorded at:** 2026-10-23T16:00:00+07:00

**Evidence source:** Commits `7cd78dd`, `15b2aa0`, `9382cca`, `175d68f`, `manifests/RQ1-TRN-00/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Formulate the baseline modeling strategy comparing Goodness of Pronunciation (GOP) against end-to-end Wav2Vec 2.0 multi-task MDD, establish dual execution environments, plan GPU compute resources, and harden training contracts with explicit loss masking.

## 2. Inputs & Dependencies
- Source branch: `main` (commit `175d68f`).
- Data pipeline package: `src/langfens_prep/`.
- Frozen splits: `manifests/RQ1-SPLIT-01/`.
- Training contract review: `manifests/RQ1-TRN-00/TRAINING_CONTRACT_REVIEW.md`.

## 3. Work Performed
1. Conducted comparative design assessment between Baseline A (classical GOP over forced alignment) and Baseline B (Wav2Vec 2.0 self-supervised multi-task MDD).
2. Defined experimental protocol and research execution order, prioritizing Wav2Vec 2.0 due to deletion handling capability ($D = 3,420$) and end-to-end representation learning.
3. Designed dual environment architecture: `.venv` (Python 3.14 CPU development) and `.venv-cuda` (Python 3.11 GPU training with PyTorch 2.14.1+cu130).
4. Sized Google Colab GPU compute requirements (T4 GPU, 16 GB VRAM, AMP fp16, gradient checkpointing).
5. Implemented explicit loss masking in `src/langfens_prep/losses.py` (`MaskedBCEWithLogitsLoss` using `target_binary_mask`, `MaskedCrossEntropyLoss` using `target_diagnosis_mask`).

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-00: PASS`.
- Contract tests passing: `tests/test_training_contract.py` (commit `7cd78dd`).
- Formalized compute plan: batch size 2, gradient accumulation 2 (effective batch 4), estimated 90 sec/epoch on Colab T4.
- Documented in [BASELINE_EXPERIMENT_DESIGN.md](BASELINE_EXPERIMENT_DESIGN.md) and [COMPUTE_RESOURCE_PLAN.md](COMPUTE_RESOURCE_PLAN.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md`, `COMPUTE_RESOURCE_PLAN.md`, `DEPENDENCY_ENVIRONMENT_REPORT.md`, `SOURCE_REPRODUCIBILITY_AUDIT.md`, `TRAINING_CONTRACT_REVIEW.md`, `EXPERIMENT_PROTOCOL.md`.
- Contract implementations: `src/langfens_prep/losses.py`, `tests/test_training_contract.py`.
- Environment lockfiles: `requirements.txt`, `requirements-lock.txt`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 (`.venv`) or Python 3.11 (`.venv-cuda`).
- Verify training contract tests:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_training_contract.py
  ```
- Lightweight CPU test suite; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Model class implementation deferred to RQ1-TRN-01A.
- Prohibited actions: Never execute training backward passes on local CPU/GPU; Colab GPU mandatory.
- Next phase: [RQ1-TRN-01A](../RQ1-TRN-01A/PROCESS_CHECKPOINT.md).
