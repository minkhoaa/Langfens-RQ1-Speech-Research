# LANGFENS — RQ1-TRN-02A / 02-Checkpoint Recovery and CPU Replay

**Status:** COMPLETED

**Recorded at:** 2026-10-26T11:55:00+07:00

**Evidence source:** Commits `9ac869e`, `3e2bc1d`, `manifests/RQ1-TRN-02A/evaluation_reconciliation/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Recover the production `best.pt` checkpoint from Google Drive, verify cryptographic integrity, implement the standalone CPU evaluation pipeline, and independently replay Protocol O validation metrics.

## 2. Inputs & Dependencies
- Checkpoint: `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`).
- Evaluator: `audit_scripts/evaluate_fold1_ctc_binary.py`.
- Validation set: 300 utterances (`NJS`, `RRBI`) from authorized fold package.

## 3. Work Performed
1. Staged `best.pt` locally and verified SHA-256 matched the training checkpoint ledger exactly.
2. Implemented `evaluate_fold1_ctc_binary.py` for headless CPU evaluation without training dependencies.
3. Executed forward passes across all 300 validation utterances under Protocol O (standard TextGrid phone boundaries).
4. Profiled per-speaker AP (`NJS` vs `RRBI`) and matched vs mismatched audio acoustic controls.

## 4. Results & Verification
- Acceptance gate: `CPU_REPLAY: PASS`.
- Validation Binary AP replicated bit-identically: **0.316062** (matches Colab online validation).
- Greedy CTC PER replicated: **23.64%** (matches Colab online validation).
- Documented in [INDEPENDENT_VALIDATION_REPLAY.md](../evaluation_reconciliation/INDEPENDENT_VALIDATION_REPLAY.md) and [ACOUSTIC_CONTRIBUTION_CONTROLS.md](../evaluation_reconciliation/ACOUSTIC_CONTRIBUTION_CONTROLS.md).

## 5. Output Artifacts
- Evaluation records: `manifests/RQ1-TRN-02A/evaluation_reconciliation/INDEPENDENT_VALIDATION_REPLAY.json`, `CPU_SMOKE_TEST_REPORT.json`.
- Evaluator script: `audit_scripts/evaluate_fold1_ctc_binary.py`.
- Test suite: `tests/test_cpu_evaluator.py`.

## 6. Restart / Recovery Instructions
- Run CPU evaluator on recovered checkpoint:
  ```bash
  python audit_scripts/evaluate_fold1_ctc_binary.py --checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt
  ```
- Fast read-only CPU script.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Evaluation without oracle TextGrid phone sequences (Protocol Z).
- Prohibited actions: Never modify recovered checkpoint weights.
- Next step: [03-protocol-z-zero-oracle-and-numerical-closure.md](03-protocol-z-zero-oracle-and-numerical-closure.md).
