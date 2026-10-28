# LANGFENS — RQ1-TRN-02A / 03-Protocol Z Zero-Oracle and Numerical Closure

**Status:** COMPLETED

**Recorded at:** 2026-10-26T16:05:00+07:00

**Evidence source:** Commits `613dc72`, `03854dc`, `9b23d82`, `c4a658f`, `manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Implement and verify the Zero-Oracle Protocol Z evaluation pipeline (generating prompt phone sequences via G2P without gold TextGrids), resolve dynamic programming tie-break sensitivity, and establish definitive numerical closure.

## 2. Inputs & Dependencies
- Evaluator: `audit_scripts/evaluate_protocol_z.py`.
- CMUDict: `resources/CMUDict/cmudict.dict` (commit `74790861`).
- Checkpoint: `best.pt` (SHA-256 `a2445c8d...`).
- Verification suite: `tests/test_protocol_z_alignment_and_scoring.py`.

## 3. Work Performed
1. Implemented G2P frontend mapping text prompts to ARPAbet sequences with dynamic programming alignment to acoustic representations.
2. Evaluated 9,919 aligned phone pairs on the 300 validation utterances (99.63% gold position coverage).
3. Repaired DP tie-break sensitivity across `sub_first`, `del_first`, and `ins_first` alignment policies (commit `9b23d82`).
4. Reconciled true training provenance metadata in evaluator (asserts `epoch == 5`, `successful_updates == 3600`, training SHA `caa5678...`).
5. Closed numerical discrepancies in `NUMERICAL_CLOSURE_REPORT.md` (commit `c4a658f`), confirming confusion counts directly from exported prediction records.

## 4. Results & Verification
- Acceptance gate: `PROTOCOL_Z_CLOSURE: PASS`.
- Aligned-only Protocol Z AP: **0.313530** (9,919 tokens, 1,145 errors; prevalence 0.115435).
- Confusion at $\tau=0.20$: TP=605, FP=1,510, FN=540, TN=7,264; F1=**0.3712**.
- Confusion at $\tau=0.50$: TP=208, FP=213, FN=937, TN=8,561; F1=**0.2656**.
- Acoustic controls: Zero acoustic AP = 0.300588, Random acoustic AP = 0.300843; net acoustic lift = **+0.012942**.
- Documented in [NUMERICAL_CLOSURE_REPORT.md](../evaluation_reconciliation/zero_oracle_verification/NUMERICAL_CLOSURE_REPORT.md) and [FINAL_ERRATA_REPORT.md](../evaluation_reconciliation/zero_oracle_verification/final_errata/FINAL_ERRATA_REPORT.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/*.md`, `final_errata/*.md`.
- Records: `PROTOCOL_Z_PREDICTION_RECORDS.json`, `PROTOCOL_Z_VERIFICATION_REPORT.json`.
- Test suite: `tests/test_protocol_z_alignment_and_scoring.py` (11 passed).

## 6. Restart / Recovery Instructions
- Run Protocol Z regression tests:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_protocol_z_alignment_and_scoring.py
  ```
- Fast read-only CPU validation.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: None for Fold 1 evaluation; test split remains sealed.
- Prohibited actions: Never use Protocol Z metrics to select model checkpoints (exploratory eval only).
- Next phase: [RQ1-ARTIFACT-01](../../RQ1-ARTIFACT-01/PROCESS_CHECKPOINT.md).
