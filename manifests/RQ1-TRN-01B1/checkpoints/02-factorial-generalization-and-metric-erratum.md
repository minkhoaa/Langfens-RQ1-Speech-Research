# LANGFENS — RQ1-TRN-01B1 / 02-Factorial Generalization and Metric Erratum

**Status:** COMPLETED

**Recorded at:** 2026-10-24T18:15:00+07:00

**Evidence source:** Commit `493b7ab`, `manifests/RQ1-TRN-01B1/reconciliation/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Execute a 2x2 factorial ablation evaluating backbone fine-tuning against frozen representations, correct majority-class baseline reporting errata, and establish rigorous internal train-holdout validation.

## 2. Inputs & Dependencies
- Training protocol: `manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json`.
- Runner: `audit_scripts/run_rq1_trn_01b1_reconciliation_campaign.py`.

## 3. Work Performed
1. Trained 4 factorial cells: Cell 1 (frozen backbone, lr 1e-3), Cell 2 (frozen backbone, lr 1e-4), Cell 3 (unfrozen backbone, lr 1e-3), Cell 4 (unfrozen backbone, lr 1e-4).
2. Corrected majority-class baseline erratum in evaluation scripts: correctly reported naive majority classifier F1 is 0.0 for binary mispronunciation detection (minority class $E=1$).
3. Assessed acoustic dependence and checkpoint continuity across seeds.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B1-R: PASS`.
- Factorial outcome: Cell 1 (frozen backbone, lr 1e-3) achieved lowest PER and highest Binary AP without catastrophic forgetting.
- Unfreezing the backbone led to severe acoustic representation degradation under small sample regimes.
- Documented in [FACTORIAL_ABLATION_COMPARISON.md](../reconciliation/FACTORIAL_ABLATION_COMPARISON.md) and [MAJORITY_BASELINE_AND_METRIC_ERRATUM.md](../reconciliation/MAJORITY_BASELINE_AND_METRIC_ERRATUM.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01B1/reconciliation/FACTORIAL_ABLATION_COMPARISON.md`, `MAJORITY_BASELINE_AND_METRIC_ERRATUM.md`, `CTC_LENGTH_RECOVERY_REPORT.md`.
- Metrics: `manifests/RQ1-TRN-01B1/reconciliation/FACTORIAL_2X2_METRICS.json`, `metrics_cell_*.json`.

## 6. Restart / Recovery Instructions
- Run reconciliation regression tests:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_rq1_trn_01b1_r_generalization_and_reconciliation.py
  ```
- Lightweight CPU checks.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Verification of acoustic feature necessity over canonical text prior.
- Prohibited actions: Never unfreeze Wav2Vec 2.0 backbone without extensive multi-speaker data.
- Next step: [03-acoustic-conditioning-validation.md](03-acoustic-conditioning-validation.md).
