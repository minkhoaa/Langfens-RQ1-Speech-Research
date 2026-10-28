# LANGFENS — RQ1-TRN-01B0 / 02-Learning Diagnostics and Reconciliation

**Status:** COMPLETED_WITH_LIMITATIONS

**Recorded at:** 2026-10-23T23:50:00+07:00

**Evidence source:** Commit `d63f25f`, `manifests/RQ1-TRN-01B0/reconciliation/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Conduct forensic post-hoc analysis of RQ1-TRN-01B0 learning dynamics, diagnose the root cause of CTC blank collapse, correct step accounting discrepancies, and assess substitution diagnosis feasibility.

## 2. Inputs & Dependencies
- Checkpoints and training logs from `manifests/RQ1-TRN-01B0/`.
- Evaluation script: `audit_scripts/run_training_sanity.py`.

## 3. Work Performed
1. Evaluated CTC greedy decoding on trained B0 checkpoint: discovered greedy PER near 100% with >98% blank predictions.
2. Identified mathematical mechanism: without learning rate warm-up, the randomly initialized linear CTC head rapidly collapses to the majority blank class ($c=0$) to minimize CTC loss.
3. Rectified step accounting: resolved confusion between micro-steps (50 forward passes) and actual optimizer updates (25 parameter updates under accumulation factor 2).
4. Analyzed substitution diagnosis: found 39-class diagnosis head cannot generalize on 100 utterances without pretrained phoneme embeddings.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B0-R: PASS (WITH KNOWN LIMITATIONS)`.
- Root cause established for CTC blank collapse: absence of LR warmup and unconstrained initial learning rate.
- Documented in [CTC_DECODING_AND_PER_DIAGNOSTICS.md](../reconciliation/CTC_DECODING_AND_PER_DIAGNOSTICS.md) and [GPU_AND_STEP_ACCOUNTING_ERRATUM.md](../reconciliation/GPU_AND_STEP_ACCOUNTING_ERRATUM.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01B0/reconciliation/CTC_DECODING_AND_PER_DIAGNOSTICS.md`, `GPU_AND_STEP_ACCOUNTING_ERRATUM.md`, `SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`, `CHECKPOINT_CONTINUITY_REPORT.md`.
- Metrics: `manifests/RQ1-TRN-01B0/reconciliation/CORRECTED_LEARNING_METRICS.json`.

## 6. Restart / Recovery Instructions
- Diagnostic inspection only:
  ```bash
  python -c "import json; data=json.load(open('manifests/RQ1-TRN-01B0/reconciliation/CORRECTED_LEARNING_METRICS.json')); print('Final loss:', data['final_loss'])"
  ```
- No authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: CTC stabilization required before baseline training.
- Prohibited actions: Never begin full fold training with unmitigated CTC blank collapse.
- Next phase: [RQ1-TRN-01B1](../../RQ1-TRN-01B1/PROCESS_CHECKPOINT.md).
