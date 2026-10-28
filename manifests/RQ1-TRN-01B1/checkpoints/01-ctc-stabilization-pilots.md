# LANGFENS — RQ1-TRN-01B1 / 01-CTC Stabilization Pilots

**Status:** COMPLETED

**Recorded at:** 2026-10-24T15:45:00+07:00

**Evidence source:** Commit `25ffb08`, `manifests/RQ1-TRN-01B1/CTC_STABILIZATION_EXPERIMENTS.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Design and test four hyperparameter pilot variants (A, B, C, D) to eliminate CTC blank collapse, recover greedy phoneme sequence lengths, and verify bit-level checkpoint resumption determinism.

## 2. Inputs & Dependencies
- Architecture: `LangfensWav2Vec2MDD`.
- Training subset: B0-SUBSET (100 utterances).
- Runner: `audit_scripts/run_ctc_stabilization_experiments.py`.

## 3. Work Performed
1. Configured 4 pilot arms: Variant A (constant lr=1e-3), Variant B (constant lr=1e-4), Variant C (linear warmup 5 epochs, lr=1e-3), Variant D (cosine warmup 5 epochs, peak lr=1e-3).
2. Monitored greedy decoded sequence length ratio ($L_{\text{hyp}} / L_{\text{ref}}$) and blank token emission fraction.
3. Conducted resume determinism test verifying bitwise loss reproduction after loading mid-run checkpoints.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B1: PASS`.
- Variant D completely resolved blank collapse: decoded length ratio increased from 0.02 (Variant A) to 0.97 (Variant D).
- CTC greedy PER reduced to 28.4% on train holdout.
- Resume determinism verified: identical loss values produced before and after restart.
- Documented in [CTC_STABILIZATION_EXPERIMENTS.md](../CTC_STABILIZATION_EXPERIMENTS.md) and [CHECKPOINT_RESUME_DETERMINISM_REPORT.md](../CHECKPOINT_RESUME_DETERMINISM_REPORT.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01B1/CTC_STABILIZATION_EXPERIMENTS.md`, `CTC_BLANK_COLLAPSE_ANALYSIS.md`, `CHECKPOINT_RESUME_DETERMINISM_REPORT.md`.
- Metrics: `manifests/RQ1-TRN-01B1/metrics_Variant_A.json` through `metrics_Variant_D.json`.

## 6. Restart / Recovery Instructions
- Run pilot regression test suite:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_ctc_stabilization_and_pilot.py
  ```
- Lightweight CPU checks.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Generalization testing on larger holdouts.
- Prohibited actions: Never use constant lr=1e-3 without warmup for CTC head.
- Next step: [02-factorial-generalization-and-metric-erratum.md](02-factorial-generalization-and-metric-erratum.md).
