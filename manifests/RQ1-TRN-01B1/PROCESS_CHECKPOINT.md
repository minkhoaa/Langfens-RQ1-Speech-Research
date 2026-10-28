# LANGFENS — RQ1-TRN-01B1 / CTC Stabilization and Pilot Experiments

**Status:** COMPLETED

**Recorded at:** 2026-10-25T19:30:00+07:00

**Evidence source:** Commits `25ffb08`, `493b7ab`, `73eae7e`, `manifests/RQ1-TRN-01B1/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Eliminate CTC blank collapse through learning rate warmup schedules and architectural pilots, execute 2x2 factorial generalization experiments (frozen vs unfrozen backbone, learning rates 1e-3 vs 1e-4), and empirically validate acoustic feature conditioning for phoneme-level MDD.

## 2. Inputs & Dependencies
- Source branch: `feat/rq1-trn-01b1-ctc-stabilization` through `feat/rq1-trn-01b1-r2-acoustic-gate` (commits `25ffb08`, `493b7ab`, `73eae7e`).
- Dataset: B0-SUBSET and internal train-holdout protocol (`manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json`).
- Pretrained backbone: `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).
- Pilot runner: `audit_scripts/run_ctc_stabilization_experiments.py`.

## 3. Work Performed
1. Tested four stabilization variants (A: baseline 1e-3, B: reduced lr 1e-4, C: linear warmup, D: cosine warmup with head lr=1e-3) in RQ1-TRN-01B1 (`25ffb08`).
2. Confirmed Variant D completely eliminates CTC blank collapse, reducing greedy PER from 100% to ~28-30% and recovering full phoneme decoding length.
3. Conducted 2x2 factorial ablation across 4 training cells in `RQ1-TRN-01B1-R` (`493b7ab`), proving frozen backbone with lr 1e-3 yields best generalization.
4. Corrected majority baseline erratum and established strict train-only holdout protocol.
5. In `RQ1-TRN-01B1-R2` (`73eae7e`), performed acoustic conditioning ablation: proved multi-task Wav2Vec 2.0 with cross-attention substantially outperforms canonical-only text prior baselines.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B1: PASS`, `RQ1-TRN-01B1-R: PASS`, `RQ1-TRN-01B1-R2: PASS`.
- CTC recovery: Decoded length ratio restored to 0.96–0.98; blank token fraction stabilized at ~79%.
- Factorial verdict: Frozen backbone + lr 1e-3 achieves superior convergence and prevents catastrophic forgetting.
- Acoustic conditioning verdict: Acoustic features provide authentic statistical lift over canonical text prior.
- Documented in [CTC_STABILIZATION_EXPERIMENTS.md](CTC_STABILIZATION_EXPERIMENTS.md), [FACTORIAL_ABLATION_COMPARISON.md](reconciliation/FACTORIAL_ABLATION_COMPARISON.md), and [DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md](reconciliation/final_acoustic_gate/DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01B1/*.md`, `manifests/RQ1-TRN-01B1/reconciliation/*.md`, `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/*.md`.
- Experiment metrics: `manifests/RQ1-TRN-01B1/metrics_Variant_*.json`, `FACTORIAL_2X2_METRICS.json`, `canonical_only_baselines.json`.
- Test suite: `tests/test_ctc_stabilization_and_pilot.py`, `tests/test_rq1_trn_01b1_r_generalization_and_reconciliation.py`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.11 with CUDA (`.venv-cuda`) or local GPU.
- Run stabilization regression tests:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_ctc_stabilization_and_pilot.py
  ```
- Fast read-only CPU tests; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Scaling to full 2,399-utterance Fold 1 training.
- Prohibited actions: Never train full Fold 1 on local laptop; reserved strictly for Google Colab GPU runtimes.
- Next phase: [RQ1-TRN-02A](../RQ1-TRN-02A/PROCESS_CHECKPOINT.md).
