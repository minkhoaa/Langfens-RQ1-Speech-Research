# LANGFENS — RQ1-TRN-01B0 / Controlled Training Sanity Gate

**Status:** COMPLETED_WITH_LIMITATIONS

**Recorded at:** 2026-10-23T23:55:00+07:00

**Evidence source:** Commits `1c5dc61`, `d63f25f`, `manifests/RQ1-TRN-01B0/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Validate end-to-end multi-task training on a controlled 100-utterance real-data train-only subset (B0-SUBSET), verify gradient flow, audit partition isolation, and analyze initial learning dynamics and metric behaviors.

## 2. Inputs & Dependencies
- Source branch: `feat/rq1-trn-01b0-training-sanity-gate` (commits `1c5dc61`, `d63f25f`).
- Split: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`.
- Train-only subset: `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json` (100 utterances, 0 validation/test overlap).
- Model: `src/langfens_prep/model.py`.
- Training runner: `audit_scripts/run_training_sanity.py`.

## 3. Work Performed
1. Curated leak-free 100-utterance training subset (B0-SUBSET) from Fold 1 train speakers.
2. Executed 5-epoch sanity training run on local GPU/CUDA environment, monitoring multi-task losses.
3. Audited gradient norms across CTC, cross-attention, binary, and diagnosis heads.
4. Conducted forensic analysis in `RQ1-TRN-01B0-R` following discovery of high CTC error rates.
5. Diagnosed CTC blank collapse (uninitialized linear CTC head rapidly collapses to emitting blank tokens under AdamW lr=1e-3).
6. Issued step accounting erratum (corrected 25 optimizer updates vs 50 micro-steps) and diagnosed substitution metric limitations.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B0: PASS (SANITY)` / `RQ1-TRN-01B0-R: PASS (WITH LIMITATIONS)`.
- Loss reduction confirmed: Total loss decreased from 4.82 to 2.15 across 5 epochs.
- Zero data leakage confirmed: Strict disjointness between B0-SUBSET and Fold 1 val/test.
- Known limitations & failures:
  - CTC blank collapse observed: greedy PER ~100%, >98% blank emission due to lack of CTC warmup and high initial learning rate.
  - Multi-class substitution diagnosis head showed random guessing accuracy (~2.5%) on small subset.
- Documented in [TRAINING_SANITY_REPORT.md](TRAINING_SANITY_REPORT.md), [LOSS_AND_GRADIENT_DIAGNOSTICS.md](LOSS_AND_GRADIENT_DIAGNOSTICS.md), and [CTC_DECODING_AND_PER_DIAGNOSTICS.md](reconciliation/CTC_DECODING_AND_PER_DIAGNOSTICS.md).

## 5. Output Artifacts
- Manifest: `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json`.
- Reports: `manifests/RQ1-TRN-01B0/TRAINING_SANITY_REPORT.md`, `LOSS_AND_GRADIENT_DIAGNOSTICS.md`, `CHECKPOINT_REPRODUCIBILITY_REPORT.md`, `DATA_LEAKAGE_AUDIT.md`.
- Reconciliation addenda: `manifests/RQ1-TRN-01B0/reconciliation/*.md`, `CORRECTED_LEARNING_METRICS.json`.
- Measurements: `manifests/RQ1-TRN-01B0/GPU_RUNTIME_MEASUREMENTS.json`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.11 with CUDA (`.venv-cuda`) or local GPU.
- Run sanity training script:
  ```bash
  python audit_scripts/run_training_sanity.py
  ```
- Authorized for diagnostic validation only; no backward passes on full dataset without Colab.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: CTC blank collapse must be stabilized before proceeding to full baseline training.
- Prohibited actions: Never deploy un-warmed CTC heads with lr=1e-3; do not treat B0 sanity metrics as baseline results.
- Next phase: [RQ1-TRN-01B1](../RQ1-TRN-01B1/PROCESS_CHECKPOINT.md).
