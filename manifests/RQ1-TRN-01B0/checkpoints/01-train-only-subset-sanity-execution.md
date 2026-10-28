# LANGFENS — RQ1-TRN-01B0 / 01-Train-Only Subset Sanity Execution

**Status:** COMPLETED

**Recorded at:** 2026-10-23T22:45:00+07:00

**Evidence source:** Commit `1c5dc61`, `manifests/RQ1-TRN-01B0/TRAINING_SANITY_REPORT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Execute a controlled 5-epoch training sanity run on a 100-utterance train-only subset (B0-SUBSET), verify gradient backward passes and loss reduction, and ensure zero test split leakage.

## 2. Inputs & Dependencies
- Dataset: `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`.
- Manifest: `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json`.
- Runner: `audit_scripts/run_training_sanity.py`.

## 3. Work Performed
1. Sampled 100 utterances strictly from Fold 1 training speakers (`BWC`, `HKK`, `HQTV`, etc.).
2. Executed 5 training epochs with AdamW (lr=1e-3, batch size 2, gradient accumulation 2).
3. Monitored per-epoch CTC, Binary, and Diagnosis losses and GPU memory consumption.
4. Audited data leakage asserting zero overlap with validation or test speakers.

## 4. Results & Verification
- Acceptance gate: `TRAINING_SANITY: PASS`.
- Losses converged smoothly: Train loss dropped from 4.82 (epoch 1) to 2.15 (epoch 5).
- Zero data leakage verified: All 100 utterances confirmed to belong strictly to training speakers.

## 5. Output Artifacts
- Manifest: `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json`.
- Reports: `manifests/RQ1-TRN-01B0/TRAINING_SANITY_REPORT.md`, `DATA_LEAKAGE_AUDIT.md`.

## 6. Restart / Recovery Instructions
- Re-run sanity script:
  ```bash
  python audit_scripts/run_training_sanity.py --dry-run
  ```
- Lightweight check.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: High CTC decoding error discovered during post-hoc inspection.
- Prohibited actions: Never include validation utterances in the sanity subset.
- Next step: [02-learning-diagnostics-and-reconciliation.md](02-learning-diagnostics-and-reconciliation.md).
