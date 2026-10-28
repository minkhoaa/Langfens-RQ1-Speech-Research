# LANGFENS — RQ1-TRN-02A / Full Fold 1 Baseline Training and Validation

**Status:** COMPLETED

**Recorded at:** 2026-10-26T16:15:00+07:00

**Evidence source:** Commits `b6f7dd3` through `c4a658f`, `manifests/RQ1-TRN-02A/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Execute full Fold 1 multi-task baseline training on Google Colab T4 GPU, audit remote checkpoint persistence to Google Drive, implement local CPU validation replay, and resolve evaluation protocols (Protocol O vs Protocol Z) with complete numerical closure.

## 2. Inputs & Dependencies
- Source commits: Training commit `caa56783113d40eac438e5f458cdb313b8932218`; evaluation reconciliation commits `3e2bc1d` through `c4a658f`.
- Split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- Authorized data: `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
- Pretrained backbone: `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, frozen).

## 3. Work Performed
1. Configured two-head training contract (CTC + Binary MDD only; multi-class diagnosis head was explicitly kept frozen and untrained).
2. Trained Fold 1 on Google Colab Tesla T4 GPU (`rq1-trn02a-restart-01`), synchronizing checkpoints atomically to Google Drive per epoch.
3. Selected production Baseline R checkpoint (`best.pt`, SHA-256 `a2445c8d...`) at epoch 6 (0-indexed epoch 5) after 3,600 successful updates.
4. Completed full training schedule through epoch 11 (6,600 successful updates) where early stopping terminated the run.
5. Implemented CPU evaluation pipeline (`audit_scripts/evaluate_fold1_ctc_binary.py`) and verified bit-identical Protocol O replay.
6. Evaluated Zero-Oracle Protocol Z (`audit_scripts/evaluate_protocol_z.py`) without gold TextGrid timestamps, repaired tie-break sensitivity, and achieved numerical closure.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-02A: PASS`, `PROTOCOL_Z_NUMERICAL_CLOSURE: PASS`.
- Selected Checkpoint (`best.pt`): Epoch 6 (index 5), 3,600 successful updates, val loss 0.3547.
- Complete Training Run (`last.pt`): Epoch 11 (index 10), 6,600 successful updates, early stopping patience 5.
- Training Head Contract: CTC and Binary heads trained; Diagnosis head was **NOT** trained.
- Evaluation Protocol Metrics (Distinguished Provenance):
  - **Protocol O (Oracle TextGrid tokens, 9,956 phones, 1,157 errors):** Binary AP = **0.316062** (2.72× over prevalence 0.116211); empirical prior AP = 0.303063 (acoustic delta = +0.012999); optimal F1 = 0.3775 at $\tau=0.20$; CTC PER = 23.64%.
  - **Protocol Z (Zero-Oracle G2P tokens, 9,919 aligned pairs, 1,145 errors):** Binary AP = **0.313530** (prevalence 0.115435); zero acoustic control AP = 0.300588 (net acoustic gain = +0.012942); F1 = 0.3712 at $\tau=0.20$.
- Documented in [FINAL_ERRATA_REPORT.md](evaluation_reconciliation/zero_oracle_verification/final_errata/FINAL_ERRATA_REPORT.md) and [NUMERICAL_CLOSURE_REPORT.md](evaluation_reconciliation/zero_oracle_verification/NUMERICAL_CLOSURE_REPORT.md).

## 5. Output Artifacts
- Checkpoint: `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`).
- Google Drive Checkpoints: [`11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`](https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx).
- Evaluation reports: `manifests/RQ1-TRN-02A/evaluation_reconciliation/` (commits `3e2bc1d` through `c4a658f`).
- Test suite: `tests/test_fold1_training_driver.py`, `tests/test_protocol_z_alignment_and_scoring.py`.

## 6. Restart / Recovery Instructions
- Environment: Python 3.14 CPU (`.venv`) with verified `best.pt`.
- Run Protocol Z verification:
  ```bash
  PYTHONPATH=src:. .venv/bin/pytest tests/test_protocol_z_alignment_and_scoring.py
  ```
- Re-run CPU evaluation replay:
  ```bash
  python audit_scripts/evaluate_fold1_ctc_binary.py --checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt
  ```
- Read-only CPU evaluation; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Official test split remains strictly sealed (900 utterances).
- Prohibited actions: Never use Protocol Z to select training checkpoints; never claim diagnosis head was evaluated.
- Next phase: [RQ1-ARTIFACT-01](../RQ1-ARTIFACT-01/PROCESS_CHECKPOINT.md) and [RQ1-TRN-02B](../RQ1-TRN-02B/PROCESS_CHECKPOINT.md).
