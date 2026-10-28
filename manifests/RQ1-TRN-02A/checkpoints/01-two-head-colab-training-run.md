# LANGFENS — RQ1-TRN-02A / 01-Two-Head Colab Training Run

**Status:** COMPLETED

**Recorded at:** 2026-10-25T22:40:00+07:00

**Evidence source:** Commits `b6f7dd3`, `caa5678`, `a6e8035`, `manifests/RQ1-TRN-02A/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Execute two-head (CTC + Binary MDD) Fold 1 baseline training on Google Colab Tesla T4 GPU, stream authorized data from local VM disk, and atomically synchronize epoch checkpoints to Google Drive.

## 2. Inputs & Dependencies
- Training config: `configs/rq1_trn_02a_fold1.yaml`, `manifests/RQ1-TRN-02A/TRAINING_CONFIG.yaml`.
- Authorized zip: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`.
- Split manifest: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- Driver: `audit_scripts/train_fold1_ctc_binary.py`.

## 3. Work Performed
1. Preflight verified VM storage, SHA-256 of authorized fold data, and Colab GPU environment.
2. Executed full Fold 1 training across 11 epochs (600 updates/epoch; effective batch size 4).
3. Trained CTC and Binary heads with cosine LR schedule; diagnosis head was explicitly kept frozen/untrained.
4. Atomically mirrored checkpoints (`last.pt`, `best.pt`, `history.json`) to Google Drive per epoch.
5. Achieved peak validation AP at epoch 6 (3,600 updates), saved as `best.pt`.
6. Terminated at epoch 11 after early stopping patience reached 5 (cumulative 6,600 updates).

## 4. Results & Verification
- Acceptance gate: `COLAB_TRAINING: PASS`.
- Selected checkpoint (`best.pt`): Epoch 6, 3,600 successful updates, val AP = 0.316062, val CTC PER = 23.64%.
- Full run termination: Epoch 11, 6,600 successful updates, val AP = 0.304382.
- Zero AMP scaler skips observed (`skipped_amp_attempts_total = 0`).
- Documented in [CHECKPOINT_AND_RESUME_REPORT.md](../CHECKPOINT_AND_RESUME_REPORT.md) and [TRAINING_HISTORY.json](../TRAINING_HISTORY.json).

## 5. Output Artifacts
- Checkpoints: `best.pt` (SHA-256 `a2445c8d...`), `last.pt` (SHA-256 `bf46fab3...`).
- Remote Drive: [`11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`](https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx).
- Training log: `manifests/RQ1-TRN-02A/TRAINING_HISTORY.json`.

## 6. Restart / Recovery Instructions
- Resuming training requires Google Colab GPU runtime and explicit operator authorization:
  ```bash
  # REQUIRES OPERATOR AUTHORIZATION — COLAB GPU ONLY:
  python audit_scripts/train_fold1_ctc_binary.py --config configs/rq1_trn_02a_fold1.yaml --resume checkpoints/RQ1-TRN-02A/last.pt
  ```
- Local execution strictly prohibited by AGENTS.md.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Checkpoint recovery and local evaluation replay.
- Prohibited actions: Never train diagnosis head without explicit architecture update; never touch sealed test data.
- Next step: [02-checkpoint-recovery-and-cpu-replay.md](02-checkpoint-recovery-and-cpu-replay.md).
