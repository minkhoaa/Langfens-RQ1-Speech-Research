# RQ1-TRN-02A — Preflight and Split Integrity Report

**Decision: PASS — Full Fold 1 Clean Restart, Preflight, and Training Completed**

---

## 1. Provenance and Environment

- **Dedicated Feature Branch**: `feat/rq1-trn-02a-colab-restart`
- **Starting Base SHA**: `9b32800bde2c9fbcfd9814aeb651eab57d4ce486`
- **Execution Commit**: `caa56783113d40eac438e5f458cdb313b8932218` (Pushed to GitHub; verified remote HEAD)
- **Repository Remote**: `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git`
- **Remote `main`**: Unaltered, separate, and unmerged.
- **Root Governance Policy**: Audited `AGENTS.md` integrated (34 lines).

---

## 2. Dataset and Partition Integrity

- **Split Manifest**: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`
- **Manifest Checksum**: SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (Verified exact match).
- **Fold 1 Partition Counts**:
  - `TRAIN`: 2,399 utterances across 16 speakers.
  - `Validation`: 300 utterances across 2 speakers (`NJS`, `RRBI`).
  - `TEST`: 900 utterances across 6 speakers. **STRICTLY SEALED** (zero test utterances staged or accessed).
- **Reusable Staged Archive**:
  - Path: `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip`
  - Size: 761,922,918 bytes
  - SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`
  - Local VM disk SHA-256 verified after transfer. Exactly 2,699 WAVs and 2,699 TextGrids; zero test speaker records.

---

## 3. Pretrained Backbone Integrity

- **Pretrained Model**: `facebook/wav2vec2-base-960h`
- **Pinned Revision**: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- **Weights Integrity**: Pretrained Wav2Vec2 encoder kept strictly frozen in `eval()` mode. Zero gradients allowed in backbone.
- **Task Heads**: CTC phone recognition head and Canonical Binary MDD head initialized freshly and trained jointly with AdamW.
- **Substitution Diagnosis Head**: Preserved in model architecture but completely frozen and excluded from loss and optimizer.

---

## 4. Production Execution and Persistence Summary

- **Google Colab Session**: `rq1-trn02a-restart-01` (Tesla T4, 15,360 MiB VRAM).
- **Execution Run**: 11 completed epochs (13,200 micro-steps, 6,600 optimizer updates, 0 AMP skips).
- **Early Stopping**: Triggered after Epoch 11 due to 5 epochs without validation AP improvement beyond Epoch 6.
- **Selected Checkpoint**: Epoch 6 (`best.pt`, Validation AP = 0.316062, Validation PER = 0.236413).
- **Google Drive Persistence**: Atomically written and verified at `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/`.
