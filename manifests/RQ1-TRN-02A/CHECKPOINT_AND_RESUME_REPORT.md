# Checkpoint and Resume Report — Fold 1

**Status: COMPLETED & VERIFIED | Remote Storage: Google Drive**

---

## 1. Persisted Checkpoint Artifacts

Checkpoints were produced during Fold 1 training on Google Colab Tesla T4 GPU (`rq1-trn02a-restart-01`) and synchronized to Google Drive immediately upon completion of each epoch.

| Artifact Name | Size (Bytes) | Epoch | Successful Updates | Micro Steps | Best AP | Remote Path on Google Drive |
|---|---:|---:|---:|---:|---:|---|
| **`best.pt`** | 412,366,029 | 6 | 3,600 | 7,200 | **0.316062** | `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/best.pt` |
| **`last.pt`** | 412,367,181 | 11 | 6,600 | 13,200 | 0.304382 | `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/last.pt` |
| **`history.json`** | 8,819 | 11 | 6,600 | 13,200 | 0.316062 | `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/history.json` |
| **`EVALUATION_REPORT.json`** | 5,611 | 6 | 3,600 | 7,200 | 0.316062 | `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/EVALUATION_REPORT.json` |
| **`BASELINE_COMPARISON.json`** | 1,227 | 6 | 3,600 | 7,200 | 0.316062 | `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/BASELINE_COMPARISON.json` |

---

## 2. Checkpoint State Completeness

Every checkpoint contains a complete, self-consistent state dictionary:
- **`model`**: Full state dict for `LangfensWav2Vec2MDD` (trainable CTC head, cross-attention, binary head; frozen Wav2Vec2 backbone; frozen diagnosis head).
- **`optimizer`**: AdamW optimizer internal states (momentum, variance, step counts per parameter group).
- **`scheduler`**: `get_cosine_schedule_with_warmup` internal state.
- **`scaler`**: `torch.amp.GradScaler` current scale and step counters.
- **`epoch`**: 0-indexed completed epoch index.
- **`successful_updates`**: Cumulative optimizer update count (accumulated batches).
- **`micro_steps`**: Cumulative micro-batch forward-backward steps.
- **`best_ap`**: Predeclared best validation AP achieved so far.
- **`patience`**: Early stopping counter (persisted across restarts to guarantee contract continuity).
- **`skipped_amp_attempts_total`**: Count of scaler-skipped optimizer steps (0 across entire training run).
- **`consecutive_skips`**: Consecutive skipped steps counter.
- **`history`**: Complete history list of per-epoch metrics up to current epoch.
- **`rng`**: Exact random generator states for `random`, `numpy`, `torch` CPU, and `torch.cuda`.
- **`metadata`**: Exact Git commit SHA (`caa56783113d40eac438e5f458cdb313b8932218`), dataset split SHA-256, and full YAML training configuration.

---

## 3. Atomic Synchronization and Verification

1. **Local VM Staging**: The checkpoint is saved first to a `.tmp` file on the VM disk (`/content/repo/checkpoints/RQ1-TRN-02A/`) and renamed atomically with `os.replace`.
2. **Drive Remote Copy**: The completed local file is copied to a `.tmp` destination in the Google Drive mount (`/content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/`), then atomically replaced.
3. **Verification**: Remote existence and exact byte-for-byte size match are verified after every epoch before declaring the epoch safely persisted.

---

## 4. Resume Continuity Verification

- **Integration Gate Test**:
  - Epoch 1 was trained from scratch -> saved to `last.pt` and `best.pt`.
  - Driver was invoked with `--resume checkpoints/RQ1-TRN-02A/last.pt --num-epochs 1`.
  - Resumed seamlessly at `start_epoch=1`, `updates=600`, `best_ap=0.309634`, `patience=0`.
  - Completed Epoch 2, advancing `updates` to 1,200 and micro-steps to 2,400.
- **Full Run Continuity**:
  - Resumed from Epoch 2 (`start_epoch=2`, `updates=1200`, `patience=1`).
  - Ran through Epoch 11, where early stopping patience reached 5, terminating cleanly.
  - Total updates: 6,600. Total micro-steps: 13,200.
