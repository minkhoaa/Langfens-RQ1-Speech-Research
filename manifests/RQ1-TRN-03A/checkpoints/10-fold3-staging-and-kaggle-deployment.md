# LANGFENS — RQ1-TRN-03A / 10-Fold 3 Staging & Kaggle Deployment

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-30T15:50:00+07:00

**Evidence source:** `configs/rq1_trn_03a_fold3.yaml`, `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip`, `deploy/kaggle_fold3_staging/`, `deploy/kaggle_fold3_preflight/`, `deploy/kaggle_fold3_smoke/`, `deploy/kaggle_fold3_train/`, `audit_scripts/build_simplified_kaggle_deployment.py`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, 289/289 passed, 1 skipped)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

This checkpoint prepares the complete, leak-free remote execution infrastructure for **Fold 3 Baseline R Training** on Kaggle Dual Tesla T4 GPUs (`KAGGLE_T4_CUDA_FP16_AMP`), fully replicating the proven operational and scientific workflow established and verified in Fold 2:

1. **Fold 2 Completion & Verification Certified:**
   - Fold 2 smoke run verified (1 completed epoch, ~600 updates, Val Binary AP: `0.1960`, Val CTC PER: `30.44%`).
   - Fold 2 official training run verified (7 completed epochs, 4,200 total updates, 761.9s runtime, early stopping patience 5 on validation AP).
   - Dual checkpoint accounting verified on private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints`:
     - Fixed-budget primary weights: `fold2/baseline_r_fold2/fixed_budget_checkpoint.pt` (Epoch 6 / 3,600 updates, SHA-256 `ab4c589445feac...`).
     - Validation-selected peak weights: `fold2/baseline_r_fold2/best.pt` (Epoch 2 / 1,200 updates, Val AP `0.2170`, SHA-256 `b9032ee9ba8d...`).
     - Phase 6 post-training classification passed with `COMPLETED_3600_UPDATES`.

2. **Fold 3 Pinned Configuration Verified:**
   - `configs/rq1_trn_03a_fold3.yaml` validated against frozen Baseline R contract:
     - Pretrained backbone: `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, frozen).
     - Fresh initialization: zero warm-start or checkpoint transfer from Fold 1 or Fold 2.
     - Hyperparameters: `seed: 1337`, `batch_size: 2`, `gradient_accumulation_steps: 2`, AdamW (`lr: 1e-3`, `weight_decay: 0.01`), FP16 AMP (`grad_scaler_initial_scale: 4096.0`), gradient clip `1.0`.
     - Schedule: 25-epoch cosine decay horizon (~15,000 updates, 100 warmup updates), 10-epoch execution cap (max 6,000 updates), early stopping patience 5 on validation AP.

3. **Fold 3 Authorized Train/Val Package & Test Isolation Audit:**
   - Package: `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip` (760,283,230 bytes).
   - Cryptographic SHA-256: `e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2`.
   - Authorized Speakers Present (18):
     - Train (16): `ABA`, `ASI`, `BWC`, `EBVS`, `HJK`, `HKK`, `HQTV`, `LXC`, `MBMPS`, `NJS`, `PNV`, `SKA`, `SVBI`, `TLV`, `TNI`, `YKWK`.
     - Val (2): `TXHC`, `ZHAA`.
   - Active Test Partition Speakers Strictly Excluded (6):
     - `ERMS`, `NCC`, `RRBI`, `THV`, `YBAA`, `YDCK` are 100% physically absent (0 audio, 0 TextGrid). Zero active test leakage proved.

4. **Kaggle Fold 3 Staging & 3-Package Entrypoint Deployment:**
   - **Staging Directory (`deploy/kaggle_fold3_staging/`):** Contains `dataset-metadata.json` for private Kaggle dataset `minkhoaa/langfens-rq1-fold3-staging`.
   - **Preflight Package (`deploy/kaggle_fold3_preflight/`):** Dedicated preflight notebook running `verify_fold3` (manifest SHA, dataset SHA, speaker isolation, live Hugging Face write-readback-delete probe).
   - **Smoke Package (`deploy/kaggle_fold3_smoke/`):** Dedicated 1-epoch smoke execution notebook running preflight gate, then `train_fold3` with `--run-id fold3_smoke_1epoch` and `--execution-cap-epochs 1`.
   - **Official Training Package (`deploy/kaggle_fold3_train/`):** Dedicated Baseline R execution notebook running preflight gate, then `train_fold3` with `--run-id baseline_r_fold3`, locking `fixed_budget_checkpoint.pt` at Epoch 6 / 3,600 updates, and Phase 6 post-training classification.

5. **Operational Safety & Boundary Compliance:**
   - Zero GPU execution dispatched during local staging.
   - Zero remote dataset uploads performed (pending operator authorization).
   - Zero active test partition data accessed or staged.
   - Zero credentials or tokens written to disk or exposed in CLI arguments.
   - 289 automated regression tests passed (1 skipped locally per `AGENTS.md` GPU prohibition).

---

## 2. Fold 3 Deployment Architecture & Specifications

| Dimension | Dataset Staging (`staging/`) | Preflight Package (`preflight/`) | Smoke Package (`smoke/`) | Official Training Package (`train/`) |
|---|---|---|---|---|
| **Package Directory** | `deploy/kaggle_fold3_staging/` | `deploy/kaggle_fold3_preflight/` | `deploy/kaggle_fold3_smoke/` | `deploy/kaggle_fold3_train/` |
| **Kaggle Slug** | `minkhoaa/langfens-rq1-fold3-staging` | `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-preflight` | `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-smoke` | `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2` |
| **Title** | `LANGFENS RQ1 Fold 3 Train/Val Staging Package` | `LANGFENS RQ1 Fold 3 Baseline R T4x2 (Preflight)` | `LANGFENS RQ1 Fold 3 Baseline R T4x2 (Smoke)` | `LANGFENS RQ1 Fold 3 Baseline R T4x2` |
| **Code File** | N/A (Dataset metadata) | `kaggle_preflight.ipynb` | `kaggle_smoke.ipynb` | `kaggle_train.ipynb` |
| **Pinned Commit** | N/A | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` |
| **Execution Mode** | N/A | Preflight only (`--mode verify_fold3`) | Preflight gate + 1-epoch smoke (`train_fold3`) | Preflight gate + 10-epoch Baseline R (`train_fold3`) |
| **Execution Cap** | N/A | 0 epochs (0 updates) | 1 completed epoch (~600 updates) | 10 completed epochs (max 6,000 updates; fixed budget at Epoch 6 / 3,600 updates) |
| **Run ID** | N/A | N/A | `fold3_smoke_1epoch` | `baseline_r_fold3` |
| **Local Checkpoint Path** | N/A | None | `/kaggle/working/checkpoints/fold3/fold3_smoke_1epoch/` | `/kaggle/working/checkpoints/fold3/baseline_r_fold3/` |
| **Remote Mirror Prefix** | N/A | Probe verification only (`_probe_write_test`) | `fold3/fold3_smoke_1epoch/` | `fold3/baseline_r_fold3/` |
| **Machine Target** | N/A | `NvidiaTeslaT4` (Dual T4 FP16 AMP) | `NvidiaTeslaT4` (Dual T4 FP16 AMP) | `NvidiaTeslaT4` (Dual T4 FP16 AMP) |
| **Dataset Source** | N/A | `minkhoaa/langfens-rq1-fold3-staging` | `minkhoaa/langfens-rq1-fold3-staging` | `minkhoaa/langfens-rq1-fold3-staging` |

---

## 3. Verification Evidence & Test Ledger

- **Suite Execution:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/`
- **Result:** **289 passed, 1 skipped, 1 warning in 173.49s** (100% green).
- **Key Regression Tests Verified:**
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold3_three_entrypoints_isolation`: PASS. Proves Fold 3 preflight, smoke, and official train packages possess mutually exclusive kernel IDs and matching titles without collision.
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold3_staging_and_speaker_isolation`: PASS. Proves Fold 3 staging metadata and authorized archive (`authorized_RQ1-TRN_fold3_train_val_only.zip`) match SHA-256 `e81ac9e5d95a...` and strictly exclude all 6 active test speakers (`ERMS`, `NCC`, `RRBI`, `THV`, `YBAA`, `YDCK`).
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold3_training_invariants`: PASS. Proves Fold 3 training package enforces preflight-first gate, explicit run ID `baseline_r_fold3`, local output path `/kaggle/working/checkpoints/fold3/baseline_r_fold3`, and Phase 6 post-training classification.

---

## 4. Operator Deployment Protocol (Exact Commands for Step 11)

When operator authorization is granted, execute the deployment in four stages following the permanent Kaggle workflow:

### Stage 1: Stage and Create Private Fold 3 Dataset on Kaggle
1. Prepare staging archive on laptop disk (symlink or copy binary):
   ```bash
   cp datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip deploy/kaggle_fold3_staging/authorized_RQ1-TRN_fold3_train_val_only.bin
   ```
2. Upload private dataset via Kaggle CLI:
   ```bash
   kaggle datasets create -p deploy/kaggle_fold3_staging/
   ```
3. Remove local temporary binary copy:
   ```bash
   rm deploy/kaggle_fold3_staging/authorized_RQ1-TRN_fold3_train_val_only.bin
   ```

### Stage 2: Push Preflight Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold3_preflight/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-preflight`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to verify preflight staging and Hugging Face probe write.

### Stage 3: Push Smoke Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold3_smoke/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-smoke`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to run 1-epoch smoke test and verify Hugging Face persistence.

### Stage 4: Push Official Training Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold3_train/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to execute official 10-epoch Baseline R training on Fold 3.
