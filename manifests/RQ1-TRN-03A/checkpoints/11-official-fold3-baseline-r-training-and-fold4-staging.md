# LANGFENS — RQ1-TRN-03A / 11-Official Fold 3 Baseline R Training & Fold 4 Staging

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-31T15:35:00+07:00

**Evidence source:** Kaggle CLI, live remote kernel execution logs `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2`, smoke kernel logs `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-smoke`, private Hugging Face model repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`, `configs/rq1_trn_03a_fold4.yaml`, `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`, `deploy/kaggle_fold4_staging/`, `deploy/kaggle_fold4_preflight/`, `deploy/kaggle_fold4_smoke/`, `deploy/kaggle_fold4_train/`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, 292/292 passed, 1 skipped)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

This checkpoint certifies the successful remote execution, persistent cloud mirroring, and cryptographic verification of **Fold 3 Baseline R Training** on Kaggle Dual Tesla T4 GPUs (`KAGGLE_T4_CUDA_FP16_AMP`), and prepares the complete, leak-free deployment infrastructure for **Fold 4 Baseline R Training**:

1. **Fold 3 Smoke Gate Clearance (Phase A):**
   - Kernel `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-smoke` executed with `HF_TOKEN` Kaggle Secret.
   - Completed strictly 1 epoch (600 updates, 1,200 micro steps) in 105.79s wall time (subprocess total: 161.2s).
   - Val Binary AP: `0.283933` (prevalence: `0.113652`), Val CTC PER: `32.34%` (0.323364), Skipped AMP attempts: `0`.
   - Mirrored atomic checkpoints and reports to private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` under `fold3/fold3_smoke_1epoch/` with `training_endpoint_report.json` declaring expected status `INCOMPLETE`. Smoke gate declared **`PASS`**.

2. **Fold 3 Official Training Execution & Completion (Phase B):**
   - Kernel `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2` executed with `HF_TOKEN` Kaggle Secret.
   - Pinned training source commit verified: `cc56b82ca1a9a6ceeca85929e07db4cac8537348`.
   - Completed 10 training epochs (6,000 total optimizer updates, 12,000 micro steps) in 1005.9s (~16.8 min subprocess runtime, sum of epoch wall seconds: 791.9s) on Dual Tesla T4 GPUs under CUDA FP16 AMP.
   - Hit 10-epoch execution cap (`execution_cap_epochs: 10`) with early stopping patience counter reaching 4 of 5 (peak occurred at Epoch 6; Epochs 7–10 did not exceed Epoch 6 validation AP).
   - Skipped AMP updates: `0` across all 6,000 updates (100% numerical stability; zero scale collapses from initial `4096.0`).

3. **Fold 3 Dual Checkpoint Accounting & Endpoint Verification (Phase C):**
   - **Primary Fixed-Budget Endpoint (Epoch 6 / Exactly 3,600 Updates):**
     - Successfully locked and saved as `fixed_budget_checkpoint.pt`.
     - Mirrored atomically to Hugging Face with SHA-256 readback.
     - Size: 412,371,371 bytes | SHA-256: `bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441`.
     - Epoch 6 Metrics: Val Binary AP: `0.307479`, Val CTC PER: `24.80%` (0.247977), Train CTC Loss: `0.8182`, Train Binary Loss: `0.3703`.
   - **Validation-Selected Endpoint (`best.pt` at Epoch 6 / 3,600 Updates):**
     - Peak Validation Binary AP: `0.307479` (prevalence: `0.113652`).
     - In Fold 3, the validation peak coincided exactly with the fixed-budget milestone at Epoch 6!
     - Size: 412,365,195 bytes | SHA-256: `679f15c545384b90d003d9184d42ab945c7c7adcc67029dfdef7f79c93f25eea`.
   - **Run Endpoint (`last.pt` at Epoch 10 / 6,000 Updates):**
     - Size: 412,366,155 bytes | SHA-256: `804cbfa0d03280fa8b988400b254dbfe79e08c7a6b23449379acdb63490b866e`.
     - Epoch 10 Metrics: Val Binary AP: `0.298220`, Val CTC PER: `22.97%` (0.229745), Train CTC Loss: `0.7665`, Train Binary Loss: `0.3693`.
   - **Endpoint Classification Invariant:** Post-training classifier Phase 6 executed and passed with `COMPLETED_3600_UPDATES`.

4. **Zero Safety Boundary Violations on Fold 3:**
   - Zero active test partition audio or TextGrid files accessed (`ERMS`, `NCC`, `RRBI`, `THV`, `YBAA`, `YDCK` strictly absent).
   - Zero warm-start from Fold 1 or Fold 2 weights; fresh initialization verified from `facebook/wav2vec2-base-960h`.
   - Zero local laptop training backward passes executed.
   - Zero credentials written to disk or logged.

5. **Fold 4 Infrastructure Staged & Ready (Phase D):**
   - Fold 4 configuration validated: `configs/rq1_trn_03a_fold4.yaml`.
   - Fold 4 authorized train/val archive audited: `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip` (783,006,402 bytes, SHA-256 `653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94`).
   - 18 train/val speakers present: `ABA`, `ASI`, `BWC`, `EBVS`, `ERMS`, `HJK`, `HKK`, `HQTV`, `LXC`, `MBMPS`, `NCC`, `PNV`, `RRBI`, `SKA`, `SVBI`, `THV`, `YBAA`, `YDCK`.
   - 6 active test speakers strictly excluded: `NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA` (0 files present; 100% leak-free).
   - 4 deployment packages created with 100% Fold 3 parity: `deploy/kaggle_fold4_staging/`, `deploy/kaggle_fold4_preflight/`, `deploy/kaggle_fold4_smoke/`, `deploy/kaggle_fold4_train/`.
   - 292 automated regression tests passing (1 skipped locally per `AGENTS.md` GPU prohibition).

---

## 2. Remote Training Execution Metrics (Fold 3 Baseline R)

The table below documents the full epoch-by-epoch training progression extracted directly from verified Hugging Face persistent history (`fold3/baseline_r_fold3/history.json`, SHA-256 `33174b40c127...`):

| Epoch | Cumulative Updates | Micro Steps | Train CTC Loss | Train Binary Loss | Val Binary AP | Val CTC PER | Sub Rate | Del Rate | Ins Rate | Skipped AMP | Wall Time (s) | Checkpoint Role |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 600 | 1,200 | 2.8019 | 0.3874 | 0.2839 | 32.34% | 20.66% | 7.87% | 3.81% | 0 | 98.0s | Initial snapshot |
| **2** | 1,200 | 2,400 | 1.0044 | 0.3742 | 0.2871 | 28.72% | 19.07% | 6.86% | 2.79% | 0 | 76.0s | Progression |
| **3** | 1,800 | 3,600 | 0.9157 | 0.3726 | 0.2927 | 27.25% | 18.51% | 6.14% | 2.60% | 0 | 77.2s | Progression |
| **4** | 2,400 | 4,800 | 0.8696 | 0.3728 | 0.3021 | 26.05% | 17.71% | 6.06% | 2.27% | 0 | 76.8s | Progression |
| **5** | 3,000 | 6,000 | 0.8391 | 0.3713 | 0.2979 | 25.24% | 16.94% | 6.14% | 2.16% | 0 | 75.1s | Patience 1 |
| **6** | **3,600** | 7,200 | 0.8182 | 0.3703 | **0.3075** | **24.80%** | 16.76% | 6.05% | 1.99% | 0 | 77.8s | **Fixed Budget Primary & Val Peak (`fixed_budget_checkpoint.pt`, `best.pt`)** |
| **7** | 4,200 | 8,400 | 0.8009 | 0.3710 | 0.2916 | 23.96% | 16.40% | 5.18% | 2.38% | 0 | 76.5s | Patience 1 |
| **8** | 4,800 | 9,600 | 0.7876 | 0.3688 | 0.2893 | 23.88% | 16.00% | 5.89% | 1.99% | 0 | 75.8s | Patience 2 |
| **9** | 5,400 | 10,800 | 0.7756 | 0.3707 | 0.2994 | 22.95% | 15.87% | 4.83% | 2.25% | 0 | 80.1s | Patience 3 |
| **10** | 6,000 | 12,000 | 0.7665 | 0.3693 | 0.2982 | **22.97%** | 15.89% | 5.03% | 2.06% | 0 | 78.7s | Patience 4 $\rightarrow$ Cap Reach (`last.pt`) |

*Total Subprocess Runtime: 1005.9 seconds (~16.8 minutes).*  
*Sum of Epoch Wall Times: 791.9 seconds (~13.2 minutes).*  
*Acoustic Gap / Diagnostic Alignment Note: In accordance with `AGENTS.md`, gold TextGrid timestamps are strictly excluded from model inference inputs. Validation CTC PER and Binary AP constitute the authoritative, unbiased validation evaluation.*

---

## 3. Remote Checkpoint Ledger & Cryptographic Integrity

All artifacts verified live via Hugging Face Hub LFS pointers in private repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`:

| Remote Path | Size (Bytes) | Cryptographic SHA-256 Digest | Role / Invariant Classification |
|---|---|---|---|
| `fold3/baseline_r_fold3/fixed_budget_checkpoint.pt` | 412,371,371 | `bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441` | Primary Endpoint (Epoch 6 / 3,600 updates) |
| `fold3/baseline_r_fold3/best.pt` | 412,365,195 | `679f15c545384b90d003d9184d42ab945c7c7adcc67029dfdef7f79c93f25eea` | Validation Peak (Epoch 6 / 3,600 updates, AP 0.3075) |
| `fold3/baseline_r_fold3/last.pt` | 412,366,155 | `804cbfa0d03280fa8b988400b254dbfe79e08c7a6b23449379acdb63490b866e` | Training Endpoint (Epoch 10 / 6,000 updates) |
| `fold3/baseline_r_fold3/history.json` | 8,014 | `33174b40c12720d5c48c5b4cce220372b48fadc833172d4901c4056fb3171fcc` | Complete 10-epoch metrics log |
| `fold3/baseline_r_fold3/training_endpoint_report.json` | 561 | `97d1ca48d072b08807134e0cbe2f1252df7c79e3ec38a503fc96f87a1c3dd7af` | Endpoint classification report (`COMPLETED_3600_UPDATES`) |
| `fold3/fold3_smoke_1epoch/best.pt` | 412,364,043 | `6b23299f8cab2a6ddeb80f11420fd21bf17febfa47ba1b37ece75a005b8dcde7` | Smoke run validation checkpoint |
| `fold3/fold3_smoke_1epoch/last.pt` | 412,364,043 | `fcddeb198e5203e9e519e216a14b796c6e8786874ec4630a52f3775d4826cb05` | Smoke run endpoint checkpoint |
| `fold3/fold3_smoke_1epoch/history.json` | 805 | `f3ae8468c847b638b49e9756bc84603688fd8934fa5bcb587d4026eb465eadf3` | Smoke 1-epoch metrics log |
| `fold3/fold3_smoke_1epoch/training_endpoint_report.json` | 150 | `9a020c123307f6382e665b23789c564751b2150b72e47c0451f446689f97eb4c` | Smoke report (`INCOMPLETE` fixed budget expected) |

---

## 4. Post-Training Classification & Protocol Verification

The endpoint classification report was read directly from Hugging Face:
```json
{
  "status": "COMPLETED_3600_UPDATES",
  "fold": 3,
  "successful_updates": 3600,
  "epoch": 5,
  "best_val_ap": 0.307479,
  "fixed_budget_sha256": "bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441",
  "best_sha256": "679f15c545384b90d003d9184d42ab945c7c7adcc67029dfdef7f79c93f25eea",
  "last_sha256": "804cbfa0d03280fa8b988400b254dbfe79e08c7a6b23449379acdb63490b866e",
  "history_sha256": "33174b40c12720d5c48c5b4cce220372b48fadc833172d4901c4056fb3171fcc",
  "history_path": "/kaggle/working/checkpoints/fold3/baseline_r_fold3/history.json"
}
```
- **Fold Verification:** `fold == 3` confirms correct fold attribution.
- **Fixed Budget Updating:** `successful_updates == 3600` confirms exact fixed-budget termination criterion.
- **Epoch Indexing:** 0-indexed `epoch == 5` corresponds to completed Epoch 6.
- **SHA-256 Match:** Mirrored remote LFS SHA-256 matches bit-for-bit with trainer report.

---

## 5. Descriptive Cross-Fold Progression Comparison (Folds 1, 2, 3)

The table below provides a side-by-side descriptive summary of Baseline R training across the three completed folds:

| Dimension | Fold 1 Baseline R | Fold 2 Baseline R | Fold 3 Baseline R |
|---|---|---|---|
| **Platform & Hardware** | Colab Tesla T4 (FP16 AMP) | Kaggle Dual Tesla T4 (FP16 AMP) | Kaggle Dual Tesla T4 (FP16 AMP) |
| **Validation Speakers** | `NJS`, `RRBI` | `ASI`, `TNI` | `TXHC`, `ZHAA` |
| **Validation L1 Profile** | Spanish, Hindi | Hindi, Arabic | Mandarin Chinese, Arabic |
| **Val Reference Tokens** | 9,881 | 9,926 | 9,763 |
| **Val Mispronunciation Prevalence** | 0.1032 | 0.0933 | 0.1137 |
| **Total Completed Epochs** | 11 epochs | 7 epochs | 10 epochs (capped) |
| **Total Optimizer Updates** | 6,600 updates | 4,200 updates | 6,000 updates |
| **Termination Reason** | Early stop (patience 5) | Early stop (patience 5) | Execution cap (10 epochs, patience 4) |
| **Peak Epoch (Val AP)** | Epoch 6 (3,600 updates) | Epoch 2 (1,200 updates) | Epoch 6 (3,600 updates) |
| **Peak Validation Binary AP** | **0.317021** | **0.216994** | **0.307479** |
| **Val CTC PER at Peak** | 23.68% | 27.43% | 24.80% |
| **Fixed-Budget Epoch** | Epoch 6 (3,600 updates) | Epoch 6 (3,600 updates) | Epoch 6 (3,600 updates) |
| **Fixed-Budget Val Binary AP** | **0.317021** | **0.201083** | **0.307479** |
| **Fixed-Budget Val CTC PER** | **23.68%** | **23.53%** | **24.80%** |
| **Endpoint Val CTC PER (`last.pt`)** | 23.47% (Epoch 11) | 23.30% (Epoch 7) | 22.97% (Epoch 10) |
| **Skipped AMP Updates** | 0 | 0 | 0 |
| **Subprocess Runtime** | ~18.5 min | 761.9s (~12.7 min) | 1005.9s (~16.8 min) |

### 5.1 Scientific Caveats & Descriptive Observations
1. **No Cross-Fold Statistical Superiority Inferences:**
   In cross-validation, each fold evaluates on an entirely disjoint pair of validation speakers with distinct L1 phonological transfer profiles and varying baseline mispronunciation prevalence (0.0933 in Fold 2 vs 0.1137 in Fold 3). Therefore, a higher validation AP on Fold 3 (0.3075) compared to Fold 2 (0.2011) **must not be interpreted as model improvement or statistical superiority**. It reflects differences in evaluation cohort acoustic-phonetic characteristics and label distributions.
2. **Remarkable Acoustic Consistency (CTC PER):**
   At the primary fixed-budget milestone (3,600 updates), CTC phoneme error rate exhibits striking stability across all three distinct validation cohorts: **23.68%** (Fold 1), **23.53%** (Fold 2), and **24.80%** (Fold 3). This narrow spread (1.27 pp) confirms the consistent representation capability of the frozen Wav2Vec2 acoustic backbone.
3. **Fixed-Budget vs Validation Peak Alignment:**
   In Fold 1 and Fold 3, peak validation AP was reached at Epoch 6 (3,600 updates), perfectly aligning `best.pt` with `fixed_budget_checkpoint.pt`. In Fold 2, peak validation AP occurred earlier (Epoch 2 / 1,200 updates), underscoring the critical necessity of dual checkpoint accounting to avoid confounding training exposure with model selection.
4. **Complete Numerical Stability:**
   Across all three folds (totaling 16,800 optimizer updates), exactly **0 skipped AMP updates** occurred, confirming that `grad_scaler_initial_scale: 4096.0` with gradient clipping at `1.0` maintains 100% numerical stability on Tesla T4 FP16 AMP.

---

## 6. Fold 4 Staging Preparation & Specifications

| Dimension | Dataset Staging (`staging/`) | Preflight Package (`preflight/`) | Smoke Package (`smoke/`) | Official Training Package (`train/`) |
|---|---|---|---|---|
| **Package Directory** | `deploy/kaggle_fold4_staging/` | `deploy/kaggle_fold4_preflight/` | `deploy/kaggle_fold4_smoke/` | `deploy/kaggle_fold4_train/` |
| **Kaggle Slug** | `minkhoaa/langfens-rq1-fold4-staging` | `minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2-preflight` | `minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2-smoke` | `minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2` |
| **Title** | `LANGFENS RQ1 Fold 4 Train/Val Staging Package` | `LANGFENS RQ1 Fold 4 Baseline R T4x2 (Preflight)` | `LANGFENS RQ1 Fold 4 Baseline R T4x2 (Smoke)` | `LANGFENS RQ1 Fold 4 Baseline R T4x2` |
| **Code File** | N/A (Dataset metadata) | `kaggle_preflight.ipynb` | `kaggle_smoke.ipynb` | `kaggle_train.ipynb` |
| **Pinned Commit** | N/A | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` |
| **Execution Mode** | N/A | Preflight only (`--mode verify_fold4`) | Preflight gate + 1-epoch smoke (`train_fold4`) | Preflight gate + 10-epoch Baseline R (`train_fold4`) |
| **Execution Cap** | N/A | 0 epochs (0 updates) | 1 completed epoch (~600 updates) | 10 completed epochs (max 6,000 updates; fixed budget at Epoch 6 / 3,600 updates) |
| **Run ID** | N/A | N/A | `fold4_smoke_1epoch` | `baseline_r_fold4` |
| **Local Checkpoint Path** | N/A | None | `/kaggle/working/checkpoints/fold4/fold4_smoke_1epoch/` | `/kaggle/working/checkpoints/fold4/baseline_r_fold4/` |
| **Remote Mirror Prefix** | N/A | Probe verification only (`_probe_write_test`) | `fold4/fold4_smoke_1epoch/` | `fold4/baseline_r_fold4/` |
| **Machine Target** | N/A | `NvidiaTeslaT4` (Dual T4 FP16 AMP) | `NvidiaTeslaT4` (Dual T4 FP16 AMP) | `NvidiaTeslaT4` (Dual T4 FP16 AMP) |
| **Dataset Source** | N/A | `minkhoaa/langfens-rq1-fold4-staging` | `minkhoaa/langfens-rq1-fold4-staging` | `minkhoaa/langfens-rq1-fold4-staging` |

### 6.1 Fold 4 Authorized Package & Test Isolation Audit
- **Archive Path:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`
- **File Size:** 783,006,402 bytes
- **SHA-256 Digest:** `653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94`
- **Train Speakers (16):** `ABA`, `ASI`, `BWC`, `EBVS`, `ERMS`, `HJK`, `HQTV`, `LXC`, `MBMPS`, `PNV`, `RRBI`, `SKA`, `SVBI`, `THV`, `YBAA`, `YDCK`
- **Val Speakers (2):** `HKK`, `NCC`
- **Test Partition Speakers Strictly Excluded (6):** `NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA` (0 files present; zero active test leakage proved).

---

## 7. Verification Evidence & Test Ledger

- **Suite Execution:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/`
- **Result:** **292 passed, 1 skipped, 1 warning in 178.43s** (100% green).
- **Key Regression Tests Verified:**
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold4_three_entrypoints_isolation`: PASS. Proves Fold 4 preflight, smoke, and official train packages possess mutually exclusive kernel IDs and matching titles without collision.
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold4_staging_and_speaker_isolation`: PASS. Proves Fold 4 staging metadata and authorized archive (`authorized_RQ1-TRN_fold4_train_val_only.zip`) match SHA-256 `653bdad079ee...` and strictly exclude all 6 active test speakers (`NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA`).
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold4_training_invariants`: PASS. Proves Fold 4 training package enforces preflight-first gate, explicit run ID `baseline_r_fold4`, local output path `/kaggle/working/checkpoints/fold4/baseline_r_fold4`, and Phase 6 post-training classification.

---

## 8. Operator Deployment Protocol (Exact Commands for Fold 4)

When operator authorization is granted, execute the deployment in four stages following the permanent Kaggle workflow:

### Stage 1: Stage and Create Private Fold 4 Dataset on Kaggle
1. Prepare staging archive on laptop disk (copy binary):
   ```bash
   cp datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip deploy/kaggle_fold4_staging/authorized_RQ1-TRN_fold4_train_val_only.bin
   ```
2. Upload private dataset via Kaggle CLI:
   ```bash
   kaggle datasets create -p deploy/kaggle_fold4_staging/
   ```
3. Remove local temporary binary copy:
   ```bash
   rm deploy/kaggle_fold4_staging/authorized_RQ1-TRN_fold4_train_val_only.bin
   ```

### Stage 2: Push Preflight Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold4_preflight/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2-preflight`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to verify preflight staging and Hugging Face probe write.

### Stage 3: Push Smoke Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold4_smoke/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2-smoke`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to run 1-epoch smoke test and verify Hugging Face persistence.

### Stage 4: Push Official Training Kernel via CLI
```bash
kaggle kernels push -p deploy/kaggle_fold4_train/ --no-run
```
In Kaggle Web UI (`https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-4-baseline-r-t4x2`):
- Attach `HF_TOKEN` under **Add-ons -> Secrets**.
- Click **"Save & Run All"** to execute official 10-epoch Baseline R training on Fold 4.
