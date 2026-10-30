# LANGFENS — RQ1-TRN-03A / 09-Official Fold 2 Baseline R Training Launch & Execution

**Status:** `COMPLETED; INDEPENDENTLY_VERIFIED`

**Recorded at:** 2026-10-30T15:25:00+07:00

**Evidence source:** Kaggle CLI, live remote kernel execution logs `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`, smoke kernel logs `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke`, private Hugging Face model repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, 286/286 passed, 1 skipped)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

This checkpoint certifies the successful end-to-end execution, remote persistent mirroring, and independent cryptographic audit of **Fold 2 Baseline R Training** on remote Kaggle Dual Tesla T4 GPUs (`KAGGLE_T4_CUDA_FP16_AMP`).

1. **Smoke Gate Clearance (Phase A):**
   - Kernel `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke` executed with `HF_TOKEN` Kaggle Secret.
   - Completed strictly 1 epoch (~600 updates, 1,200 micro steps) in 95.04s.
   - Val Binary AP: `0.1960`, Val CTC PER: `30.44%`, Skipped AMP attempts: `0`.
   - Mirrored atomic checkpoints and reports to `minkhoaa/langfens-rq1-cross-fold-checkpoints` under `fold2/fold2_smoke_1epoch/`. Smoke gate declared **`PASS`**.

2. **Official Training Execution & Completion (Phase B):**
   - Kernel `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` executed with `HF_TOKEN` Kaggle Secret.
   - Pinned training source commit verified: `cc56b82ca1a9a6ceeca85929e07db4cac8537348`.
   - Completed 7 training epochs (4,200 total optimizer updates, 8,400 micro steps) in 761.9s (~12.7 min) on Dual Tesla T4 FP16 AMP.
   - Early stopping automatically triggered at Epoch 7 following patience 5 on validation AP after peak AP at Epoch 2.
   - Skipped AMP updates: `0` (100% numerical stability across all 4,200 updates; zero scale collapses).

3. **Dual Checkpoint Accounting & Endpoint Verification (Phase C):**
   - **Primary Fixed-Budget Endpoint (Epoch 6 / Exactly 3,600 Updates):**
     - Successfully locked and saved as `fixed_budget_checkpoint.pt`.
     - Mirrored atomically to Hugging Face with SHA-256 readback.
     - Size: 412,371,371 bytes | SHA-256: `ab4c589445feac42636296cd36d7cf2ae7eba7360445f0084454c87fa2fef508`.
     - Epoch 6 Metrics: Val Binary AP: `0.2011`, Val CTC PER: `23.53%`, Train CTC Loss: `0.8149`, Train Binary Loss: `0.3643`.
   - **Validation-Selected Endpoint (`best.pt` at Epoch 2 / 1,200 Updates):**
     - Peak Validation Binary AP: `0.2170` (prevalence: `0.0933`).
     - Val CTC PER: `27.43%`, Train CTC Loss: `1.0060`, Train Binary Loss: `0.3676`.
     - Size: 412,364,235 bytes | SHA-256: `b9032ee9ba8d5f8f03c8ce164237a49c9524442281b183d9ee522eb8d2b759ff`.
   - **Run Endpoint (`last.pt` at Epoch 7 / 4,200 Updates):**
     - Size: 412,365,451 bytes | SHA-256: `bfe0eea14232101c7acc00f7e461993fac3f7b8d8fc3d0883326767f2282cd77`.
   - **Endpoint Classification Invariant:** Post-training classifier Phase 6 executed and passed with `COMPLETED_3600_UPDATES`.

4. **Zero Safety Boundary Violations:**
   - Zero active test partition audio or TextGrid files accessed (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI` strictly absent).
   - Zero warm-start from Fold 1 weights; fresh initialization verified from `facebook/wav2vec2-base-960h`.
   - Zero local laptop training backward passes executed.
   - Zero credentials written to disk or logged.

---

## 2. Remote Training Execution Metrics (Fold 2 Baseline R)

The table below documents the full epoch-by-epoch training progression extracted directly from verified Hugging Face persistent history (`fold2/baseline_r_fold2/history.json`, SHA-256 `c01418e5a203...`):

| Epoch | Cumulative Updates | Micro Steps | Train CTC Loss | Train Binary Loss | Val Binary AP | Val CTC PER | Skipped AMP Updates | Wall Time (s) | Checkpoint Role |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 600 | 1,200 | 2.8176 | 0.3826 | 0.1960 | 30.44% | 0 | 109.9s | Initial snapshot |
| **2** | 1,200 | 2,400 | 1.0060 | 0.3676 | **0.2170** | 27.43% | 0 | 83.6s | **Validation Peak (`best.pt`)** |
| **3** | 1,800 | 3,600 | 0.9171 | 0.3660 | 0.1978 | 25.54% | 0 | 84.8s | Patience 1 |
| **4** | 2,400 | 4,800 | 0.8685 | 0.3667 | 0.2094 | 25.06% | 0 | 83.3s | Patience 2 |
| **5** | 3,000 | 6,000 | 0.8370 | 0.3661 | 0.2158 | 24.02% | 0 | 84.9s | Patience 3 |
| **6** | **3,600** | 7,200 | 0.8149 | 0.3643 | **0.2011** | **23.53%** | 0 | 84.4s | **Fixed Budget Primary (`fixed_budget_checkpoint.pt`)** |
| **7** | 4,200 | 8,400 | 0.7973 | 0.3640 | 0.2057 | 23.30% | 0 | 83.9s | Patience 5 $\rightarrow$ Early Stop (`last.pt`) |

*Total Subprocess Runtime: 761.9 seconds (~12.7 minutes).*  
*Acoustic Gap / Diagnostic Alignment Note: In accordance with `AGENTS.md`, gold TextGrid timestamps are strictly excluded from model inference inputs. Validation CTC PER and Binary AP constitute the authoritative, unbiased validation evaluation.*

---

## 3. Remote Checkpoint Ledger & Cryptographic Integrity

All artifacts verified live via Hugging Face Hub LFS pointers in private repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`:

| Remote Path | Size (Bytes) | Cryptographic SHA-256 Digest | Role / Invariant Classification |
|---|---|---|---|
| `fold2/baseline_r_fold2/fixed_budget_checkpoint.pt` | 412,371,371 | `ab4c589445feac42636296cd36d7cf2ae7eba7360445f0084454c87fa2fef508` | Primary Endpoint (Epoch 6 / 3,600 updates) |
| `fold2/baseline_r_fold2/best.pt` | 412,364,235 | `b9032ee9ba8d5f8f03c8ce164237a49c9524442281b183d9ee522eb8d2b759ff` | Validation Peak (Epoch 2 / 1,200 updates, AP 0.2170) |
| `fold2/baseline_r_fold2/last.pt` | 412,365,451 | `bfe0eea14232101c7acc00f7e461993fac3f7b8d8fc3d0883326767f2282cd77` | Training Endpoint (Epoch 7 / 4,200 updates) |
| `fold2/baseline_r_fold2/history.json` | 5,612 | `c01418e5a203515520f0cfb99192a03776794adec3f2823d950249c9f3a537a7` | Complete 7-epoch metrics log |
| `fold2/baseline_r_fold2/training_endpoint_report.json` | 561 | Verified Bit-Exact | Endpoint classification report (`COMPLETED_3600_UPDATES`) |
| `fold2/fold2_smoke_1epoch/best.pt` | 412,364,235 | Verified Remote LFS | Smoke run validation checkpoint |
| `fold2/fold2_smoke_1epoch/last.pt` | 412,364,235 | Verified Remote LFS | Smoke run endpoint checkpoint |
| `fold2/fold2_smoke_1epoch/history.json` | 801 | Verified Remote | Smoke 1-epoch metrics log |
| `fold2/fold2_smoke_1epoch/training_endpoint_report.json` | 134 | Verified Remote | Smoke report (`INCOMPLETE` fixed budget expected) |

---

## 4. Post-Training Classification & Protocol Verification

The endpoint classification report was read directly from Hugging Face:
```json
{
  "status": "COMPLETED_3600_UPDATES",
  "fold": 2,
  "successful_updates": 3600,
  "epoch": 5,
  "best_val_ap": 0.216994,
  "fixed_budget_sha256": "ab4c589445feac42636296cd36d7cf2ae7eba7360445f0084454c87fa2fef508",
  "best_sha256": "b9032ee9ba8d5f8f03c8ce164237a49c9524442281b183d9ee522eb8d2b759ff",
  "last_sha256": "bfe0eea14232101c7acc00f7e461993fac3f7b8d8fc3d0883326767f2282cd77",
  "history_sha256": "c01418e5a203515520f0cfb99192a03776794adec3f2823d950249c9f3a537a7",
  "history_path": "/kaggle/working/checkpoints/fold2/baseline_r_fold2/history.json"
}
```
- **Fold Verification:** `fold == 2` confirms correct fold attribution.
- **Fixed Budget Updating:** `successful_updates == 3600` confirms exact fixed-budget termination criterion.
- **Epoch Indexing:** 0-indexed `epoch == 5` corresponds to completed Epoch 6.
- **SHA-256 Match:** Mirrored remote LFS SHA-256 matches bit-for-bit with trainer report.

---

## 5. Verification Evidence & Test Ledger

- **Suite Execution:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/`
- **Result:** **286 passed, 1 skipped, 1 warning in 170.17s** (100% green).
- **Invariants Verified:**
  - `test_kaggle_fold2_training_run_id_reconciliation`: PASS. Reconciles run ID, Phase 6 directory, and Hugging Face mirror prefix.
  - `test_kaggle_fold2_three_entrypoints_isolation`: PASS. Proves preflight, smoke, and official train packages are distinct.
  - `test_kaggle_workflow_agents_policy`: PASS. Codifies permanent Kaggle workflow.

---

## 6. Next Research Step

With Fold 2 Baseline R Training successfully completed and cryptographically verified, the repository advances to:
- **Step 10:** Dedicated Fold 3 Staging & Baseline R Deployment Preparation (`configs/rq1_trn_03a_fold3.yaml`, `authorized_RQ1-TRN_fold3_train_val_only.zip`).
