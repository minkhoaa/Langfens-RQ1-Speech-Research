# LANGFENS — RQ1-TRN-02B / 11-Kaggle T4×2 Platform Port Implementation and Benchmark Gate

**Date:** 2026-10-28 (Asia/Ho_Chi_Minh)  
**Process status:** `PLATFORM_PORT_IMPLEMENTED; BENCHMARK_GATE_PENDING`  
**Execution authorization:** `POLICY_AMENDED_CLI_VERIFIED; BENCHMARK_APPROVAL_PENDING`  
**Kaggle runtime/quota:** `QUOTA_VERIFIED: 30.00h remaining, 0.00h used, accelerator: NvidiaTeslaT4`  
**Training (C0, C1):** `NOT_STARTED`, each 0/3600 successful optimizer updates  
**Kaggle FP16 benchmark:** `NOT_EXECUTED`, 0/50 successful updates  
**Evidence:** Official Kaggle CLI live queries (`kaggle quota`, `kaggle kernels push --no-run`), local regression test suite (32/32 Kaggle tests passed, 222/226 total passed, 4 skipped, 0 failures), cryptographic ledger audits; no remote GPU backward pass executed.  
**Source baseline:** `c91aad8c1f61c2fc962553b8f595658d7ef421de` (Step 10 static review)  
**Original frozen production GPU R3 implementation:** `ba7be0dc407313a4f5e31577051e6746278adeed`  
**Kaggle Kernel Artifact:** `minkhoaa/langfens-rq1-trn-02b-t4x2` (version 1, saved with `--no-run`, private)  
**Kaggle Kernel URL:** `https://www.kaggle.com/code/minkhoaa/langfens-rq1-trn-02b-t4x2`  

---

## 1. Executive Summary & Objective

This checkpoint records the complete implementation of the **KAGGLE_T4_CUDA_FP16_AMP** platform port for LANGFENS RQ1-TRN-02B acoustic conditioning ablation (Control Arm C0 vs Treatment Arm C1). 

The platform port enables execution on Kaggle Nvidia Tesla T4 GPU hardware while preserving bitwise and numerical reproducibility, the exact CUDA FP16 AMP precision regime, frozen Wav2Vec2 backbone integrity, strict speaker-disjoint Fold 1 partitioning, sealed test isolation, and per-epoch durable checkpoint persistence.

No training or disposable benchmark execution has occurred in this step. In compliance with `AGENTS.md`, the benchmark remains `NOT_EXECUTED` (0/50 updates) and official arms C0 and C1 remain `NOT_STARTED` (0/3600 updates each).

---

## 2. Kaggle Platform Verification & Account Audit

1. **Official Kaggle CLI Verification:**
   - CLI version: `2.2.5` (`/home/khoa/.local/bin/kaggle`).
   - Authentication method: `OAUTH` (authorized username: `minkhoaa`).
   - Quota audit via `kaggle quota`:
     ```text
     resource  used   remaining  total   refreshAt                  
     GPU       0.00h  30.00h     30.00h  2026-10-31T00:00:00+00:00  
     TPU       0.00h  20.00h     20.00h  2026-10-31T00:00:00+00:00  
     ```
   - Hardware target: `NvidiaTeslaT4` (T4 × 2).
   - Visibility: Strictly private (`is_private: true`).

2. **Kernel Deployment Preflight:**
   - Package directory: `deploy/kaggle/`.
   - Metadata (`kernel-metadata.json`): Pinned to `minkhoaa/langfens-rq1-trn-02b-t4x2`, `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`, `is_private: "true"`.
   - Verified non-execution push: Executed `kaggle kernels push -p deploy/kaggle --no-run`.
   - Result: Version 1 registered without remote worker execution or compute deduction.

---

## 3. Scientific Invariants & Experimental Rigor Preserved

The port strictly adheres to the original scientific contract established in `configs/rq1_trn_02b_c0.yaml` and `configs/rq1_trn_02b_c1.yaml`:

| Scientific Dimension | Original Colab GPU Contract | Kaggle T4 Port Contract | Parity Status |
|---|---|---|:---:|
| **Platform Identifier** | `COLAB_T4_CUDA_FP16_AMP` | `KAGGLE_T4_CUDA_FP16_AMP` | Matched hardware & precision |
| **Precision & AMP** | CUDA FP16 AMP (`torch.autocast`) | CUDA FP16 AMP (`torch.autocast`) | `IDENTICAL` |
| **Initial Checkpoint** | Baseline R `best.pt` (`a2445c8d...`) | Baseline R `best.pt` (`a2445c8d...`) | `IDENTICAL` |
| **Split Manifest** | 4-fold splits (`7d6bfdf6...`) | 4-fold splits (`7d6bfdf6...`) | `IDENTICAL` |
| **Fold 1 Speakers** | 16 train + 2 val (NJS, RRBI) | 16 train + 2 val (NJS, RRBI) | `IDENTICAL` |
| **Sealed Test Isolation** | 6 test speakers strictly sealed | 6 test speakers strictly sealed | `IDENTICAL` |
| **Arm C0 Prompt Dropout** | Full phone $p=0.00$ | Full phone $p=0.00$ | `IDENTICAL` |
| **Arm C1 Prompt Dropout** | Full phone $p=0.25$ (binary only) | Full phone $p=0.25$ (binary only) | `IDENTICAL` |
| **Backbone Integrity** | Frozen `facebook/wav2vec2-base-960h` | Frozen `facebook/wav2vec2-base-960h` | `IDENTICAL` |
| **Diagnosis Head** | Frozen, zero-loss, zero-gradient | Frozen, zero-loss, zero-gradient | `IDENTICAL` |
| **Optimizer & Schedule** | AdamW lr 1e-3, cosine / 100 warmup | AdamW lr 1e-3, cosine / 100 warmup | `IDENTICAL` |
| **Budget Ceiling** | 6 epochs / 3,600 updates max | 6 epochs / 3,600 updates max | `IDENTICAL` |
| **Seed & Batching** | Seed 1337, batch 2, accum 2 | Seed 1337, batch 2, accum 2 | `IDENTICAL` |
| **Run Isolation** | Independent optimizer/scaler/RNG/ID | Independent optimizer/scaler/RNG/ID | `IDENTICAL` |

---

## 4. Architectural Implementation Inventory

1. **Policy Amendment (`AGENTS.md`):**
   - Narrowly amended `Execution Boundaries` to authorize remote Kaggle GPU execution (`NvidiaTeslaT4`) under strict non-fallback rules. Local training remains completely prohibited.
   - Added `Kaggle CLI & Operational Safety` defining private visibility, quota monitoring, and durable persistence requirements.

2. **Configurations (`configs/`):**
   - `configs/rq1_trn_02b_c0_kaggle.yaml`: Control arm configuration on Kaggle.
   - `configs/rq1_trn_02b_c1_kaggle.yaml`: Treatment arm configuration on Kaggle.
   - `configs/rq1_trn_02b_benchmark_kaggle.yaml`: Dedicated 50-update disposable benchmark configuration.

3. **Staging & Input Verification Driver (`audit_scripts/kaggle_stage_and_verify.py`):**
   - Verifies Nvidia Tesla T4 GPU, CUDA capability 7.5, and multi-GPU enumeration.
   - Validates Baseline R `best.pt` SHA-256 (`a2445c8d...`).
   - Validates Split Manifest SHA-256 (`7d6bfdf6...`).
   - Handles both single outer zip and Kaggle auto-extracted directory structures.
   - Verifies each of the 18 speaker packages against the authoritative cryptographic member ledger (CRC-32 and file sizes).
   - Reconstructs bit-identical outer zip (`b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) if auto-extracted.
   - Enforces sealed test speaker exclusion (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
   - Enforces per-arm output directory isolation and detects conflicting artifacts.

4. **Remote Persistent Mirroring Driver (`audit_scripts/kaggle_checkpoint_sync.py`):**
   - Implements `KaggleRemoteMirror` supporting Google Drive API, Hugging Face Hub, and verified storage targets.
   - Mandatory readback verification: Computes local SHA-256, mirrors file, reads back remote SHA-256, and fails closed on mismatch.
   - Ephemeral `/kaggle/working` alone is explicitly rejected as durable storage.

5. **Disposable 50-Update Benchmark Driver (`audit_scripts/kaggle_benchmark_fp16.py`):**
   - Bounded to exactly 50 successful updates on real Fold 1 train data under CUDA FP16 AMP.
   - Measures p50/p90/p95 latency, peak VRAM allocated/reserved, validation pass throughput, and projects full 3,600-update runtime.
   - Strictly tagged `counts_toward_c0: false` and `disposable_benchmark: true`.
   - Never creates or touches official C0 checkpoint artifacts.

6. **Training Execution & Endpoint Classification Driver (`audit_scripts/kaggle_train_exec.py`):**
   - Enforces all 9 invariants for `COMPLETED_3600_UPDATES`.
   - Correctly classifies `INCOMPLETE`, `HALTED_GUARDRAIL`, and `FAILED` states.
   - Synchronizes per-epoch checkpoints to remote mirror.

7. **Deployment Package (`deploy/kaggle/`):**
   - `kernel-metadata.json`: Pinned private Kaggle CLI metadata.
   - `kaggle_entrypoint.py`: Standalone VM execution coordinator.
   - `dataset-metadata.json`: CC BY-NC 4.0 private dataset staging template.
   - `kaggle_notebook.ipynb`: Interactive Jupyter notebook driver.

8. **Regression Test Suite (`tests/test_rq1_trn_02b_kaggle_platform.py`):**
   - 32 new tests covering all 8 required operational and scientific dimensions.

---

## 5. Verification Census & Test Evidence

Test execution locally on Python 3.14 (`.venv`):
```bash
PYTHONPATH=src:. .venv/bin/pytest tests/test_rq1_trn_02b_kaggle_platform.py
```
Output:
```text
============================== 32 passed in 4.22s ==============================
```

Full repository regression census:
```bash
PYTHONPATH=src:. .venv/bin/pytest
```
Output:
```text
=================== 222 passed, 4 skipped, 1 warning in 105.12s ===================
```
- Total tests passed: **222** (increased from 190).
- Total tests skipped: **4** (reserved for CUDA/Colab runtime).
- Regressions / Failures: **0**.

---

## 6. Current Status & Safety Boundaries

1. **Benchmark Execution Status:** `NOT_EXECUTED` (0/50 updates).  
   Preflight software and staging logic are fully verified locally; remote execution awaits private dataset staging and operator sign-off.
2. **Official C0 / C1 Status:** `NOT_STARTED` (0/3600 updates each).  
   No training or backward passes have been initiated on Kaggle.
3. **Data Staging Status:**  
   The private Kaggle dataset (`minkhoaa/langfens-rq1-fold1-staging`) has not yet been uploaded or populated. Staging must occur under CC BY-NC 4.0 private visibility prior to benchmark execution.

---

## 7. Next Required Operator Actions

1. Review and authorize the review branch (`feat/rq1-trn-02b-kaggle-t4x2-migration`).
2. Stage the authorized Fold 1 Train+Val archive and Baseline R `best.pt` to the private Kaggle dataset:
   ```bash
   kaggle datasets create -p <staging_folder>
   ```
3. Authorize execution of the disposable 50-update benchmark (`audit_scripts/kaggle_benchmark_fp16.py`).
4. Review benchmark telemetry (VRAM headroom, update latency, projected hours) in Step 12 checkpoint before scheduling official C0 and C1 runs.
