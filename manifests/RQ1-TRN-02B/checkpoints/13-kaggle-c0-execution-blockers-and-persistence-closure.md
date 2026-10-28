# LANGFENS — RQ1-TRN-02B / 13-Kaggle C0 Execution Attempt, Persistence Closure, and Accelerator Blocker

**Recorded:** 2026-10-29T00:30:00+07:00 (Asia/Ho_Chi_Minh)
**Source examined / pinned:** `a701f319c75c39c007f683ed34f0bd86b00f0495` on `review/rq1-trn-02b-kaggle-official-runs` (remote branch HEAD verified)
**Process state:** `REMEDIATION_VERIFIED; DATASET_STAGED; REMOTE_PERSISTENCE_VERIFIED; EXECUTION_BLOCKED_KAGGLE_GPU_PROVISIONING`
**Benchmark decision:** `WAIVED_BY_OPERATOR_FOR_THIS_EXECUTION_PLAN` (user explicitly waived disposable 50-update benchmark; proceeding toward official C0 → C1).
**Official C0:** `0/3600` successful optimizer updates (`EXECUTION_HALTED_BEFORE_OPTIMIZER_STEP`).
**Official C1:** `0/3600` successful optimizer updates (`NOT_STARTED`; pending completed C0).
**Kaggle kernel ID:** `minkhoaa/langfens-rq1-trn-02b-c0-gpu` (Version 1, Version 3).
**Kaggle dataset ID:** `minkhoaa/langfens-rq1-fold1-staging` (Version 2, status: `ready`).
**Remote mirror target:** `minkhoaa/langfens-rq1-trn-02b-checkpoints` (Private Hugging Face repository, authenticated write-readback probe verified).
**TEST partition:** Remained strictly sealed; zero test audio or annotations accessed, staged, or inspected.

---

## 1. Executive Summary & Operator Waiver Acceptance

In accordance with operator instructions, the disposable 50-successful-update Kaggle benchmark was waived. This waiver does **not** waive software correctness, data integrity, persistent checkpoint durability, account quota, device verification, scientific protocol, or per-epoch validation guardrails.

All eight critical defects identified in Step 12 were resolved and verified through 37 platform regression tests (227 total tests passed across the full test suite). The private Kaggle staging dataset was populated and verified, and an authenticated remote mirror to a private Hugging Face repository was established with live write-readback cryptographic SHA-256 verification.

Upon launching official C0 via the official Kaggle CLI (`kaggle kernels push -p deploy/kaggle --accelerator NvidiaTeslaT4`), the Kaggle execution runtime booted without a GPU (`nvidia-smi` not found, `torch.cuda.is_available() == False`). Our strict fail-closed safety guardrail immediately tripped as designed at Phase 1, halting execution with `RuntimeError: FAIL-CLOSED: CUDA GPU is not available on this runtime!`.

Strictly zero unaccelerated CPU backward passes or optimizer updates were executed.

---

## 2. Technical Remediation of Step 12 Defects

### 2.1 Genuine Private Remote Mirroring (`audit_scripts/kaggle_checkpoint_sync.py`)
- Replaced simulated `SYNCED` stubs with genuine private remote mirroring to Hugging Face (`minkhoaa/langfens-rq1-trn-02b-checkpoints`).
- Implemented `verify_readiness()` with live authenticated probe: writes `.probe_mirror_<timestamp>.tmp`, reads it back via `hf_hub_download()`, computes cryptographic SHA-256, verifies exact bit-identical match, and deletes the probe.
- Implemented `mirror_file()` with genuine upload via `HfApi.upload_file()` and mandatory independent readback SHA-256 verification via `hf_hub_download()`. Raises `IOError` on any checksum mismatch. Never reports simulated `SYNCED`.
- Enforced rejection of ephemeral storage: `KaggleRemoteMirror` strictly rejects `/kaggle/working` or any path under `/kaggle` as a durable remote mirror target (`allow_local_test` permitted only in isolated unit tests outside Kaggle).

### 2.2 Per-Epoch Atomic Remote Synchronization (`audit_scripts/train_rq1_trn_02b.py`)
- Integrated `remote_mirror` parameter and `--remote-mirror-backend`, `--remote-mirror-target`, and `--remote-mirror-token` CLI arguments directly into `train_rq1_trn_02b.py`.
- Synchronizes checkpoints atomically during the training loop at **every epoch**:
  - `outdir / "last.pt"`
  - `outdir / "best.pt"` (when validation AP improves)
  - `outdir / "fixed_budget_checkpoint.pt"` (when 3,600 updates are reached)
  - `outdir / "halted_guardrail.pt"` (if CTC PER guardrail trips)
  - `outdir / "history.json"` (upon completion or termination)
- If remote upload or SHA readback fails at any epoch, execution halts immediately (fail closed).

### 2.3 Corrected Trainer Arguments (`audit_scripts/kaggle_train_exec.py`)
- Eliminated invalid CLI arguments `--arm` and `--output-dir` from the subprocess command constructed for `train_rq1_trn_02b.py`.
- Correctly passed staged baseline weights via `--initial-checkpoint <path>` and staged archive via `--zip-path <path>`.
- Corrected `--resume` to pass the explicit path to `last.pt` rather than a bare boolean flag.

### 2.4 Uncompressed Binary Staging Archive (`audit_scripts/kaggle_stage_and_verify.py`)
- Identified that Kaggle's backend automatically decompresses any file with a `.zip` extension upon dataset ingestion.
- Staged the authorized Fold 1 Train+Val archive as `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` (761,922,918 bytes) alongside `best.pt` (412,366,029 bytes).
- Extended `verify_and_stage_fold1_data()` to detect raw `.bin` archives, verify the exact cryptographic SHA-256 (`b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`), symlink/copy to `authorized_RQ1-TRN-02A_fold1_train_val_only.zip`, and verify all 18 speaker packages with zero test leakage.

### 2.5 Pinned Commit SHA Verification (`deploy/kaggle/kaggle_entrypoint.py`, `kaggle_notebook.ipynb`)
- Replaced default benchmark mode with official `train_c0`.
- Enforced that the execution kernel clones the repository, checks out the exact immutable commit SHA, verifies `git rev-parse HEAD == target_commit`, and verifies that `git status --porcelain` is clean before launching training.

### 2.6 Removal of `--allow-mock-gpu` from Production CLIs
- Removed `--allow-mock-gpu` from `audit_scripts/kaggle_stage_and_verify.py` and `audit_scripts/kaggle_train_exec.py`.
- Filtered `parse_known_args()` to reject unknown CLI flags with exit code 2 while preserving Jupyter ipykernel connection arguments.

---

## 3. Dataset Staging & Remote Mirroring Verification Ledger

### 3.1 Kaggle Private Dataset (`minkhoaa/langfens-rq1-fold1-staging`)
- **Status:** `ready` (Version 2)
- **Visibility:** `is_private: true`
- **License:** `CC-BY-NC-4.0`
- **Files Verified:**
  | File Name | File Size | Cryptographic SHA-256 Digest | Status |
  |---|:---:|---|:---:|
  | `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` | 761,922,918 B | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **MATCH** |
  | `best.pt` | 412,366,029 B | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **MATCH** |

### 3.2 Private Remote Checkpoint Mirror (`minkhoaa/langfens-rq1-trn-02b-checkpoints`)
- **Backend:** `huggingface`
- **Visibility:** Private model repository
- **Live Readiness Probe:** Verified via `KaggleRemoteMirror(backend="huggingface")`. Successfully wrote probe file, downloaded via `hf_hub_download`, verified SHA-256 digest match, and deleted probe.

---

## 4. Execution Attempt & Live Runtime Findings

### 4.1 Kaggle Execution Attempt
- **Command:** `kaggle kernels push -p deploy/kaggle --accelerator NvidiaTeslaT4`
- **Kernel Ref:** `minkhoaa/langfens-rq1-trn-02b-c0-gpu` (Version 1, Version 3)
- **Metadata Configuration:**
  - `kernel_type`: `notebook`
  - `is_private`: `true`
  - `enable_gpu`: `true`
  - `machine_shape`: `NvidiaTeslaT4`
  - `dataset_sources`: `["minkhoaa/langfens-rq1-fold1-staging"]`

### 4.2 Observed Runtime Log Output
```text
/bin/bash: line 1: nvidia-smi: command not found
[GPU] CUDA Available: False
RuntimeError: FAIL-CLOSED: CUDA GPU is not available on this runtime!
```

### 4.3 Root Cause Analysis
Although the Kaggle account has `30.00h` weekly GPU quota (`kaggle quota` verified), Kaggle requires **SMS Phone Verification** in user account settings (`https://www.kaggle.com/settings`) before allocating physical GPU hardware to notebook or script containers. When an unverified account submits a kernel requesting GPU, Kaggle's worker allocator silently provisions an unaccelerated CPU container.

Our Phase 1 hardware guard (`if not torch.cuda.is_available(): raise RuntimeError(...)`) detected the absence of GPU hardware immediately and halted execution within 15 seconds.

### 4.4 Invariants Preserved
- **Optimizer updates:** `0 / 3,600`.
- **Backward passes:** `0`.
- **CPU fallback:** Strictly prohibited and prevented.
- **Sealed TEST partition:** Zero access or exposure.

---

## 5. Preflight Regression Test Census

| Test File | Passed | Skipped | Status |
|---|:---:|:---:|:---:|
| `tests/test_rq1_trn_02b_kaggle_platform.py` | 37 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_colab_launch.py` | 33 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_preflight_r2_safety.py` | 11 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_corrections_r1.py` | 10 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_implementation_preflight.py` | 13 | 1 (AGENTS.md GPU boundary) | **PASS** |
| **Combined RQ1-TRN-02B Focused Census** | **104** | **1** | **PASS** |
| **Full Repository Regression Census** | **227** | **4** (CUDA-only placeholders) | **PASS** |

---

## 6. Remaining Operational Blockers & Recovery Protocol

### Remaining Blockers:
1. **Kaggle Account Phone Verification:** The account `minkhoaa` must complete SMS phone verification at `https://www.kaggle.com/settings` to unlock physical GPU instance provisioning.
2. **Kaggle GPU Provisioning:** Re-launch the pushed kernel once phone verification is completed.

### Safe Restart Command:
Once the operator completes SMS phone verification on Kaggle:
```bash
kaggle kernels push -p deploy/kaggle --accelerator NvidiaTeslaT4
kaggle kernels status minkhoaa/langfens-rq1-trn-02b-c0-gpu
```

### Sequential Execution Rule:
Under no circumstances may Treatment Arm C1 be initialized or executed until Control Arm C0 has completed 3,600 successful updates, satisfied all 9 endpoint invariants, and mirrored all checkpoints. C1 must be cold-initialized from the **SAME Baseline R checkpoint** (`best.pt`, SHA-256 `a2445c8d...`), never from C0.
