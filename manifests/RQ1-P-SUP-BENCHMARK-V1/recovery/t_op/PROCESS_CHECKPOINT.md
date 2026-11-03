# RQ1-P-SUP-BENCHMARK-V1 — Treatment T-OP Recovery Process Checkpoint

**State:** `DELIVERED_AND_VERIFIED; HISTORICAL_UPDATE_3600_RECOVERED; THIS_PROCESS_UPDATES_ZERO; INPUTS_PASS`.  
**Process Lead:** Agent A (Independent Verification & Recovery Subagent)  
**Worktree:** `/home/khoa/Projects/KLTN/psup-recover-t-op-v1`  
**Branch:** `review/psup-recover-t-op-v1` (base `40bc3770776ee1c99462d44101fdd145602b76f5`)  
**Timestamp:** `2026-11-03T18:15:00Z`  
**Policy Authority:** `AGENTS.md` (Mandatory Process Checkpoint Policy)

---

## 1. Scope & Execution Boundaries

- **Operator Authorization:** Authorized recovery and independent verification only.
- **Strict Prohibitions Enforced:**
  - NO local or remote training, backward passes, or optimizer updates executed in this process.
  - NO architecture modifications, configuration mutations, or hyperparameter changes.
  - NO access to, staging of, or extraction of sealed TEST audio/labels (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
  - NO remote settings mutations (Kaggle/HF permissions kept unchanged).
  - NO Git commits or push operations performed (review worktree preserved).
- **Process Accounting:**
  - `this_process_successful_updates`: **0**
  - `this_process_skipped_updates`: **0**
  - `this_process_backward_passes`: **0**
  - `this_process_optimizer_steps`: **0**

---

## 2. Remote Sources & Authentication Provenance

### 2.1 Kaggle GPU Training Platform
- **Kernel Slug:** `minkhoaa/langfens-rq1-psup-t-op-t4x2`
- **Official Status:** `KernelWorkerStatus.COMPLETE`
- **Hardware Verified:** Dual Tesla T4 (`NvidiaTeslaT4` machine shape, 15,360 MiB per GPU, Driver 580.178.04, CUDA 13.0)
- **Staging Dataset:** `minkhoaa/langfens-rq1-fold1-staging` (verified private)
- **Archive File:** `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` (761,922,918 bytes, SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`)
- **Pinned Git Commit:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`
- **Output Artifacts Retrieved:** Downloaded via official CLI `kaggle kernels output` and `kaggle kernels logs` to approved local scratch.

### 2.2 Hugging Face Checkpoint Persistence
- **Repository:** `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- **Visibility:** Verified Private (`private: True`)
- **Pinned Revision:** `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
- **Prefix:** `psup/t_op/`
- **Commit History:** Artifacts committed during training epochs and finalized in commit `c05b00e3e3c62be5225f1e17ca390f405364a096` before B-Control commits up to pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`. Tree SHA and file LFS pointers verified identical across revisions `c05b00e3`, `4aa9f2a6`, and `main`.

---

## 3. Recovered Artifacts & Cryptographic Verification

All raw artifacts were recovered into approved scratch directory outside Git:
`/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`

| Artifact Name | Size (bytes) | SHA-256 Checksum | HF Hub LFS Match | Kaggle Log Match | Historical Candidate Match |
|---|---|---|---|---|---|
| `fixed_budget_3600.pt` | 3,241,295 | `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` | **PASS** | **PASS** | **PASS** |
| `best.pt` | 3,239,578 | `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` | **PASS** | **PASS** | **PASS** |
| `last.pt` | 3,239,898 | `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811` | **PASS** | **PASS** | RECOVERED (absent in local Dataset) |
| `training_summary.json` | 3,339 | `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c` | **PASS** | **PASS** | **PASS** |
| `history.json` | 2,900 | `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425` | **PASS** | **PASS** | **PASS** |
| `langfens-rq1-psup-t-op-t4x2.ipynb` | 13,975 | `f94d610f209c775bb049d9766f02cf95652c1fc58b553c375e165b915a326de4` | N/A | **PASS** (Kaggle pull) | N/A |
| `kaggle_plain_execution.log` | 186,251 | `92f5ff522dec30a8d224ffc241ef32418b861e1f40d0811d8a1008783604df50` | N/A | **PASS** (Kaggle logs) | N/A |
| `kaggle_execution.log` | 338,333 | `d7ba5ab250d4792abb7739e52ad9394fe5467959c3da55f348ab628bb3f892b6` | N/A | **PASS** (Kaggle logs raw) | N/A |

### Historical Input Parity Note:
Historical local files at `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op/` were inspected in read-only mode and were **never modified or overwritten**.
The local candidate folder was missing `last.pt`, which has now been recovered and verified from both Hugging Face Hub LFS and Kaggle output storage.

---

## 4. Checkpoint Payload Inspection (CPU weights_only first)

All checkpoints were loaded safely on CPU using `torch.load(..., map_location='cpu', weights_only=True)` with minimal safe globals for numpy multiarray reconstruction (`numpy._core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype`).

### 4.1 Architecture & Parameter Accounting
- **Head Architecture:** 16 parameter tensors, total **266,667** parameters.
  - `acoustic_projection.weight` [128, 768], `bias` [128]
  - `phone_embedding.weight` [40, 128]
  - `position_embedding.weight` [256, 128]
  - `attention.in_proj_weight` [384, 128], `in_proj_bias` [384]
  - `attention.out_proj.weight` [128, 128], `out_proj.bias` [128]
  - `acoustic_norm.weight` [128], `bias` [128]
  - `operation_head.0.weight` [128, 256], `bias` [128]
  - `operation_head.3.weight` [3, 128], `bias` [3] (3-class operation classification: Correct, Substitution, Omission)
  - `ctc_head.weight` [40, 768], `bias` [40]
- **Finite Tensors Check:** 100% of head weight and bias tensors are strictly finite (0 NaN, 0 Inf).
- **Frozen Backbone:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) kept frozen; backbone weights not duplicated in head checkpoint.

### 4.2 Counter Verification & Endpoint Semantics
- **`fixed_budget_3600.pt`:**
  - `successful_updates`: **3,600**
  - `micro_steps`: **7,200** (effective batch size 4 = batch size 2 x grad_accum 2)
  - `skipped_amp_attempts_total`: **0**
  - `epoch`: **7** (8th epoch, 450 updates/epoch)
  - `is_fixed_budget_endpoint`: `True`
  - `best_ap`: `0.46497542138006975`
  - `patience`: `1`
  - Optimizer step: `3600.0` for all 16 parameters; `exp_avg` and `exp_avg_sq` 100% finite.
  - Scheduler last epoch: `3600`, learning rate `0.0008699219550575954`.
  - Scaler scale: `8192.0`.
- **`best.pt`:**
  - `successful_updates`: **2,700**
  - `micro_steps`: **5,400**
  - `skipped_amp_attempts_total`: **0**
  - `epoch`: **5** (6th epoch)
  - `is_fixed_budget_endpoint`: `False`
  - `best_ap`: `0.46497542138006975` (peak validation selection AP achieved at epoch 5)
  - `patience`: `0`
  - Optimizer step: `2700.0` for all 16 parameters; `exp_avg` and `exp_avg_sq` 100% finite.
  - Scheduler last epoch: `2700`, learning rate `0.0009267326725404599`.
  - Scaler scale: `8192.0`.
  - **Endpoint Semantics Clarification:** In accordance with the dual-checkpoint accounting contract, `best.pt` represents the early-stopping validation-selected checkpoint based on the `selection` partition (`MBMPS`, `TLV`). Since epoch 5 achieved the maximum selection AP (`0.464975`), update count for `best.pt` is legitimately **2,700 (< 3,600)**. `fixed_budget_3600.pt` represents the mandated 3,600-update fixed horizon checkpoint.
- **`last.pt`:**
  - `successful_updates`: **3,600**, `epoch`: **7**, `is_fixed_budget_endpoint`: `False`. Identical parameter and optimizer states to `fixed_budget_3600.pt`, differing only in the boolean `is_fixed_budget_endpoint` flag.

### 4.3 Objective & Contract Provenance
- `meta.registration`: `P-SUP-COMPACT-OP-V1`
- `meta.objective`: `operation_ce`
- `meta.contract_sha256`: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` (matches `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`)
- `meta.split_manifest_sha256`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (matches `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`)
- `meta.zip_sha256`: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (matches `authorized_RQ1-TRN-02A_fold1_train_val_only.bin`)
- `meta.git_sha`: `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` (matches training checkout HEAD in Kaggle execution logs)

### 4.4 Data Partition Identities & Zero Test Leakage
- `fit`: 1,799 utterances across 12 speakers: `['BWC', 'ERMS', 'HKK', 'HQTV', 'SKA', 'SVBI', 'TNI', 'TXHC', 'YBAA', 'YDCK', 'YKWK', 'ZHAA']`
- `selection` (`SELECT`): 300 utterances across 2 speakers: `['MBMPS', 'TLV']`
- `calibration` (`CAL`): 300 utterances across 2 speakers: `['NCC', 'THV']`
- `evaluation` (`VAL`): 300 utterances across 2 speakers: `['NJS', 'RRBI']` (held strictly unseen during training)
- `test` (`TEST`): 900 utterances across 6 speakers: `['ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV']` (strictly sealed; zero audio/labels staged or read)

---

## 5. Mismatches, Access Blockers & Unavailable Provenance

1. **Unfiltered Kaggle Output Download Interruption Note:**
   - The initial unfiltered `kaggle kernels output` invocation was cancelled via SIGINT by the parent orchestrator to abort redundant re-downloading of the full cloned repository tree.
   - **Crucial operational distinction:** This cancellation was an intentional operator intervention against bulk repository export, **not a checkpoint failure, network error, or data corruption**.
   - All checkpoint files (`checkpoints/RQ1-P-SUP/t_op/T-OP_treatment/`) had completely and successfully downloaded prior to the SIGINT interrupt.
   - All recovered binary files were cross-verified bit-identically against direct pinned Hugging Face Hub LFS downloads (`4aa9f2a68f0709889eacacf170b6446d3c45c73d`).
   - Operational directive incorporated: Future Kaggle artifact pulls must inspect files via `kaggle kernels files` and use bounded `--file-pattern` or direct Hugging Face pinned downloads with timeouts $\le 90$s.

2. **Parent Archival HF Benchmark Upload:**
   - The parent orchestrator committed subsequent archival benchmark evidence under the new prefix `benchmarks/psup-v1/` on Hugging Face Hub (commits `d8806b9e` and `6d1ec321`).
   - The target experimental directory `psup/t_op/` and the pinned commit `4aa9f2a68f0709889eacacf170b6446d3c45c73d` were untouched, verified bit-identical, and remain fully authoritative for this recovery.

3. **Source Git Commit Reconciliation:**
   - Checkpoint metadata and Kaggle execution logs explicitly verify that training ran on commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` ("fix(trainer): import datetime, timezone, and subprocess for checkpoint metadata").
   - Previous documentation in `INDEX.md` referenced `commit 9ad1e5e` (which was an evaluator compatibility fix committed later for PyTorch 2.6) and `df55f12` (the staging commit prior to the trainer metadata import fix).
   - This difference is an accounted documentation gap; the training execution source is cryptographically verified as `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.

4. **Kaggle Kernel Visibility Metadata vs Data Privacy:**
   - Kaggle API metadata reports `is_private=false` for the notebook entities.
   - However, the underlying Kaggle dataset `minkhoaa/langfens-rq1-fold1-staging` and the Hugging Face persistence repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` are verified strictly private (`private: True`).
   - No remote settings were modified in accordance with execution boundaries.

5. **History CLI (`deja`) Unavailable:**
   - The `deja` CLI tool is unavailable in this environment, as previously noted by the parent coordinator. Official Kaggle CLI logs and API metadata were used instead.

6. **Missing Local `last.pt` Candidate:**
   - The pre-existing local directory `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op/` only contained `best.pt`, `fixed_budget_3600.pt`, `history.json`, and `training_summary.json`.
   - `last.pt` was successfully recovered from Hugging Face Hub LFS and Kaggle output storage and independently verified.

---

## 6. Recovery Disposition & Input Gate

- `fixed_budget_3600.pt`: **PASS** (3,600 updates, all tensors finite, optimizer/scheduler verified, bit-identical readback)
- `best.pt`: **PASS** (2,700 updates, all tensors finite, optimizer/scheduler verified, bit-identical readback)
- `last.pt`: **PASS** (3,600 updates, all tensors finite, recovered from persistent remote stores)
- `training_summary.json` & `history.json`: **PASS** (bit-identical readback)
- Overall Recovery Disposition: **PASS**

---

## 7. Safe Restart Instructions

1. If restarting downstream evaluation of Treatment T-OP:
   - Use recovered checkpoints from approved scratch:
     - Fixed budget: `/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/fixed_budget_3600.pt`
     - Best selection: `/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/best.pt`
   - Use safe unpickling on CPU (`weights_only=True` with numpy safe globals).
   - Verify Evaluation Contract V1 (`4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`) and Fold 1 split manifest (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
2. Maintain strict speaker isolation: evaluate only on VAL (`NJS`, `RRBI`, 300 utterances). Never touch TEST speakers.
3. Keep raw checkpoints outside Git; maintain lightweight receipts in `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/t_op/`.
