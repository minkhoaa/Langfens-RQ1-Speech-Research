# Independent Recovery and Forensic Verification Report: Treatment T-OP (`operation_ce`)

**Document Authority:** Authoritative Subagent Artifact for Session Mission `f07cb35c-ee66-413d-9df0-e7bf68b7fac9`  
**Target Path:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/f07cb35c-ee66-413d-9df0-e7bf68b7fac9/recovery/T_OP_RECOVERY.md`  
**Worktree:** `/home/khoa/Projects/KLTN/psup-recover-t-op-v1`  
**Branch:** `review/psup-recover-t-op-v1` (base `40bc3770776ee1c99462d44101fdd145602b76f5`)  
**Evaluator:** Agent A (Sole Writer, Forensic Auditor & Verification Subagent)  
**Execution Date:** 2026-11-03T18:15:00Z  
**Final Status:** **PASS (All Required Fixed and Best Inputs Recovered and Cryptographically Verified)**

---

## 1. Executive Summary

This report delivers the forensic recovery, cryptographic readback, and deep tensor payload inspection of the completed **P-SUP Treatment T-OP** (`operation_ce`) experimental artifacts.

### Key Audit Findings:
1. **Full Recovery Achieved:** All required model artifacts (`fixed_budget_3600.pt`, `best.pt`, `last.pt`, `training_summary.json`, `history.json`), source notebook (`langfens-rq1-psup-t-op-t4x2.ipynb`), and runtime logs (`kaggle_plain_execution.log`, `kaggle_execution.log`) have been recovered into approved scratch storage (`/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`).
2. **Cryptographic Identity Parity:** Checksums for `fixed_budget_3600.pt`, `best.pt`, `training_summary.json`, and `history.json` match bit-identically across:
   - Hugging Face Hub LFS sidecar metadata and binary readback at revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`.
   - Kaggle official kernel outputs from `minkhoaa/langfens-rq1-psup-t-op-t4x2`.
   - Kaggle runtime execution logs.
   - Historical local candidates at `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op` (which were preserved unmodified).
3. **Local Gap Closed:** The historical local directory in `Dataset` was missing `last.pt`. `last.pt` (SHA-256 `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811`) was successfully recovered and verified from both Hugging Face Hub and Kaggle storage.
4. **Counter & Endpoint Semantics Verified:**
   - `fixed_budget_3600.pt`: exactly **3,600 successful updates**, **0 skipped AMP updates**, **7,200 micro-steps**, `epoch: 7`, `is_fixed_budget_endpoint: True`.
   - `best.pt`: exactly **2,700 successful updates**, **0 skipped AMP updates**, **5,400 micro-steps**, `epoch: 5`, `is_fixed_budget_endpoint: False`. Update count is legitimately **2,700 (< 3,600)** because validation selection AP reached its peak (`0.464975`) at epoch 5 on the `selection` partition (`MBMPS`, `TLV`). This fully validates the dual-checkpoint accounting contract mandated by `AGENTS.md`.
5. **Payload Integrity Verified:** Loaded safely on CPU using `torch.load(..., map_location='cpu', weights_only=True)` with minimal numpy reconstruction safe globals. All 16 head tensors (266,667 parameters) and all 16 optimizer moment states are strictly finite (0 NaN, 0 Inf).
6. **Execution Boundaries Preserved:** Zero training, backward passes, or optimizer updates were run in this process (`this_process_successful_updates: 0`). Zero test partition audio/labels were accessed or staged.

---

## 2. Artifact Registry & Checksums

### 2.1 Recovered Raw Artifacts
Approved Local Scratch Directory:  
`/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`

| File Name | Size (Bytes) | SHA-256 Checksum | Hugging Face LFS SHA-256 | Kaggle Log SHA-256 | Local Historical Status |
|---|---|---|---|---|---|
| `fixed_budget_3600.pt` | 3,241,295 | `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` | `a8559b38...` (MATCH) | `a8559b38...` (MATCH) | Identical match |
| `best.pt` | 3,239,578 | `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` | `3b492049...` (MATCH) | `3b492049...` (MATCH) | Identical match |
| `last.pt` | 3,239,898 | `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811` | `93c65bc5...` (MATCH) | `93c65bc5...` (MATCH) | Recovered (was missing locally) |
| `training_summary.json` | 3,339 | `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c` | `09695993...` (MATCH) | `09695993...` (MATCH) | Identical match |
| `history.json` | 2,900 | `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425` | `6ef6ec6e...` (MATCH) | `6ef6ec6e...` (MATCH) | Identical match |
| `langfens-rq1-psup-t-op-t4x2.ipynb` | 13,975 | `f94d610f209c775bb049d9766f02cf95652c1fc58b553c375e165b915a326de4` | N/A | Kaggle source pull | N/A |
| `kaggle_plain_execution.log` | 186,251 | `92f5ff522dec30a8d224ffc241ef32418b861e1f40d0811d8a1008783604df50` | N/A | Extracted plain log | N/A |
| `kaggle_execution.log` | 338,333 | `d7ba5ab250d4792abb7739e52ad9394fe5467959c3da55f348ab628bb3f892b6` | N/A | Raw streaming log | N/A |

### 2.2 Historical Candidates at `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op`
- Historical files were audited without mutation:
  - `best.pt`: SHA-256 `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` (Identical)
  - `fixed_budget_3600.pt`: SHA-256 `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` (Identical)
  - `history.json`: SHA-256 `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425` (Identical)
  - `training_summary.json`: SHA-256 `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c` (Identical)
  - `last.pt`: Missing from local candidate set.

---

## 3. Remote Provenance & Execution Environment

### 3.1 Kaggle Kernel Execution
- **Kernel Identifier:** `minkhoaa/langfens-rq1-psup-t-op-t4x2`
- **Reported Worker Status:** `KernelWorkerStatus.COMPLETE`
- **Execution Target:** Tesla T4 x 2 (`machine_shape: NvidiaTeslaT4`, Driver 580.178.04, CUDA 13.0)
- **Staging Dataset:** `minkhoaa/langfens-rq1-fold1-staging` (verified private)
- **Archive Verified by Notebook:** `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` (761,922,918 bytes, SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`)
- **Execution Command (from notebook and log):**
  ```bash
  python3 audit_scripts/train_p_sup.py \
    --arm treatment \
    --objective operation_ce \
    --split-manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
    --zip-path /kaggle/input/datasets/minkhoaa/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.bin \
    --output-dir /kaggle/working/checkpoints/RQ1-P-SUP/t_op \
    --seed 1337 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --warmup-updates 100 \
    --max-updates 3600 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --max-epochs 25 \
    --train-partition fit \
    --select-partition selection \
    --cal-partition calibration \
    --val-partition val \
    --remote-execution \
    --remote-mirror-target minkhoaa/langfens-rq1-cross-fold-checkpoints \
    --remote-mirror-backend huggingface
  ```

### 3.2 Hugging Face Checkpoint Persistence
- **Repository:** `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- **Privacy Status:** Confirmed Private (`private: True`)
- **Pinned Git Revision:** `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
- **Direct Upload Commit for T-OP:** `c05b00e3e3c62be5225f1e17ca390f405364a096`
- **Tree Verification:** Identical file trees and LFS pointers across revisions `c05b00e3`, `4aa9f2a6`, and `main`.

---

## 4. Checkpoint Payload Inspection Details

All checkpoints were loaded and audited on CPU using `torch.load(..., map_location='cpu', weights_only=True)` with minimal numpy safe globals (`np._core.multiarray._reconstruct`, `np.ndarray`, `np.dtype`, `np.dtypes.UInt32DType`).

### 4.1 Architecture & Parameter Budget
- **Parameter Count:** Exactly **266,667** trainable parameters in 16 head tensors:
  - `acoustic_projection.weight`: shape `[128, 768]`, dtype `float32` (98,304 params)
  - `acoustic_projection.bias`: shape `[128]`, dtype `float32` (128 params)
  - `phone_embedding.weight`: shape `[40, 128]`, dtype `float32` (5,120 params)
  - `position_embedding.weight`: shape `[256, 128]`, dtype `float32` (32,768 params)
  - `attention.in_proj_weight`: shape `[384, 128]`, dtype `float32` (49,152 params)
  - `attention.in_proj_bias`: shape `[384]`, dtype `float32` (384 params)
  - `attention.out_proj.weight`: shape `[128, 128]`, dtype `float32` (16,384 params)
  - `attention.out_proj.bias`: shape `[128]`, dtype `float32` (128 params)
  - `acoustic_norm.weight`: shape `[128]`, dtype `float32` (128 params)
  - `acoustic_norm.bias`: shape `[128]`, dtype `float32` (128 params)
  - `operation_head.0.weight`: shape `[128, 256]`, dtype `float32` (32,768 params)
  - `operation_head.0.bias`: shape `[128]`, dtype `float32` (128 params)
  - `operation_head.3.weight`: shape `[3, 128]`, dtype `float32` (384 params)
  - `operation_head.3.bias`: shape `[3]`, dtype `float32` (3 params)
  - `ctc_head.weight`: shape `[40, 768]`, dtype `float32` (30,720 params)
  - `ctc_head.bias`: shape `[40]`, dtype `float32` (40 params)
- **Finiteness:** 100% of head tensors are strictly finite (`torch.isfinite(p).all().item() == True`).
- **Objective:** Multi-class operation cross-entropy (`operation_ce`), outputting 3 logits (0: Correct, 1: Substitution, 2: Omission) plus joint CTC phoneme recognition head (40 classes).
- **Frozen Backbone:** Pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) remained frozen.

### 4.2 Endpoint Accounting & Counters

#### A. `fixed_budget_3600.pt` (Mandatory Fixed Horizon Endpoint)
- `successful_updates`: `3600`
- `micro_steps`: `7200` (2 micro-steps per update)
- `skipped_amp_attempts_total`: `0`
- `epoch`: `7` (completed 8 epochs of 450 updates each)
- `is_fixed_budget_endpoint`: `True`
- `best_ap`: `0.46497542138006975`
- `selection_ap`: `0.4642429960853022`
- `patience`: `1`
- `optimizer`: 16 parameter state entries, step counter `3600.0` across all 16 entries; all first/second moments strictly finite.
- `scheduler`: CosineAnnealingLR, `last_epoch: 3600`, learning rate `0.0008699219550575954`.
- `scaler`: GradScaler scale `8192.0`.

#### B. `best.pt` (Validation-Selected Endpoint)
- `successful_updates`: `2700`
- `micro_steps`: `5400`
- `skipped_amp_attempts_total`: `0`
- `epoch`: `5` (completed 6 epochs of 450 updates each)
- `is_fixed_budget_endpoint`: `False`
- `best_ap`: `0.46497542138006975` (achieved maximum selection AP at update 2,700)
- `patience`: `0`
- `optimizer`: 16 parameter state entries, step counter `2700.0` across all 16 entries; all moments strictly finite.
- `scheduler`: `last_epoch: 2700`, learning rate `0.0009267326725404599`.
- `scaler`: GradScaler scale `8192.0`.

#### C. `last.pt` (Final Epoch State)
- `successful_updates`: `3600`, `micro_steps`: `7200`, `skipped_amp_attempts_total`: `0`, `epoch`: `7`, `is_fixed_budget_endpoint`: `False`. Identical parameter and optimizer states to `fixed_budget_3600.pt`.

---

## 5. Contract, Split, and Isolation Integrity

### 5.1 Cryptographic Hash Ledger
- **Binding Evaluation Contract V1:**
  - Checkpoint metadata `contract_sha256`: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`
  - Local repository `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` (MATCH)
- **Split Manifest:**
  - Checkpoint metadata `split_manifest_sha256`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`
  - Local repository `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (MATCH)
- **Dataset Archive:**
  - Checkpoint metadata `zip_sha256`: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`
  - Local archive `/home/khoa/Projects/KLTN/Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip`: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (MATCH)

### 5.2 Speaker Partition Disjointness & Zero Test Leakage
- **Fit Partition (1,799 utterances, 12 speakers):**
  `['BWC', 'ERMS', 'HKK', 'HQTV', 'SKA', 'SVBI', 'TNI', 'TXHC', 'YBAA', 'YDCK', 'YKWK', 'ZHAA']`
- **Selection Partition (300 utterances, 2 speakers):**
  `['MBMPS', 'TLV']`
- **Calibration Partition (300 utterances, 2 speakers):**
  `['NCC', 'THV']`
- **Evaluation Partition (300 utterances, 2 speakers):**
  `['NJS', 'RRBI']` (Held strictly unseen; zero forward/loss during training)
- **Sealed Test Partition (900 utterances, 6 speakers):**
  `['ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV']` (Strictly absent from staging archive; 0 utterances, 0 labels accessed)

Total speakers: 16 training speakers (12 fit + 2 selection + 2 calibration) + 2 validation speakers = 18 speakers. All subsets are pairwise disjoint ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$).

---

## 6. Discrepancies, Blockers, and Accounting Limits

1. **Unfiltered Kaggle Output Download Interruption Note:**
   - The initial unfiltered `kaggle kernels output` invocation was cancelled via SIGINT by the parent orchestrator to abort redundant re-downloading of the full cloned repository tree.
   - **Crucial operational distinction:** This cancellation was an intentional operator intervention against bulk repository export, **not a checkpoint failure, network error, or data corruption**.
   - All checkpoint files (`checkpoints/RQ1-P-SUP/t_op/T-OP_treatment/`) had completely and successfully downloaded prior to the SIGINT interrupt.
   - All recovered binary files were cross-verified bit-identically against direct pinned Hugging Face Hub LFS downloads (`4aa9f2a68f0709889eacacf170b6446d3c45c73d`).
   - Operational directive incorporated: Future Kaggle artifact pulls must inspect files via `kaggle kernels files` and use bounded `--file-pattern` or direct Hugging Face pinned downloads with timeouts $\le 90$s.

2. **Parent Archival HF Benchmark Upload:**
   - The parent orchestrator committed subsequent archival benchmark evidence under the new prefix `benchmarks/psup-v1/` on Hugging Face Hub (commits `d8806b9e` and `6d1ec321`).
   - The target experimental directory `psup/t_op/` and the pinned commit `4aa9f2a68f0709889eacacf170b6446d3c45c73d` were untouched, verified bit-identical, and remain fully authoritative for this recovery.

3. **Source Commit SHA Discrepancy Reconciled:**
   - In checkpoint metadata and Kaggle execution logs, the pinned and executed commit is:
     `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` ("fix(trainer): import datetime, timezone, and subprocess for checkpoint metadata").
   - Previous documentation in `INDEX.md` noted `commit 9ad1e5e` (which was an evaluator compatibility fix committed later for PyTorch 2.6) and `df55f12` (the staging commit prior to the trainer metadata import fix).
   - This difference is an accounted documentation gap; training execution source is cryptographically verified as `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.
4. **Kaggle Kernel Visibility Metadata:**
   - Kaggle API metadata reports `is_private=false` for the notebook entity.
   - However, the underlying Kaggle dataset `minkhoaa/langfens-rq1-fold1-staging` and the Hugging Face checkpoint repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` are verified private (`private: True`).
   - No remote settings were modified.
5. **Historical Candidate Gap Closed:**
   - The pre-existing local directory `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op/` was missing `last.pt`.
   - `last.pt` was successfully recovered from Hugging Face Hub LFS and Kaggle output storage.
6. **History CLI (`deja`) Unavailable:**
   - As established by the parent session, the `deja` CLI is not installed in this environment. Official Kaggle CLI logs and API metadata were used instead.

---

## 7. Receipts and Verification Test Evidence

- **Worktree Receipts Written:**
  - JSON Receipt: `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/t_op/RECOVERY_RECEIPT.json` (SHA-256 verified)
  - Process Checkpoint: `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/t_op/PROCESS_CHECKPOINT.md`
- **Unit and Regression Test Suite:**
  - Test file: `tests/test_t_op_recovery_receipt.py`
  - Command: `python3 -m unittest -v tests/test_t_op_recovery_receipt.py`
  - Output:
    ```
    test_canonical_artifact_hashes_and_files (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_canonical_artifact_hashes_and_files) ... ok
    test_checkpoint_payload_integrity (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_checkpoint_payload_integrity) ... ok
    test_receipt_structure_and_status (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_receipt_structure_and_status) ... ok
    test_speaker_partitions_and_zero_test_leakage (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_speaker_partitions_and_zero_test_leakage) ... ok

    ----------------------------------------------------------------------
    Ran 4 tests in 0.047s

    OK
    ```

---

## 8. Final Disposition & Downstream Handoff

- **Required Inputs Recovery Status:**
  - `fixed_budget_3600.pt`: **PASS**
  - `best.pt`: **PASS**
  - `last.pt`: **PASS**
  - `training_summary.json`: **PASS**
  - `history.json`: **PASS**
  - Logs and source notebook: **PASS**
- **Overall Recovery Status:** **PASS**

### Recommended Next Steps for Downstream Agents:
1. Proceed with evaluation of Treatment T-OP on Fold 1 VAL (`NJS`, `RRBI`, 300 utterances) using the recovered `fixed_budget_3600.pt` and `best.pt` checkpoints from scratch `/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`.
2. Execute evaluation strictly under Evaluation Contract V1 (`4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`) and Fold 1 split manifest (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
3. Maintain zero leakage policy: ensure sealed test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) remain strictly excluded.
