# Independent Recheck & Forensic Verification Report: Treatment T-OP (`operation_ce`)

**Document Authority:** Authoritative Subagent Artifact for Task Session `3dce8806-0948-45cf-a636-47a0095821ed`  
**Target Path:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-1.md`  
**Investigator:** Agent A (Parallel Forensic Auditor & Recovery Recheck Subagent)  
**Verification Date:** 2026-11-04T02:40:00Z  
**Coordinator Repo:** `/home/khoa/Projects/KLTN/Dataset`  
**Coordinator Branch / HEAD:** `review/psup-recovery-benchmark-v1` at commit `ea16b83`  
**Execution Environment:** Read-only verification via `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python` (Python 3.11.17, PyTorch 2.6.0+cu124)  
**Status:** **RECHECK_COMPLETED; CRYPTOGRAPHIC_AND_PAYLOAD_VERIFIED; ACCOUNTING_MISMATCH_RECONCILED; NOT_A_BLANKET_SCIENTIFIC_PASS**

---

## 1. Executive Recheck Summary & Mandate

Under user directive for independent parallel recovery and benchmark recheck (without initiating training, backwards passes, optimizer steps, full pytests, or sealed test partition access), Agent A conducted an end-to-end read-only cryptographic, structural, and numerical re-audit of the recovered **P-SUP Treatment T-OP** (`operation_ce`) artifacts.

### Key Audit Findings
1. **Zero Redundant Downloads:** Existing verified artifacts in local approved scratch storage (`/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`) were inspected directly; no duplicate remote bulk downloads or unfiltered Kaggle outputs were issued.
2. **Cryptographic Parity Across 5 Required Artifacts:**
   All five primary required artifacts (`fixed_budget_3600.pt`, `best.pt`, `last.pt`, `training_summary.json`, `history.json`) match bit-identically across:
   - Local canonical recovery cache,
   - Private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` at pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`,
   - Kaggle official kernel execution outputs from `minkhoaa/langfens-rq1-psup-t-op-t4x2`,
   - Historical local candidates at `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op` (which previously lacked `last.pt`).
3. **Safe CPU Payload Inspection (`weights_only=True`):**
   Using `.venv-cuda/bin/python` with explicit NumPy safe globals (`np._core.multiarray._reconstruct`, `np.ndarray`, `np.dtype`, `np.dtypes.UInt32DType`), all PyTorch `.pt` payloads were safely deserialized without executing untrusted code. All 16 trainable head parameter tensors (exactly 266,667 parameters) and all 16 optimizer moment states are 100% strictly finite (0 NaN, 0 Inf).
4. **Endpoint Semantics & Dual Checkpointing:**
   - `fixed_budget_3600.pt`: exactly **3,600 successful updates**, **0 skipped AMP updates**, **7,200 micro-steps**, `epoch: 7`, optimizer step counter `3600` on all 16 tensors, scheduler `last_epoch: 3600`, learning rate `0.0008699219550575954`, `is_fixed_budget_endpoint: True`.
   - `best.pt`: exactly **2,700 successful updates**, **0 skipped AMP updates**, **5,400 micro-steps**, `epoch: 5`, optimizer step counter `2700` on all 16 tensors, scheduler `last_epoch: 2700`, learning rate `0.0009267326725404599`, `is_fixed_budget_endpoint: False`. Selected at peak validation AP (`0.464975`) on the internal selection partition (`MBMPS`, `TLV`).
   - `last.pt`: exactly **3,600 successful updates**, `epoch: 7`. Bit-identical head weights and optimizer moments to `fixed_budget_3600.pt`.
5. **Exact Root-Cause Accounting of the Embedded History Mismatch (3150 vs 3600):**
   The embedded `history` inside `fixed_budget_3600.pt` contains 7 epoch records ending at update `3150`, while the checkpoint counters and optimizer state are at `3600`. Source code inspection of `src/langfens_prep/p_sup/trainer.py` (lines 1162–1186 at commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`) proves this is an exact artifact snapshot timing phenomenon: `fixed_budget_3600.pt` was snapshotted inside the training loop immediately when `ledger.successful` hit 3,600, *before* epoch 7's post-epoch validation evaluation executed to append the 8th history entry. Once epoch 7 validation completed, the final record was appended, which is why `last.pt`, `history.json`, and `training_summary.json` correctly contain all 8 epoch entries ending at 3,600 updates.
6. **Execution Boundaries & Discrepancies:**
   Zero backward passes, zero optimizer steps, and zero test partition audio/labels were accessed in this process. No blanket scientific pass is declared: candidate validation metrics must be evaluated on Fold-1 VAL under genuine protocols (O and Z), separate from the internal selection partition metrics recorded during training.

---

## 2. Artifact Registry & Checksum Verification

### 2.1 All Five Required Artifacts (and Execution Evidence)
Approved Local Cache: `/home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/`

| Artifact Filename | File Size (Bytes) | SHA-256 Checksum | Hugging Face LFS / Readback SHA-256 | Local Historical Candidate Status |
|---|---|---|---|---|
| `fixed_budget_3600.pt` | 3,241,295 | `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` | `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` | Identical bit-for-bit match |
| `best.pt` | 3,239,578 | `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` | `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` | Identical bit-for-bit match |
| `last.pt` | 3,239,898 | `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811` | `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811` | Recovered (missing from Dataset/) |
| `training_summary.json` | 3,339 | `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c` | `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c` | Identical bit-for-bit match |
| `history.json` | 2,900 | `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425` | `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425` | Identical bit-for-bit match |
| `langfens-rq1-psup-t-op-t4x2.ipynb` | 13,975 | `f94d610f209c775bb049d9766f02cf95652c1fc58b553c375e165b915a326de4` | N/A (Kaggle notebook pull) | N/A |
| `kaggle_plain_execution.log` | 186,251 | `92f5ff522dec30a8d224ffc241ef32418b861e1f40d0811d8a1008783604df50` | N/A (Extracted plain log) | N/A |
| `kaggle_execution.log` | 338,333 | `d7ba5ab250d4792abb7739e52ad9394fe5467959c3da55f348ab628bb3f892b6` | N/A (Raw streamed log) | N/A |

### 2.2 Independent Checksum Command Output
```bash
$ sha256sum /home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/*
3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9  .../best.pt
a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9  .../fixed_budget_3600.pt
6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425  .../history.json
d7ba5ab250d4792abb7739e52ad9394fe5467959c3da55f348ab628bb3f892b6  .../kaggle_execution.log
92f5ff522dec30a8d224ffc241ef32418b861e1f40d0811d8a1008783604df50  .../kaggle_plain_execution.log
f94d610f209c775bb049d9766f02cf95652c1fc58b553c375e165b915a326de4  .../langfens-rq1-psup-t-op-t4x2.ipynb
93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811  .../last.pt
096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c  .../training_summary.json
```

---

## 3. Safe CPU Payload Inspection & Tensor Finiteness

### 3.1 Deserialization Protocol
In strict compliance with PyTorch 2.6 security standards, checkpoints were loaded on CPU without invoking arbitrary Python code:
```python
import numpy as np
import torch
safe_globals = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
with torch.serialization.safe_globals(safe_globals):
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

### 3.2 Parameter Budget & Weight Architecture (266,667 Parameters)
Across all three checkpoints (`fixed_budget_3600.pt`, `best.pt`, `last.pt`), the model architecture is confirmed identical:
- **Total Head Parameters:** Exactly **266,667** float32 parameters across 16 tensors:
  - `acoustic_projection.weight`: shape `[128, 768]` (98,304 params)
  - `acoustic_projection.bias`: shape `[128]` (128 params)
  - `phone_embedding.weight`: shape `[40, 128]` (5,120 params)
  - `position_embedding.weight`: shape `[256, 128]` (32,768 params)
  - `attention.in_proj_weight`: shape `[384, 128]` (49,152 params)
  - `attention.in_proj_bias`: shape `[384]` (384 params)
  - `attention.out_proj.weight`: shape `[128, 128]` (16,384 params)
  - `attention.out_proj.bias`: shape `[128]` (128 params)
  - `acoustic_norm.weight`: shape `[128]` (128 params)
  - `acoustic_norm.bias`: shape `[128]` (128 params)
  - `operation_head.0.weight`: shape `[128, 256]` (32,768 params)
  - `operation_head.0.bias`: shape `[128]` (128 params)
  - `operation_head.3.weight`: shape `[3, 128]` (384 params)
  - `operation_head.3.bias`: shape `[3]` (3 params)
  - `ctc_head.weight`: shape `[40, 768]` (30,720 params)
  - `ctc_head.bias`: shape `[40]` (40 params)
- **Finiteness Check:** 100% of head parameter tensors and 100% of optimizer moment tensors (`exp_avg`, `exp_avg_sq`) are strictly finite (`torch.isfinite(t).all() == True`). Zero NaN or Inf values detected.
- **Head Weights Equivalence:** Direct tensor equality comparison (`torch.equal`) confirmed that `fixed_budget_3600.pt` and `last.pt` have 100% bit-for-bit identical head weights and optimizer moments.

---

## 4. Counter & Endpoint Semantics Verification

### 4.1 Comparative Endpoint Ledger

| Field / Attribute | `fixed_budget_3600.pt` | `best.pt` | `last.pt` | Verification Status |
|---|---|---|---|---|
| `successful_updates` | `3600` | `2700` | `3600` | VERIFIED |
| `micro_steps` | `7200` | `5400` | `7200` | VERIFIED (2 micro-steps/update) |
| `skipped_amp_attempts_total` | `0` | `0` | `0` | VERIFIED (Zero AMP skips) |
| `epoch` | `7` | `5` | `7` | VERIFIED |
| `is_fixed_budget_endpoint` | `True` | `False` | `False` | VERIFIED |
| Optimizer States Count | 16 | 16 | 16 | VERIFIED |
| Optimizer Step Counters | `[3600] * 16` | `[2700] * 16` | `[3600] * 16` | VERIFIED |
| Scheduler `last_epoch` | `3600` | `2700` | `3600` | VERIFIED |
| Scheduler Learning Rate | `[0.0008699219550575954]` | `[0.0009267326725404599]` | `[0.0008699219550575954]` | VERIFIED |
| GradScaler `scale` | `8192.0` | `8192.0` | `8192.0` | VERIFIED |
| RNG State Keys | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | VERIFIED |
| Embedded History Length | 7 epochs (ends update 3150) | 6 epochs (ends update 2700) | 8 epochs (ends update 3600) | RECONCILED (Snapshot timing) |
| Selection AP (`best_ap`) | `0.46497542138006975` | `0.46497542138006975` | `0.46497542138006975` | Peak achieved at Epoch 5 |

### 4.2 Endpoint Accounting Justification
- `best.pt` legitimately holds **2,700 successful updates** because validation selection AP reached its maximum (`0.464975`) at Epoch 5 on the `selection` partition (`MBMPS`, `TLV`). In accordance with early-stopping patience rules, the selection checkpoint preserves the weights at that peak.
- `fixed_budget_3600.pt` legitimately holds **3,600 successful updates**, representing the fixed-budget 3,600-step training horizon required by `AGENTS.md` dual-checkpoint accounting.

---

## 5. Root-Cause Analysis of Embedded Fixed History Mismatch (3150 vs 3600)

### 5.1 Observation
- In `fixed_budget_3600.pt`:
  - `successful_updates`: `3600`
  - Optimizer step counters: `3600` across all 16 parameter states
  - Scheduler `last_epoch`: `3600`
  - `history` array length: `7` (epochs 0 through 6), with `history[-1]['successful_updates'] == 3150`.
- In `last.pt`, `history.json`, and `training_summary.json`:
  - `history` array length: `8` (epochs 0 through 7), with `history[-1]['successful_updates'] == 3600`.

### 5.2 Architectural Cause in `src/langfens_prep/p_sup/trainer.py`
Inspection of the executed training script at commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`:
```python
# Lines 1162-1186: Inside the batch accumulation loop of Epoch 7:
if ledger.fixed_budget_reached and not fixed_budget_saved:
    fb_path = output_dir / "fixed_budget_3600.pt"
    save_checkpoint(
        path=fb_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        successful_updates=ledger.successful,  # 3600
        micro_steps=micro_steps,              # 7200
        best_ap=best_ap,
        patience=patience,
        skipped_amp=ledger.skipped,           # 0
        history=history,                      # <--- history at this moment only has epochs 0..6
        metadata=metadata,
        is_fixed_budget_endpoint=True,
        ...
    )
    fixed_budget_saved = True

# Lines 1195-1215: Post-epoch validation evaluation executed AFTER batch loop completes:
val_metrics = evaluate_val_set(adapter, selection_loader, device, objective)
epoch_record = {
    "epoch": epoch,
    "successful_updates": ledger.successful,
    "val_ap": val_ap,
    ...
}
history.append(epoch_record)  # <--- 8th record (epoch 7, 3600 updates) appended HERE!

# Lines 1245-1260: Final checkpoint saving:
save_checkpoint(last_path, ..., history=history, is_fixed_budget_endpoint=False)
```

### 5.3 Audit Conclusion on Mismatch
The mismatch is **not** an indication that training terminated early or that only 3,150 updates were performed. The model weights, optimizer moments, scheduler counters, and update ledgers conclusively prove that 3,600 updates were completed. The embedded history stopped at 3,150 solely because the fixed-budget checkpoint was serialized inside the inner training step loop immediately upon reaching update 3,600, prior to the end-of-epoch validation call that appended the 8th history entry.

---

## 6. Remote Provenance & Metadata Verification

### 6.1 Hugging Face Hub Checkpoint Persistence
- **Repository:** `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- **Privacy Status:** Confirmed **Private** (`private: True`).
- **Pinned Git Revision:** `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
- **Remote Files & LFS Pointers in `psup/t_op/`:**
  - `best.pt`: 3,239,578 bytes, LFS SHA-256 `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9`
  - `fixed_budget_3600.pt`: 3,241,295 bytes, LFS SHA-256 `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9`
  - `fixed_budget_checkpoint.pt`: 3,241,295 bytes, LFS SHA-256 `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9`
  - `last.pt`: 3,239,898 bytes, LFS SHA-256 `93c65bc5ceef60c9d985492b9a1e8d9e4042cbf866a837815c705ac0e8237811`
  - `history.json`: 2,900 bytes, SHA-256 `6ef6ec6e3f553e94084d971115ba458833e7657676ff185b08c2b70a23e7b425`
  - `training_summary.json`: 3,339 bytes, SHA-256 `096959938ddbb8b0ef3fd9cf8fee9452a41e88e029d865a6cd943c3a4f995c5c`
- **Integrity Verification:** 100% cryptographic parity across all files.

### 6.2 Kaggle Remote Execution
- **Kernel Identifier:** `minkhoaa/langfens-rq1-psup-t-op-t4x2`
- **Status:** `KernelWorkerStatus.COMPLETE`
- **Machine Shape:** `NvidiaTeslaT4` (2x Tesla T4 GPUs, NVIDIA Driver 580.178.04, CUDA 13.0)
- **Staging Dataset:** `minkhoaa/langfens-rq1-fold1-staging` (verified private, SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`)
- **Kernel Output Inspection:** Verified via bounded `kaggle kernels files` metadata call; zero unfiltered downloads executed.

---

## 7. Source Code Execution Evidence & Limitations

### 7.1 Verified Execution Commit
- **Execution SHA:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` ("fix(trainer): import datetime, timezone, and subprocess for checkpoint metadata").
- **Evidence in Logs:** `kaggle_plain_execution.log` confirms checkout and HEAD verification:
  ```
  [GIT] Checking out pinned training commit: dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f...
  HEAD is now at dd2dd89 fix(trainer): import datetime, timezone, and subprocess for checkpoint metadata
  [GIT] Verified Training Commit HEAD: dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f
  ```
- **Evidence in Payloads:** All three checkpoint metadata blocks record `git_sha: "dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f"`.

### 7.2 Discrepancies and Limitations
1. **Historical Commit Documentation Gap:**
   Earlier entries in `INDEX.md` and interim reports referenced commits `9ad1e5e` (a subsequent PyTorch 2.6 evaluator fix) or `df55f12` (the staging commit before adding metadata import fixes). The actual executed training code on Kaggle was `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`. This discrepancy is an accounted documentation gap and does not affect the validity of the recovered weights.
2. **Kaggle Kernel Visibility Metadata:**
   The Kaggle notebook metadata API reports `is_private=false` for the kernel container. However, the underlying dataset `minkhoaa/langfens-rq1-fold1-staging` and the Hugging Face checkpoint repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` are confirmed private.
3. **Training Log Granularity:**
   The remote execution logs record epoch validation summaries on the `selection` partition (`MBMPS`, `TLV`). They do not contain utterance-by-utterance decoded strings or token rows for the unseen validation set (`NJS`, `RRBI`).
4. **Validation Metric Distinction (Not a Scientific PASS):**
   The selection AP reported during training (`0.464975` for `best.pt` and `0.464243` for `fixed_budget_3600.pt`) was computed on the 300 `selection` partition utterances, NOT on Fold-1 VAL (`NJS`, `RRBI`). Candidate evaluation on Fold-1 VAL requires separate benchmark inference (under both protocol O and genuine Z) and must be independently recomputed by coordinator runners.

---

## 8. Data Partitioning, Isolation & Contract Integrity

### 8.1 Partition Disjointness
Metadata partition inspection of the recovered checkpoints confirms strict compliance with the evaluation contract:
- **Fit Partition (1,799 utterances, 12 speakers):**
  `['BWC', 'ERMS', 'HKK', 'HQTV', 'SKA', 'SVBI', 'TNI', 'TXHC', 'YBAA', 'YDCK', 'YKWK', 'ZHAA']`
- **Selection Partition (300 utterances, 2 speakers):**
  `['MBMPS', 'TLV']`
- **Calibration Partition (300 utterances, 2 speakers):**
  `['NCC', 'THV']`
- **Evaluation Partition (300 utterances, 2 speakers):**
  `['NJS', 'RRBI']` (Held strictly unseen during training)
- **Sealed Test Partition (900 utterances, 6 speakers):**
  `['ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV']` (Strictly absent from training staging archive)

All subsets are pairwise disjoint ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$). Zero test partition audio or labels were staged or accessed.

### 8.2 Binding Evaluation Hashes
- Contract SHA-256: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` (MATCH)
- Split Manifest SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (MATCH)
- Dataset Archive SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (MATCH)

---

## 9. Independent Test Execution Receipt

Ran dedicated unit and regression verification tests:
```bash
$ /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -m unittest -v tests/test_t_op_recovery_receipt.py
test_canonical_artifact_hashes_and_files (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_canonical_artifact_hashes_and_files) ... ok
test_checkpoint_payload_integrity (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_checkpoint_payload_integrity) ... ok
test_receipt_structure_and_status (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_receipt_structure_and_status) ... ok
test_speaker_partitions_and_zero_test_leakage (tests.test_t_op_recovery_receipt.TestTreatmentTOpRecovery.test_speaker_partitions_and_zero_test_leakage) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.030s

OK
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Executed independent read-only recheck of recovered T-OP artifacts without widening scope; verified sha256 checksums, loaded payloads on CPU via .venv-cuda with numpy safe_globals weights_only=True, validated counters (3600 fixed/last, 2700 best, 0 skips, operation_ce), explained embedded history mismatch (3150 vs 3600), inspected source dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f evidence, and maintained zero training/optimizer/backward executions."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Generated comprehensive recheck report at managed path /home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-1.md with full artifact registry, sha256 hashes, remote HF/Kaggle provenance, python inspection commands, unit test output, and documented discrepancies."
    }
  ],
  "changedFiles": [
    "/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-1.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git -C /home/khoa/Projects/KLTN/Dataset status -s",
      "result": "passed",
      "summary": "Verified coordinator repository status and uncommitted files"
    },
    {
      "command": "git -C /home/khoa/Projects/KLTN/Dataset log -n 5 --oneline",
      "result": "passed",
      "summary": "Verified HEAD commit ea16b83 and parent commits"
    },
    {
      "command": "sha256sum /home/khoa/.cache/langfens/psup-benchmark-v1/t_op/canonical_recovery/*",
      "result": "passed",
      "summary": "Computed SHA-256 for all 8 recovered files in canonical cache"
    },
    {
      "command": "sha256sum /home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/t_op/*",
      "result": "passed",
      "summary": "Verified bit-identical checksums for historical candidate artifacts"
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python remote HF check",
      "result": "passed",
      "summary": "Verified private HF repo and LFS SHA-256 pointers at revision 4aa9f2a6"
    },
    {
      "command": "kaggle kernels status minkhoaa/langfens-rq1-psup-t-op-t4x2",
      "result": "passed",
      "summary": "Verified COMPLETE kernel worker status"
    },
    {
      "command": "kaggle kernels files minkhoaa/langfens-rq1-psup-t-op-t4x2",
      "result": "passed",
      "summary": "Read bounded file metadata without issuing unfiltered bulk output downloads"
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python payload inspection script",
      "result": "passed",
      "summary": "Safely inspected fixed_budget_3600.pt, best.pt, and last.pt with numpy safe globals; verified finite tensors, counters, and history length"
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python weight equivalence test",
      "result": "passed",
      "summary": "Verified 100% bit-for-bit weight and optimizer moment identity between fixed_budget_3600.pt and last.pt"
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -m unittest -v tests/test_t_op_recovery_receipt.py",
      "result": "passed",
      "summary": "All 4 T-OP recovery unit/regression tests passed in 0.030s"
    }
  ],
  "validationOutput": [
    "All 5 required T-OP artifacts (fixed_budget_3600.pt, best.pt, last.pt, training_summary.json, history.json) cryptographically verified against HF LFS revision 4aa9f2a68f0709889eacacf170b6446d3c45c73d and Kaggle minkhoaa/langfens-rq1-psup-t-op-t4x2.",
    "Zero AMP skips, 3600 fixed/last updates, 2700 best updates, objective operation_ce, all 266,667 head parameters and 16 optimizer moment tensors finite.",
    "Root cause of embedded history mismatch (3150 vs 3600) identified in trainer batch accumulation loop checkpoint snapshot timing.",
    "Zero training/optimizer/backward steps executed; zero test partition audio/labels accessed."
  ],
  "residualRisks": [
    "Candidate evaluation metrics on Fold-1 VAL (300 utts, NJS/RRBI) must be computed via independent benchmark inference runners (under both protocol O and genuine Z); training-time selection metrics were on SELECT (MBMPS/TLV) and do not represent final VAL comparison.",
    "Kaggle kernel API metadata reports is_private=false, though staged dataset and Hugging Face checkpoint repository are confirmed private."
  ],
  "noStagedFiles": true,
  "diffSummary": "Created authoritative recheck report at managed path /home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-1.md without modifying any repository or raw artifact files.",
  "reviewFindings": [
    "no blockers: All required artifacts recovered, cryptographically validated, payload verified on CPU, accounting mismatch reconciled, and unit tests green."
  ],
  "manualNotes": "Agent A remained completely read-only to repository source, worktrees, and raw artifacts. Only the managed subagent output report was created."
}
```
