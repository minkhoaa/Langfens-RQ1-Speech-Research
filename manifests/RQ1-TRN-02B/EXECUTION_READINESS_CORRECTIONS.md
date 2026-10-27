# LANGFENS — RQ1-TRN-02B Execution-Readiness Corrections Report

**Document ID:** `RQ1-TRN-02B-PREFLIGHT-R1`  
**Status:** COMPLETE / PASS  
**Branch:** `fix/rq1-trn-02b-preflight-r1`  
**Base Branch:** `feat/rq1-trn-02b-implementation-preflight`  
**Base Commit SHA:** `2f6345e426279dd541cc830d4e31dd2d7823a7e7`  
**Execution Scope:** Local Laptop (CPU Forward-Only). STRICTLY NO GPU training, backward passes, optimizer updates, Colab runtime allocations, or official test partition evaluations occurred.

---

## 1. Executive Summary and Acceptance Gate

### **Gate Result: RQ1-TRN-02B-PREFLIGHT-R1: PASS**

All six execution-readiness defects identified in the RQ1-TRN-02B preflight audit have been corrected in code, static-inspected, and verified with 10 new dedicated integration tests and a full-suite regression census of 146 tests (0 failures).

| Defect Identifier | Description | Resolution Status | Verification Test |
|---|---|:---:|---|
| **Defect 1** | Evaluation dataset field access mismatch (`waveform`, `target_ctc`, `speaker`) | **RESOLVED** | `test_1_dataset_field_access_contract`, `test_3_protocol_o_evaluation_pipeline` |
| **Defect 2** | Missing Protocol Z zero-oracle G2P evaluation path | **RESOLVED** | `test_4_protocol_z_zero_oracle_scoring` |
| **Defect 3** | Non-deterministic training resumption (missing DataLoader generator state & validation) | **RESOLVED** | `test_5_resume_sample_order_determinism_without_optimizer_steps`, `test_9_checkpoint_provenance_validation_on_resume` |
| **Defect 4** | Incomplete budget mislabeled as completed 3,600-update endpoint | **RESOLVED** | `test_6_amp_skipped_update_simulations_and_budget_tracker`, `test_7_fail_closed_endpoint_handling` |
| **Defect 5** | Unenforced CTC PER guardrail (+1.0 percentage point vs Reference R anchor) | **RESOLVED** | `test_8_ctc_per_guardrail_enforcement` |
| **Defect 6** | Erroneous CMUdict SHA-256 checksum in preflight report | **RESOLVED** | Verified against `resources/CMUDict/cmudict.dict` digest |

---

## 2. Detailed Technical Resolutions

### 2.1 Defect 1: Evaluation Dataset Interface Repair

- **Root Cause:** In `audit_scripts/evaluate_rq1_trn_02b.py`, the evaluation loop accessed deprecated or non-existent keys: `item["waveform"]`, `item["target_ctc"]`, and `item["speaker"]`.
- **Contract Correction:** Replaced all occurrences with the verified contract defined by `L2ArcticMDDDataset` in `src/langfens_prep/dataset.py`:
  - `item["waveform"]` $\rightarrow$ `item["audio_waveform_16k"]`
  - `donor_item["waveform"]` $\rightarrow$ `donor_item["audio_waveform_16k"]`
  - `item["target_ctc"]` $\rightarrow$ `item["ctc_target_indices"]`
  - `item["speaker"]` $\rightarrow$ `item["speaker_id"]`
- **Speaker Boundary Extraction:** Replaced inefficient TextGrid/audio reloading loop with index-based transition boundaries:
  ```python
  spk_bounds = [
      i for i in range(1, len(val_dataset))
      if (val_dataset.utterance_entries[i][0] if hasattr(val_dataset, "utterance_entries") else val_dataset[i]["speaker_id"]) !=
         (val_dataset.utterance_entries[i - 1][0] if hasattr(val_dataset, "utterance_entries") else val_dataset[i - 1]["speaker_id"])
  ]
  ```
- **Evidence:** `tests/test_rq1_trn_02b_corrections_r1.py::test_1_dataset_field_access_contract` and `test_3_protocol_o_evaluation_pipeline` execute CPU forward passes using actual authorized validation samples from `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` with zero key errors.

### 2.2 Defect 2: Protocol Z Zero-Oracle Evaluation Integration

- **Implementation:** Integrated the existing zero-oracle evaluation path into `audit_scripts/evaluate_rq1_trn_02b.py` via `evaluate_models_protocol_z(...)`.
- **Frontend & Dataset:** Consumes `CMUDictG2P(verify_checksum=True, oov_policy="strict")` and `ZeroOracleL2ArcticDataset` loaded from `resources/L2-ARCTIC/prompts.json`.
- **Strict Label Isolation:**
  - Evaluator asserts `z_item["binary_labels"] is None` and `z_item["ctc_target_indices"] is None` on every inference input.
  - Model forward pass receives strictly 16 kHz audio waveform and prompt-derived G2P canonical phone indices.
  - Gold annotations are loaded strictly offline via an independent `L2ArcticMDDDataset` instance and used only during DP Levenshtein alignment scoring.
- **Reporting Metrics:**
  - Aligned-only AP for Reference R, Control C0, and Treatment C1.
  - Token coverage ($\ge 99.63\%$).
  - Paired C1-minus-C0 contrast ($\Delta_{\text{dropout\_z}}$), C0-minus-R drift ($\Delta_{\text{tuning\_z}}$), and C1-minus-R compound contrast ($\Delta_{\text{compound\_z}}$).
- **Evidence:** `tests/test_rq1_trn_02b_corrections_r1.py::test_4_protocol_z_zero_oracle_scoring` verifies frontend checksum, zero label leakage, dynamic alignment, and paired differences.

### 2.3 Defect 3: Deterministic Training Resumption

- **DataLoader Generator State Persistence:**
  - Initialized `train_dl_generator = torch.Generator().manual_seed(cfg["seed"])`.
  - Added `"dataloader_generator": train_dl_generator.get_state()` to the saved checkpoint dictionary.
  - On resumption (`--resume`), restored generator state:
    ```python
    if "dataloader_generator" in ck and ck["dataloader_generator"] is not None:
        train_dl_generator.set_state(ck["dataloader_generator"])
    ```
- **Provenance & Configuration Validation on Resumption:**
  - Implemented `validate_checkpoint_for_resume(...)` in `audit_scripts/train_rq1_trn_02b.py`.
  - Validates:
    1. Arm ID match (`ck_arm == cfg["arm"]`).
    2. Random seed match (`ck_seed == cfg["seed"]`).
    3. Manifest checksum match (`ck_manifest_sha == manifest_sha`).
    4. Initial checkpoint SHA-256 match (`ck_init_sha == init_ckpt_sha`).
    5. Dropout probability match (`ck_p == binary_prompt_dropout_p`).
    6. Non-negative and consistent update counters:
       - `successful_updates >= 0`
       - `step_attempts >= successful_updates + skipped_updates`
       - `micro_steps >= step_attempts`
       - `epoch >= 0`
- **Sample-Order Determinism Verification:**
  - Tested uninterrupted Epoch 1 $\rightarrow$ Epoch 2 sequence vs resumed Epoch 2 sequence (loading generator state from Epoch 1 boundary).
  - Confirmed 100% bitwise sample-order equivalence without executing any optimizer steps or backward passes.
- **Evidence:** `tests/test_rq1_trn_02b_corrections_r1.py::test_5_resume_sample_order_determinism_without_optimizer_steps` and `test_9_checkpoint_provenance_validation_on_resume`.

### 2.4 Defect 4: Fixed-Budget Endpoint Handling and Fail-Closed Enforcement

- **Independent Caps Enforcement:**
  - Max Epochs: 6.
  - Max Successful Updates: 3,600.
  - Training halts when either limit is reached.
- **Completed Endpoint Classification:**
  - Checkpoint is marked as a completed endpoint **if and only if** `successful_updates >= 3600`.
  - If 6 epochs complete with fewer than 3,600 successful updates due to AMP gradient unscaling skips:
    - `primary_endpoint_reached = False`.
    - `endpoint_status = "INCOMPLETE_AMP_SKIPS"`.
    - `fixed_budget_checkpoint.pt` is **not** written or labeled as a completed endpoint.
- **Evaluation Fail-Closed Enforcement:**
  - Added `validate_endpoint_readiness(...)` to `audit_scripts/evaluate_rq1_trn_02b.py`.
  - If either Control C0 or Treatment C1 checkpoint has `successful_updates < 3600` or `primary_endpoint_reached == False`, the evaluator raises `RuntimeError` and terminates:
    ```text
    FAIL-CLOSED: Checkpoint ... for arm C1 is NOT a completed 3,600-successful-update endpoint!
    Causal comparison prohibited because training budget is incomplete due to AMP skips or early termination.
    ```
  - An exploratory override flag (`--allow-incomplete-endpoints`) is provided for preliminary diagnostic inspections.
- **Evidence:** `tests/test_rq1_trn_02b_corrections_r1.py::test_6_amp_skipped_update_simulations_and_budget_tracker` and `test_7_fail_closed_endpoint_handling`.

### 2.5 Defect 5: CTC PER Regression Guardrail Enforcement

- **Safety Contract:** Validation greedy CTC PER must not degrade by more than **+1.0 absolute percentage point** relative to the Reference R historical anchor (23.64%).
- **Training Stop Condition:**
  - During training in `audit_scripts/train_rq1_trn_02b.py`, if `val_metrics["validation_ctc_per"] > 0.2464` (+1.0% vs 0.2364):
    - Logs `[GUARDRAIL VIOLATION]`.
    - Sets metadata `guardrail_violation = True` and `experiment_status = "HALTED_GUARDRAIL_VIOLATION"`.
    - Saves checkpoint and halts immediately via `RuntimeError`.
- **Evaluation Enforcement:**
  - In `audit_scripts/evaluate_rq1_trn_02b.py`, checks degradation:
    $$\Delta \text{PER} = (\text{PER}_{\text{C1}} - \text{PER}_{\text{R}}) \times 100.0$$
  - If $\Delta \text{PER} > 1.0\%$:
    - Sets `experiment_status = "INVALID_GUARDRAIL_FAILURE"` and `ctc_per_guardrail_pass = False`.
    - Writes evaluation report with full evidence to disk.
    - If `--strict-guardrails` (default: True), raises `RuntimeError` declaring the experiment invalid.
- **Evidence:** `tests/test_rq1_trn_02b_corrections_r1.py::test_8_ctc_per_guardrail_enforcement`.

### 2.6 Defect 6: CMUdict SHA-256 Checksum Correction

- **Correction:** In `manifests/RQ1-TRN-02B/IMPLEMENTATION_PREFLIGHT_REPORT.md` (line 181/200), replaced erroneous checksum `8191784338d38eb5ea26bb02cfb6a4a09a5b3a62024b457e5e34be64fa4cfd0e` with the cryptographically verified digest of `resources/CMUDict/cmudict.dict`:
  ```text
  81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22
  ```
- **Evidence:** Direct SHA-256 computation over `resources/CMUDict/cmudict.dict` (3,618,488 bytes) matches `EXPECTED_CMUDICT_SHA256` in `src/langfens_prep/g2p.py` and `manifests/RESOURCE_INVENTORY.md`.

---

## 3. Test Census and Regression Verification

### 3.1 New Preflight Corrections Regression Suite (`test_rq1_trn_02b_corrections_r1.py`)

All 10 tests passed on CPU:

```text
tests/test_rq1_trn_02b_corrections_r1.py::test_1_dataset_field_access_contract PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_2_speaker_safe_audio_permutation PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_3_protocol_o_evaluation_pipeline PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_4_protocol_z_zero_oracle_scoring PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_5_resume_sample_order_determinism_without_optimizer_steps PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_6_amp_skipped_update_simulations_and_budget_tracker PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_7_fail_closed_endpoint_handling PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_8_ctc_per_guardrail_enforcement PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_9_checkpoint_provenance_validation_on_resume PASSED
tests/test_rq1_trn_02b_corrections_r1.py::test_10_no_official_test_access PASSED
```

### 3.2 Full Repository Test Suite

Command executed:
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

Results: **146 PASSED, 4 SKIPPED, 0 FAILED** (Duration: 103.99s)
- **146 Passed Tests:** All historical audio loaders, TextGrid parsers, dataset builders, phoneme normalizers, loss modules, model architectures, CTC stabilization tests, Protocol Z alignment tests, and preflight tests.
- **4 Skipped Tests:** Strictly GPU-only smoke tests and backward pass placeholders, skipped in accordance with AGENTS.md execution boundaries.

---

## 4. CPU Forward-Only CLI Verification

Both training and evaluation CLI entrypoints were verified in dry-run mode on local CPU:

### 4.1 Training Driver Dry-Run (Control Arm C0)
```text
$ python3 audit_scripts/train_rq1_trn_02b.py --config configs/rq1_trn_02b_c0.yaml --dry-run
================================================================================
LANGFENS — RQ1-TRN-02B Training Entrypoint
Arm: C0 | Binary Prompt Dropout p: 0.00
Manifest SHA-256: 7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b
Initial Checkpoint SHA-256: a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b
================================================================================
[MODEL] Initialized model weights from checkpoints/RQ1-TRN-02A/recovered/best.pt (epoch 5)
[MODEL] Intervention generator seeded with 1337 (p=0.0)
[TRAINING] Cold reset of AdamW optimizer, cosine scheduler (100 warmup, 3600 updates), and AMP GradScaler.
[DRY-RUN] Preflight dry-run completed successfully. No training executed.
```

### 4.2 Training Driver Dry-Run (Treatment Arm C1)
```text
$ python3 audit_scripts/train_rq1_trn_02b.py --config configs/rq1_trn_02b_c1.yaml --dry-run
================================================================================
LANGFENS — RQ1-TRN-02B Training Entrypoint
Arm: C1 | Binary Prompt Dropout p: 0.25
Manifest SHA-256: 7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b
Initial Checkpoint SHA-256: a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b
================================================================================
[MODEL] Initialized model weights from checkpoints/RQ1-TRN-02A/recovered/best.pt (epoch 5)
[MODEL] Intervention generator seeded with 1337 (p=0.25)
[TRAINING] Cold reset of AdamW optimizer, cosine scheduler (100 warmup, 3600 updates), and AMP GradScaler.
[DRY-RUN] Preflight dry-run completed successfully. No training executed.
```

### 4.3 Paired Evaluation Driver Dry-Run
```text
$ python3 audit_scripts/evaluate_rq1_trn_02b.py \
    --ref-checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
    --c0-checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
    --c1-checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
    --dry-run
================================================================================
LANGFENS — RQ1-TRN-02B Paired Evaluation Harness
Manifest SHA-256: 7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b
Reference R Checkpoint SHA-256: a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b
Control C0 Checkpoint: checkpoints/RQ1-TRN-02A/recovered/best.pt
Treatment C1 Checkpoint: checkpoints/RQ1-TRN-02A/recovered/best.pt
================================================================================
[DRY-RUN] Verified checkpoint and manifest hashes successfully. No inference executed.
```

---

## 5. Explicit Scientific and Operational Limitations

1. **Validation Selection Bias:** Both Control C0 and Treatment C1 initialize from `best.pt`, which was historically selected on the Fold 1 validation partition.
2. **Single-Seed Limitation:** Initial authorization executes a single seed (`1337`). Projection head variance across multiple seeds is deferred to follow-on stages.
3. **Limited Speaker Diversity:** Validation encompasses exactly two non-native speakers (`NJS`, `RRBI`). Utterance-cluster bootstrap intervals quantify utterance sampling uncertainty within these two speakers, but do not establish cross-speaker population generalizability.
4. **Official Test Partition Sealed:** Official test data (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) remains completely sealed and unstaged.
5. **GPU Kernel Non-Determinism:** CTC loss backward pass and CUDA atomic scatter/gather operations possess inherent kernel-level non-determinisms that cannot be completely eliminated on GPU, even with seeded generators.

---

## 6. Execution Readiness Decision

### **FINAL GATE: RQ1-TRN-02B-PREFLIGHT-R1: PASS**

The codebase is fully repaired, static-inspected, and regression-verified. All technical blockers to execution readiness have been resolved with source code modifications and integration tests.
