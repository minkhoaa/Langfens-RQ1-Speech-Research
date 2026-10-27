# LANGFENS — RQ1-TRN-02B Preflight-R2 Fail-Closed Safety Closure Report

**Document ID:** `RQ1-TRN-02B-PREFLIGHT-R2`  
**Status:** COMPLETE / PASS  
**Branch:** `fix/rq1-trn-02b-preflight-r2`  
**Base Branch:** `fix/rq1-trn-02b-preflight-r1`  
**Base Commit SHA:** `9fe993b6af0346a0d8170e328b5416cab30bdbdd`  
**Execution Scope:** Local Laptop (CPU Forward-Only). STRICTLY NO GPU training, backward passes, optimizer updates, Colab runtime allocations, or official test partition evaluations occurred.

---

## 1. Executive Summary and Acceptance Gate

### **Gate Result: RQ1-TRN-02B-PREFLIGHT-R2: PASS**

The bounded fail-closed safety patch has been fully applied to the training driver, evaluation harness, checkpoint schemas, and regression tests. All five target areas (Strict Resume Validation, Guardrail Artifact Preservation, Paired Endpoint Validation, Dual-Arm CTC Guardrail, and Bounded Regression Tests) are completed and verified on local CPU.

| Safety Patch Area | Description | Status | Primary Source Location | Verification Tests |
|---|---|:---:|---|---|
| **A. Strict Resume Validation** | Watertight validation of arm ID, seed, dropout, hashes, full optimizer/scheduler configs, counters, and states | **PASS** | `audit_scripts/train_rq1_trn_02b.py` lines 334–465 | `test_rq1_trn_02b_preflight_r2_safety.py` (Tests 1, 2, 3, 5) |
| **B. Guardrail Artifact Integrity** | `halted_guardrail.pt` written on violation; `last.pt` preserved intact; resume from halted artifact rejected | **PASS** | `audit_scripts/train_rq1_trn_02b.py` lines 860–905 | `test_rq1_trn_02b_preflight_r2_safety.py` (Tests 4, 5) |
| **C. Strengthened Endpoint Validation** | C0/C1 arm label verification, cross-arm compatibility check, fail-closed causal estimates under exploratory overrides | **PASS** | `audit_scripts/evaluate_rq1_trn_02b.py` lines 215–310, 715–860 | `test_rq1_trn_02b_preflight_r2_safety.py` (Tests 6, 7) |
| **D. Dual-Arm CTC Guardrail** | Both C0 and C1 evaluated against measured Reference R and historical anchor (23.64%); separate recording; violation marks experiment invalid | **PASS** | `audit_scripts/evaluate_rq1_trn_02b.py` lines 760–845 | `test_rq1_trn_02b_preflight_r2_safety.py` (Tests 8, 9) |
| **E. Bounded CPU Regression Tests** | 10 new targeted unit/integration tests covering all failure and boundary modes | **PASS** | `tests/test_rq1_trn_02b_preflight_r2_safety.py` | 10 passed in 2.44s |

---

## 2. Source Modifications and Technical Details

### 2.1 Strict Resume Checkpoint Validation (`audit_scripts/train_rq1_trn_02b.py`)
- **Required Top-Level Keys:** `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `successful_updates`, `step_attempts`, `skipped_updates`, `micro_steps`, `rng`, `dataloader_generator`, `metadata`. Any missing key triggers immediate fail-closed termination (`ValueError`).
- **Required Component States:** Verifies non-empty state dictionaries for `model`, `optimizer` (`state`, `param_groups`), `scheduler`, `scaler`, `rng` (`python`, `numpy`, `torch`), and ByteTensor generator state for `dataloader_generator`. If arm is C1, verifies presence of non-empty `intervention_rng`.
- **Halted Artifact Prohibition:** Unconditionally rejects checkpoints where `guardrail_violation == True` or `experiment_status` is in `{"HALTED_GUARDRAIL_VIOLATION", "HALTED", "CORRUPTED", "INVALID"}`.
- **Cross-Arm Rejection:** Strictly rejects resumption if checkpoint arm does not match current configuration arm (`ck_arm != cfg["arm"]`).
- **Configuration & Provenance Match:** Asserts exact matching of `seed`, `binary_prompt_dropout_p`, `split_sha256`, `initial_checkpoint_sha256`, `learning_rate`, `weight_decay`, `warmup_updates`, `max_successful_updates`, `gradient_accumulation_steps`, `max_epochs`, and `batch_size`.
- **Counter Consistency:** Verifies `step_attempts == successful_updates + skipped_updates` and `micro_steps >= step_attempts`.

### 2.2 Preservation of Resumable Checkpoint on Guardrail Violation (`audit_scripts/train_rq1_trn_02b.py`)
- **Artifact Isolation:** When greedy validation CTC PER exceeds the Reference R anchor (23.64%) by more than +1.0 percentage point ($>24.64\%$), the driver constructs a complete state dictionary (including all model weights, optimizer buffers, scheduler state, scaler scale, RNGs, dataloader state, and metadata) and saves it to:
  ```text
  checkpoints/.../halted_guardrail.pt
  ```
- **Preservation of `last.pt`:** The previous valid epoch checkpoint (`last.pt`) is **never overwritten** by the guardrail failure handler. It remains intact as the latest valid resumable checkpoint.
- **Endpoint Integrity:** `primary_endpoint_reached` and `primary_3600_updates_completed` are set strictly to `False`. `fixed_budget_checkpoint.pt` is not written.

### 2.3 Paired Endpoint and Cross-Arm Compatibility Validation (`audit_scripts/evaluate_rq1_trn_02b.py`)
- **Genuinely Labeled Arm Check:** `validate_endpoint_readiness` asserts `meta["arm"] == expected_arm`. Passing a C1 checkpoint where C0 is expected (or vice versa) raises `ValueError`.
- **Guardrail Checkpoint Rejection:** Rejects any checkpoint marked with `guardrail_violation == True` or `experiment_status == "HALTED_GUARDRAIL_VIOLATION"`.
- **Cross-Arm Compatibility (`validate_paired_checkpoints_compatibility`):**
  - Asserts identical `initial_checkpoint_sha256`.
  - Asserts identical `split_sha256`.
  - Asserts identical `seed`.
  - Asserts identical shared hyperparameter configuration (`learning_rate`, `weight_decay`, etc.).
  - Asserts Control C0 has $p=0.0$ and Treatment C1 has $p=0.25$.
- **Exploratory Override Contract (`--allow-incomplete-endpoints`):**
  - If incomplete checkpoints are loaded under `--allow-incomplete-endpoints`:
    - `experiment_status` is strictly `"EXPLORATORY_INCOMPLETE_BUDGET"` (never `"VALID"`).
    - `is_primary_causal_comparison` is strictly `False`.
    - `causal_comparison_authorized` is strictly `False`.
    - The estimand is labeled `"exploratory_unauthorized_delta_dropout (C1 - C0)"`.

### 2.4 Dual-Arm CTC PER Guardrail Enforcement (`audit_scripts/evaluate_rq1_trn_02b.py`)
- **Both Arms Evaluated:** Degradations are computed independently for Control C0 and Treatment C1 against the newly measured Reference R anchor and the historical anchor (23.64%):
  $$\Delta \text{PER}_{\text{C0}} = (\text{PER}_{\text{C0}} - \text{PER}_{\text{R}}) \times 100.0$$
  $$\Delta \text{PER}_{\text{C1}} = (\text{PER}_{\text{C1}} - \text{PER}_{\text{R}}) \times 100.0$$
- **Dual Guardrail Rule:**
  - `c0_guardrail_pass = c0_deg <= 1.0`
  - `c1_guardrail_pass = c1_deg <= 1.0`
  - `overall_guardrail_pass = c0_guardrail_pass and c1_guardrail_pass`
- **Fail-Closed Reporting:** If either C0 or C1 violates the guardrail, `experiment_status` is marked `"INVALID_GUARDRAIL_FAILURE"`. Under `--strict-guardrails` (default), the evaluator raises `RuntimeError` citing the specific violating arm(s).

---

## 3. Checkpoint Schema and State Transitions

### 3.1 Checkpoint Schema Contract (Version 1.0.0)
```json
{
  "model": "<state_dict>",
  "optimizer": "<state_dict>",
  "scheduler": "<state_dict>",
  "scaler": "<state_dict>",
  "epoch": "<int >= 0>",
  "display_epoch": "<int >= 1>",
  "successful_updates": "<int >= 0>",
  "step_attempts": "<int == successful_updates + skipped_updates>",
  "skipped_updates": "<int >= 0>",
  "micro_steps": "<int >= step_attempts>",
  "best_ap": "<float>",
  "patience": "<int >= 0>",
  "skipped_amp_attempts_total": "<int>",
  "consecutive_skips": "<int >= 0>",
  "history": "<list of epoch dicts, len == epoch + 1>",
  "rng": {
    "python": "<tuple>",
    "numpy": "<tuple>",
    "torch": "<ByteTensor>"
  },
  "intervention_rng": "<Optional[ByteTensor]>",
  "dataloader_generator": "<ByteTensor>",
  "metadata": {
    "schema_version": "1.0.0",
    "checkpoint_schema_version": "1.0.0",
    "arm": "C0 | C1",
    "experiment": "RQ1-TRN-02B-C0 | RQ1-TRN-02B-C1",
    "binary_prompt_dropout_p": "0.0 | 0.25",
    "git_sha": "<40-char commit SHA>",
    "split_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
    "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
    "config": "<full YAML configuration dict>",
    "primary_endpoint_reached": "<bool>",
    "primary_3600_updates_completed": "<bool>",
    "epoch_cap_reached": "<bool>",
    "guardrail_violation": "<bool>",
    "experiment_status": "IN_PROGRESS | COMPLETED | HALTED_GUARDRAIL_VIOLATION",
    "endpoint_status": "COMPLETED_3600_UPDATES | INCOMPLETE_AMP_SKIPS | HALTED_EPOCH_CAP | HALTED_GUARDRAIL_VIOLATION"
  }
}
```

### 3.2 State Transition Matrix

```
[Start Training]
       │
       ▼
 [IN_PROGRESS] ────(CTC PER > Anchor + 1.0%)────► [HALTED_GUARDRAIL_VIOLATION]
       │                                                   │
       │                                                   ├─► Writes halted_guardrail.pt
       │                                                   ├─► Preserves existing last.pt
       │                                                   └─► Terminates execution (Resume PROHIBITED)
       ▼
(Epoch 6 / 3600 Steps)
       │
       ├─► (successful_updates >= 3600) ──────► [COMPLETED_3600_UPDATES]
       │                                               │
       │                                               ├─► Writes fixed_budget_checkpoint.pt
       │                                               └─► Eligible for Primary Causal Comparison
       │
       └─► (successful_updates < 3600) ───────► [INCOMPLETE_AMP_SKIPS]
                                                       │
                                                       ├─► No fixed_budget_checkpoint.pt
                                                       └─► Evaluator FAILS CLOSED (Exploratory override only)
```

---

## 4. Test Census and Regression Verification

### 4.1 Preflight-R2 Safety Test Suite (`tests/test_rq1_trn_02b_preflight_r2_safety.py`)
All 10 tests passed on local CPU:

```text
$ PYTHONPATH=. .venv/bin/pytest tests/test_rq1_trn_02b_preflight_r2_safety.py -v
============================= test session starts ==============================
collected 10 items

tests/test_rq1_trn_02b_preflight_r2_safety.py::test_1_missing_checkpoint_fields_rejection PASSED [ 10%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_2_cross_arm_checkpoint_resume_rejection PASSED [ 20%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_3_malformed_contradictory_update_counters PASSED [ 30%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_4_guardrail_failure_artifact_schema_and_last_pt_preservation PASSED [ 40%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_5_rejection_of_halted_checkpoints_during_resume PASSED [ 50%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_6_incorrect_endpoint_arm_and_incomplete_update_count PASSED [ 60%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_7_exploratory_override_without_false_valid_reporting PASSED [ 70%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_8_c0_only_guardrail_failure PASSED [ 80%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_9_c1_only_guardrail_failure PASSED [ 90%]
tests/test_rq1_trn_02b_preflight_r2_safety.py::test_10_frozen_split_and_official_test_access_boundaries PASSED [100%]

============================== 10 passed in 2.44s ==============================
```

### 4.2 Full Repository Test Suite
```text
$ PYTHONPATH=. .venv/bin/pytest tests/
============================= test session starts ==============================
collected 160 items

tests/... [156 passed, 4 skipped in 110.38s]
```
- **Passed Tests:** 156
- **Skipped Tests:** 4 (strictly GPU-only / CUDA backward smoke tests per AGENTS.md)
- **Failed Tests:** 0

---

## 5. Provenance and Budget Verification

| Artifact Role | File Path | Expected SHA-256 Digest | Verification Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **VERIFIED MATCH** |
| **Initial Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **VERIFIED MATCH** |
| **CMUdict Dictionary** | `resources/CMUDict/cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | **VERIFIED MATCH** |
| **Authorized Audio Zip** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | 727 MB archive (Fold 1 Train & Val only) | **SEALED TEST VERIFIED** |

---

## 6. Remaining Blockers and Execution Readiness

- **Zero Operational Blockers:** All checkpoint validation, guardrail failure persistence, paired endpoint verification, and regression tests are complete and passing.
- **Safety Boundary Preserved:** No Colab instance was provisioned, no GPU was allocated, no backward pass was run, and no official test data was touched.

### **FINAL GATE: RQ1-TRN-02B-PREFLIGHT-R2: PASS**

---

## 7. Git Provenance and Verification

- **Branch:** `fix/rq1-trn-02b-preflight-r2`
- **Parent Base SHA:** `9fe993b6af0346a0d8170e328b5416cab30bdbdd` (branch `fix/rq1-trn-02b-preflight-r1`)
- **Remote Upstream:** `origin/fix/rq1-trn-02b-preflight-r2`
- **Execution Boundary Preserved:** Zero GPU allocations, backward passes, optimizer updates, Colab runtime launches, or official test partition evaluations occurred.

