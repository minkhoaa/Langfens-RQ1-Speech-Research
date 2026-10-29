# LANGFENS — RQ1-TRN-03A / 04B-Kaggle Durability & Recovery Closure

**Status:** `COMPLETED`

**Recorded at:** 2026-10-29T19:30:00+07:00

**Evidence source:** Commits `2200050fe45eb081056873cc0c3ae4eefcfa8164`, `audit_scripts/train_fold1_ctc_binary.py`, `audit_scripts/kaggle_fold_train_exec.py`, `audit_scripts/kaggle_checkpoint_sync.py`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Executable Mocked Recovery, Full Test Suite & Durability Verification)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Close all remaining correctness, durability, and recovery gaps before authorizing Kaggle GPU execution for Fold 2 Baseline R training. Guarantee that `fixed_budget_checkpoint.pt` represents strictly and immutably 3,600 successful updates at Epoch 6 (including after interruption and resume); enforce rigorous endpoint validation; implement comprehensive resume metadata validation rejecting cross-fold checkpoint reuse; ensure safe recovery from interrupted Epoch 6 uploads; require verified private remote persistence for production Kaggle execution; enforce per-epoch `history.json` upload with SHA-256 readback; remove credential-file fallbacks and CLI token arguments in favor of environment variables / Kaggle Secrets; author executable mocked recovery tests; test test-speaker leak detection independently of SHA mismatch; and verify preflight readiness.

---

## 2. Inputs & Dependencies
- **Base Commit:** `2200050fe45eb081056873cc0c3ae4eefcfa8164` on branch `review/rq1-trn-03a-cross-fold-planning`.
- **Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py`.
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`.
- **Remote Mirror Module:** `audit_scripts/kaggle_checkpoint_sync.py`.
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py`.
- **Master Policy:** `AGENTS.md`.

---

## 3. Implemented Durability & Correctness Remediations

### 3.1 Fixed-Budget Checkpoint Invariance & Epoch 6 Lock
- **Requirement:** `fixed_budget_checkpoint.pt` must represent strictly and immutably **EXACTLY 3,600 successful updates at Epoch 6** (`epoch == 5`, 0-indexed).
- **Implementation in `audit_scripts/train_fold1_ctc_binary.py`:**
  - Added parameter `is_fixed_budget_endpoint: bool = False` to `checkpoint()`.
  - Enforced fail-closed assertion inside `checkpoint()`: if `is_fixed_budget_endpoint` is True, it asserts `updates == 3600 and epoch == 5`. Any attempt to write a fixed-budget checkpoint with different updates or epochs raises `ValueError`.
  - In the training loop, saving `fixed_budget_checkpoint.pt` is conditioned strictly on `epoch == 5 and updates == 3600 and not fixed_budget_saved`.
  - Once saved, `fixed_budget_saved` is set to `True`, locking the artifact against any subsequent overwrite by later training epochs (e.g. Epochs 7–11).

### 3.2 Comprehensive Resume Metadata & Cross-Fold Isolation Validation
- Implemented `validate_resume_checkpoint()` in `audit_scripts/train_fold1_ctc_binary.py`:
  - **Required Keys:** Verifies `"model"`, `"optimizer"`, `"scheduler"`, `"scaler"`, `"epoch"`, `"successful_updates"`, `"micro_steps"`, `"best_ap"`, `"history"`, `"rng"`, `"metadata"`.
  - **Non-Empty States:** Verifies state dicts are non-empty dictionaries.
  - **Fold Identity Check:** Extracts checkpoint fold identity from metadata or config. If `ck_fold != expected_fold`, raises `ValueError("FAIL-CLOSED: Cross-fold resume rejected: checkpoint is from fold {ck_fold}, expected fold {expected_fold}")`. Cross-fold checkpoint reuse is strictly impossible.
  - **Provenance Invariants:** Asserts `seed == 1337`, `pretrained_model == "facebook/wav2vec2-base-960h"`, and `split_sha256 == manifest_sha`.
  - **Epoch-to-History Consistency:** Asserts `len(history) == epoch + 1`.

### 3.3 Safe Recovery from Interrupted Epoch 6 Upload
- In `audit_scripts/train_fold1_ctc_binary.py`, on resumption:
  1. If `fixed_budget_checkpoint.pt` exists locally:
     Loads and verifies that `successful_updates == 3600 and epoch == 5`. Sets `fixed_budget_saved = True` and locks it against overwrite.
  2. If resuming at or past Epoch 6 (`epoch >= 5` or `updates >= 3600`) and local `fixed_budget_checkpoint.pt` is missing:
     If `remote_mirror` is configured, automatically downloads `fixed_budget_checkpoint.pt` via `remote_mirror.download_file()`, loads and verifies its 3,600 updates, and locks it.
     If remote recovery fails or remote mirror is absent, fails closed with `RuntimeError` rather than continuing in an unverified state.

### 3.4 Production Remote Persistence & Credential Hygiene
- **Strict Private Persistence Required:** In `audit_scripts/kaggle_fold_train_exec.py`, non-dry-run executions strictly require `remote_mirror_backend == "huggingface"` and explicit target repository (`minkhoaa/langfens-rq1-cross-fold-checkpoints`). Local-only persistence is rejected.
- **Mandatory Verification Probe:** Performs live authenticated write-readback-delete probe to Hugging Face before training starts.
- **Credential Hygiene:**
  - Implemented `resolve_cross_fold_hf_token()` in `audit_scripts/kaggle_checkpoint_sync.py`, reading strictly from environment variables (`HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`) or Kaggle Secrets (`UserSecretsClient().get_secret("HF_TOKEN")`).
  - Removed all fallback searches on candidate token files in datasets or repository trees.
  - Removed `--remote-mirror-token` CLI argument from `kaggle_fold_train_exec.py` and `deploy/kaggle/kaggle_entrypoint.py` to prevent token leakage in process listings (`ps aux`) and shell history.
- **Per-Epoch History Persistence:** Inside the training loop, `history.json` is serialized locally and mirrored to remote persistent storage with readback SHA-256 verification after **every single epoch**.

### 3.5 Rigorous Endpoint Classification (`classify_fold_training_endpoint`)
- Enforces strict completion invariants:
  1. `fixed_budget_checkpoint.pt` exists and has `successful_updates == 3600` (exact!).
  2. `epoch == 5` (exact!).
  3. Correct fold identity: `metadata["fold"] == fold_idx`.
  4. Immutable manifest SHA-256 match.
  5. Complete pinned hyperparameter contract verified against checkpoint metadata config.
  6. `best.pt`, `last.pt`, and `history.json` (with $\ge 6$ completed epochs) all present and valid.

### 3.6 Isolated Speaker-Leak Detection & Failure Path Coverage
- Implemented `audit_archive_for_test_speaker_leak(zip_path, fold_idx)` in `audit_scripts/kaggle_fold_train_exec.py` to inspect zip member entries independently of file hashing.
- Covered all failure paths in `tests/test_rq1_trn_03a_kaggle_deployment.py`:
  - Test speaker leakage caught independently of SHA mismatch (raises `PermissionError`).
  - Corrupt / non-zip archive caught (raises `ValueError`).
  - Cross-fold resume caught (raises `ValueError`).
  - Missing token in production Kaggle mode caught (raises `RuntimeError`).
  - Incomplete / corrupted endpoint classification caught (status `INCOMPLETE` or `FAILED`).

### 3.7 Executable Mocked Checkpoint Recovery Tests
- Replaced superficial string-check tests with `test_executable_mocked_checkpoint_recovery()`:
  - Saves a populated model, optimizer, scheduler, scaler, RNG, and metadata.
  - Loads into a newly initialized model with distinct initial weights.
  - Asserts bit-exact weight restoration (`torch.allclose`), optimizer state restoration, scheduler epoch restoration, scaler state restoration, and counter restoration.
- Added `test_epoch_6_fixed_budget_checkpoint_recovery_and_lock()`:
  - Proves that saving a fixed-budget checkpoint with `updates != 3600` or `epoch != 5` raises `ValueError`.

---

## 4. Results & Verification
- **Automated Regression Suite:** All 261 tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - 261 passed, 4 skipped in 140.23s (100% green).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` passes 11/11 tests in 14.02s:
  - `test_executable_mocked_checkpoint_recovery`: PASS
  - `test_epoch_6_fixed_budget_checkpoint_recovery_and_lock`: PASS
  - `test_fresh_pretrained_initialization_strictly_enforced`: PASS
  - `test_find_and_stage_fold_package_success`: PASS
  - `test_actual_test_speaker_leak_detection_independent_of_sha`: PASS
  - `test_corrupt_archive_detection`: PASS
  - `test_wrong_fold_resume_rejected`: PASS
  - `test_remote_mirror_missing_token_production_failure`: PASS
  - `test_endpoint_classification`: PASS
  - `test_kaggle_fold_train_exec_dry_run_all_folds`: PASS
  - `test_kaggle_entrypoint_mode_dispatch`: PASS
- **Preflight Dry-Run:** Verified for Folds 2, 3, and 4 with zero errors.

---

## 5. Output Artifacts
- **Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py` (validated resume, locked 3,600-update fixed budget, per-epoch history sync).
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py` (strict secret tokens, independent speaker leak audit, strict endpoint invariants).
- **Remote Mirror Module:** `audit_scripts/kaggle_checkpoint_sync.py` (strict secrets resolver, `download_file`, `file_exists`).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` (11 comprehensive tests).
- **Step Checkpoint:** `manifests/RQ1-TRN-03A/checkpoints/04b-kaggle-durability-and-recovery-closure.md`.

---

## 6. Restart / Recovery Instructions
- Review `manifests/RQ1-TRN-03A/checkpoints/04b-kaggle-durability-and-recovery-closure.md`.
- All technical and durability blockers are resolved.
- Remote Kaggle execution remains stopped awaiting operator approval.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:** No local training, no GPU training, no unsealing of test partitions, no remote uploads without explicit operator sign-off.
- **Pending Approvals:** Operator authorization to stage datasets and execute Fold 2 Baseline R training.
