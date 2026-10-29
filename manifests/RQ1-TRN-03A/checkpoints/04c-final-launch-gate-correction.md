# LANGFENS — RQ1-TRN-03A / 04C-Final Launch Gate Correction

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-29T21:00:00+07:00

**Evidence source:** Commits `edb4e4971b97404a0d48fd1d59e753560fd46a45` (base) -> `c9d2060110b6c429fb2d0d8973755fc86149156c` (implementation), `audit_scripts/train_fold1_ctc_binary.py`, `audit_scripts/kaggle_fold_train_exec.py`, `deploy/kaggle/kaggle_entrypoint.py`, `configs/rq1_trn_03a_fold{2,3,4}.yaml`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Executable Mocked Recovery, Full Test Suite & Durability Verification)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Execute final launch-gate corrections before authorizing Kaggle GPU execution for Fold 2 Baseline R training.
1. Independently verify the real Git HEAD and correct the invalid commit SHA (`2200050`) recorded in Step 04B's Kaggle launch instructions.
2. Strengthen resume validation to fail-closed on missing metadata, mismatched fold identity, altered hyperparameters, missing manifest digest, unverified git SHA, forbidden checkpoint ancestry, counter inconsistencies, or incomplete RNG generator states.
3. Decouple the cosine learning rate scheduler horizon (25 epochs, ~15,000 updates) from the Kaggle execution cap (10 completed epochs, max 6,000 updates), document the difference from historical Fold 1 training, enforce the cap across resumed runs, and retain the 3,600-update / Epoch 6 primary endpoint.
4. Remove the CLI token argument path for all cross-fold entrypoint execution modes while keeping legacy C0/C1 compatibility strictly isolated.
5. Ensure the child trainer strictly uses the environment / Kaggle Secrets credential path (`strict_secrets_only=True`) and verified remote persistent mirroring.
6. Verify Kaggle dataset mounting and automatic ZIP extraction handling, enabling bit-exact outer archive reconstruction and preflight verification without launching training.
7. Author comprehensive automated tests for all failure and recovery paths.
8. Maintain Fold 1 scientific evidence, frozen splits, sealed TEST partitions, and Baseline R architecture unchanged.

---

## 2. Inputs & Dependencies
- **Base Commit:** `edb4e4971b97404a0d48fd1d59e753560fd46a45` on branch `review/rq1-trn-03a-cross-fold-planning`.
- **Implementation Commit:** `c9d2060110b6c429fb2d0d8973755fc86149156c`.
- **Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py`.
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`.
- **Kaggle Entrypoint:** `deploy/kaggle/kaggle_entrypoint.py`.
- **Configurations:** `configs/rq1_trn_03a_fold{2,3,4}.yaml`.
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py`.
- **Master Policy:** `AGENTS.md`.

---

## 3. Implemented Launch-Gate Corrections

### 3.1 Correction of Invalid Commit SHA in Launch Instructions
- **Defect Identified:** Step 04B documentation in `docs/process-checkpoints/INDEX.md` and initial notes referenced parent commit `2200050fe45eb081056873cc0c3ae4eefcfa8164` (Step 04) rather than the actual verified commit closing Step 04B (`edb4e4971b97404a0d48fd1d59e753560fd46a45`).
- **Correction:** Independently verified real Git HEAD via `git rev-parse HEAD`. Updated Step 04B checkpoint, `INDEX.md`, and authoritatively pinned the exact Git HEAD for Kaggle execution launch instructions.

### 3.2 Strengthened Resume Metadata, Counter & Provenance Validation
- Overhauled `validate_resume_checkpoint()` in `audit_scripts/train_fold1_ctc_binary.py` with multi-tier fail-closed assertions:
  1. **Top-Level Structure:** Requires `"model"`, `"optimizer"`, `"scheduler"`, `"scaler"`, `"epoch"`, `"successful_updates"`, `"micro_steps"`, `"history"`, `"rng"`, `"metadata"`. State dicts must be non-empty dictionaries.
  2. **Metadata & Provenance:**
     - **Fold Identity:** Explicitly requires `meta["fold"] == expected_fold`. Missing fold or cross-fold mismatch raises `ValueError`.
     - **Manifest Digest:** Explicitly requires `meta["split_sha256"] == manifest_sha`. Missing or mismatched digest raises `ValueError`.
     - **Git Commit SHA:** Explicitly requires `meta["git_sha"]` to be present and non-empty.
     - **Checkpoint Ancestry:** Rejects any checkpoint where `meta` or `config` contains forbidden transfer keys (`initial_checkpoint`, `baseline_checkpoint`, `finetune_from`).
     - **Pinned Hyperparameters:** Enforces bit-level match against config for `seed` (1337), `batch_size` (2), `gradient_accumulation_steps` (2), `warmup_updates` (100), `learning_rate` (0.001), `weight_decay` (0.01), `loss_weights` ({"ctc": 1.0, "binary": 1.0}), `backbone_frozen` (True), `amp` ("fp16"), `early_stopping_patience` (5), `grad_scaler_initial_scale` (4096), `pretrained_model` ("facebook/wav2vec2-base-960h"), and `pretrained_revision` ("22aad52d435eb6dbaf354bdad9b0da84ce7d6156").
  3. **Counters & Step Accounting:** Verifies `micro_steps >= successful_updates * accum` and ensures valid non-negative integer counters.
  4. **History Consistency:** Requires `len(history) == epoch + 1`, validates strictly sequential epoch indices (`epoch == i + 1`), monotonic update counters, and asserts that the final history row matches `successful_updates` and `micro_steps`.
  5. **RNG State Integrity:** Validates that `rng` contains non-empty generator states for `"python"`, `"numpy"`, and `"torch"`.
  6. **Execution Cap Boundary:** Asserts that checkpoint `epoch < execution_cap - 1`, failing closed if the checkpoint has already completed `execution_cap` epochs.

### 3.3 Separation of Scheduler Horizon from Execution Cap
- **Historical Fold 1 Distinction:**
  - In Fold 1 Baseline R Colab training, `max_epochs: 25` served simultaneously as the cosine learning rate schedule horizon and the execution loop ceiling. Training naturally terminated at Epoch 11 (6,600 updates) when early stopping patience (5 epochs) triggered following peak validation AP at Epoch 6.
- **Cross-Fold Generalization Protocol (Folds 2–4):**
  - **Scheduler Horizon:** Remains strictly pinned at 25 epochs (`scheduler_horizon_epochs: 25`, ~15,000 updates), guaranteeing that the learning rate decay curve $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T_{\max}))$ is mathematically identical to Fold 1 Baseline R.
  - **Execution Cap:** Set to 10 completed epochs (`execution_cap_epochs: 10`, max 6,000 updates). Bounds GPU execution duration to ~50–80 minutes per fold (< 4% weekly Kaggle GPU quota) while providing ample margin past Epoch 6.
  - **Resume Enforcement:** Both `validate_resume_checkpoint()` and the training loop in `audit_scripts/train_fold1_ctc_binary.py` enforce `start_epoch < execution_cap_epochs`. Any attempt to resume a checkpoint that completed $\ge 10$ epochs fails closed with `ValueError`.
  - **Primary Matched-Budget Endpoint:** The fixed-budget checkpoint (`fixed_budget_checkpoint.pt`) remains strictly locked at **3,600 updates and Epoch 6** (`epoch == 5`), verified and saved independently of the execution cap or early stopping.

### 3.4 Token CLI Path Elimination & Isolation of Legacy Arms
- In `deploy/kaggle/kaggle_entrypoint.py`:
  - Added early fail-closed check rejecting `--remote-mirror-token` when running in cross-fold modes (`train_fold2`, `train_fold3`, `train_fold4`, `verify_fold*`).
  - Removed `--remote-mirror-token` argument propagation from `kaggle_entrypoint.py` to `audit_scripts/kaggle_fold_train_exec.py`.
  - Added support for `--repo-dir` parameter (defaulting to `/kaggle/working/repo`).
  - Isolated `--remote-mirror-token` strictly within legacy C0/C1 modes (`train_c0`, `train_c1`) for backward compatibility.
- In `audit_scripts/train_fold1_ctc_binary.py`:
  - Added early fail-closed check rejecting `--remote-mirror-token` when `fold_idx in (2, 3, 4)` before hardware initialization.

### 3.5 Strict Environment / Kaggle Secrets Credential Path
- In `audit_scripts/train_fold1_ctc_binary.py`:
  - Child trainer resolves credentials strictly via `resolve_cross_fold_hf_token()`, reading from `HF_TOKEN` / `HUGGINGFACE_TOKEN` environment variables or Kaggle `UserSecretsClient`.
  - Configures `KaggleRemoteMirror` with `strict_secrets_only=True` and `require_verification=True`.
  - Probes live authenticated write-readback-delete capability to Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` before beginning training.
  - Added repo root and `src` to `sys.path` in `train_fold1_ctc_binary.py` for headless subprocess stability.

### 3.6 Kaggle Dataset Mounting & Auto-Extracted Archive Verification
- In `audit_scripts/kaggle_fold_train_exec.py` (`find_and_stage_fold_package`):
  - Prioritizes `--kaggle-input-dir` as the primary search location.
  - Conducts immediate test speaker exclusion audit on input directory: searches for `{test_spk}.zip` or `{test_spk}` folder and raises `PermissionError` immediately if found.
  - Supports outer archive formats: direct `.zip`, `.bin`, and `.dat`.
  - **Automatic Extraction Support:** If Kaggle auto-extracts the uploaded dataset into the 18 inner speaker archives (`{spk}.zip`), detects all 18 authorized speakers, checks test speaker exclusion, and deterministically reconstructs the outer zip with `ZIP_STORED` and fixed timestamp `(2026, 1, 1, 0, 0, 0)`.
  - Verifies cryptographic SHA-256 digest of the reconstructed outer archive against `EXPECTED_FOLD_PACKAGE_SHA256[fold_idx]`.
  - Added verify-only modes (`--mode verify_fold2`, `verify_fold3`, `verify_fold4`) in `deploy/kaggle/kaggle_entrypoint.py` to audit dataset staging and remote mirror probe without training.

### 3.7 Comprehensive Automated Test Suite Expansion
- Expanded `tests/test_rq1_trn_03a_kaggle_deployment.py` from 11 to 20 comprehensive tests:
  1. `test_executable_mocked_checkpoint_recovery`: Bit-exact weight and state recovery (PASS).
  2. `test_epoch_6_fixed_budget_checkpoint_recovery_and_lock`: Epoch 6 / 3,600-update lock (PASS).
  3. `test_fresh_pretrained_initialization_strictly_enforced`: Rejection of Fold 1 checkpoints (PASS).
  4. `test_find_and_stage_fold_package_success`: Staging and verification for Folds 2, 3, 4 (PASS).
  5. `test_actual_test_speaker_leak_detection_independent_of_sha`: Independent leak detection (PASS).
  6. `test_corrupt_archive_detection`: Rejection of invalid archives (PASS).
  7. `test_wrong_fold_resume_rejected`: Cross-fold resume rejection (PASS).
  8. `test_remote_mirror_missing_token_production_failure`: Fail-closed on missing HF_TOKEN (PASS).
  9. `test_endpoint_classification`: Endpoint classification invariants (PASS).
  10. `test_kaggle_fold_train_exec_dry_run_all_folds`: Preflight dry-run across all folds (PASS).
  11. `test_kaggle_entrypoint_mode_dispatch`: CLI mode dispatch including verify modes (PASS).
  12. `test_resume_validation_missing_metadata`: Rejection of missing metadata/provenance (PASS).
  13. `test_resume_validation_mismatched_hyperparameters`: Rejection of altered hyperparameters/ancestry (PASS).
  14. `test_resume_validation_counters_and_rng`: Counter and RNG integrity checks (PASS).
  15. `test_resume_beyond_execution_cap_rejected`: Fail-closed rejection of resume past 10 epochs (PASS).
  16. `test_valid_checkpoint_recovery_at_various_epochs`: Valid recovery across multiple epochs (PASS).
  17. `test_find_and_stage_auto_extracted_zip`: Outer zip reconstruction from auto-extracted speaker zips (PASS).
  18. `test_find_and_stage_auto_extracted_with_test_speaker_leak`: Leak detection in auto-extracted directory (PASS).
  19. `test_kaggle_entrypoint_rejects_cli_token_for_cross_fold`: Rejection of CLI token in entrypoint (PASS).
  20. `test_child_trainer_rejects_cli_token`: Rejection of CLI token in child trainer (PASS).

---

## 4. Results & Verification
- **Automated Regression Suite:** All 270 repository tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - `270 passed, 4 skipped, 1 warning in 129.08s` (100% green).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` passes 20/20 tests in 17.75s:
  - 100% test pass rate across all failure, recovery, and compatibility paths.
- **Preflight Dry-Run:** Verified for Fold 2 via `audit_scripts/kaggle_fold_train_exec.py --fold 2 --dry-run --check-only`:
  - Split manifest SHA-256 verified (`7d6bfdf6...`).
  - Fresh pretrained initialization verified.
  - Dataset package located and verified (`112cc9e0...`, 18 speakers, 0 test speakers).
  - Remote mirror configured.
  - Zero errors reported.

---

## 5. Output Artifacts
- **Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py` (strengthened resume validation, scheduler vs execution cap separation, strict secrets).
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py` (auto-extracted dataset reconstruction, priority input dir check, 10-epoch cap enforcement).
- **Kaggle Entrypoint:** `deploy/kaggle/kaggle_entrypoint.py` (early CLI token rejection, cross-fold verify modes, `--repo-dir` parameter).
- **Fold Configurations:** `configs/rq1_trn_03a_fold{2,3,4}.yaml` (explicit `scheduler_horizon_epochs: 25`, `execution_cap_epochs: 10`).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` (20 comprehensive automated tests).
- **Step Checkpoint:** `manifests/RQ1-TRN-03A/checkpoints/04c-final-launch-gate-correction.md`.

---

## 6. Authoritative Kaggle Launch Command & Restart Instructions

### Authoritative Launch Command (To be executed on Kaggle GPU VM):
```bash
python3 deploy/kaggle/kaggle_entrypoint.py \
  --mode train_fold2 \
  --commit-sha <VERIFIED_COMMIT_SHA> \
  --kaggle-input-dir /kaggle/input/langfens-rq1-fold2-staging \
  --remote-mirror-backend huggingface \
  --remote-mirror-target minkhoaa/langfens-rq1-cross-fold-checkpoints
```

### Preflight Staging Check Command (Non-training verification):
```bash
python3 deploy/kaggle/kaggle_entrypoint.py \
  --mode verify_fold2 \
  --commit-sha <VERIFIED_COMMIT_SHA> \
  --kaggle-input-dir /kaggle/input/langfens-rq1-fold2-staging \
  --remote-mirror-backend huggingface \
  --remote-mirror-target minkhoaa/langfens-rq1-cross-fold-checkpoints
```

### Prerequisites on Kaggle VM:
1. Pinned GPU machine shape: `machine_shape: NvidiaTeslaT4` (Tesla T4 $\times$ 2).
2. Attached private staged dataset: `minkhoaa/langfens-rq1-fold2-staging` containing `authorized_RQ1-TRN_fold2_train_val_only.zip` (or auto-extracted 18 speaker zips).
3. Configured Kaggle Secret: `HF_TOKEN` with write permissions to `minkhoaa/langfens-rq1-cross-fold-checkpoints`.
4. Ephemeral working directory: `/kaggle/working`.

---

## 7. Remaining External Platform Blockers & Safety Boundaries
- **External Platform Blockers:**
  1. Operator authorization required to stage dataset `authorized_RQ1-TRN_fold2_train_val_only.zip` to private Kaggle dataset (`minkhoaa/langfens-rq1-fold2-staging`).
  2. Operator authorization required to launch Kaggle GPU kernel for Fold 2 training.
  3. Kaggle Secrets credential verification on live Kaggle kernel (`HF_TOKEN`).
- **Safety Boundaries (Strict Compliance with `AGENTS.md`):**
  - Zero local training, zero GPU execution, zero backward passes executed on local machine.
  - Zero TEST partition audio, TextGrid, or labels unsealed or evaluated (Fold 2 test speakers `BWC, HKK, HQTV, MBMPS, SKA, SVBI` remain 100% sealed).
  - Baseline R model architecture and multi-task loss contracts kept completely unchanged.
  - Fold 2 training will NOT start automatically. Execution halted awaiting explicit operator approval.
