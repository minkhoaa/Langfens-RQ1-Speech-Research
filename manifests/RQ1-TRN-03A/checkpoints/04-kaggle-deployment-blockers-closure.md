# LANGFENS — RQ1-TRN-03A / 04-Kaggle Deployment Blockers Closure

**Status:** `COMPLETED`

**Recorded at:** 2026-10-29T18:45:00+07:00

**Evidence source:** Commits `bd174ab7857235f9bcd1c600cec5bd17790d7cf4`, `audit_scripts/train_fold1_ctc_binary.py`, `audit_scripts/kaggle_fold_train_exec.py`, `deploy/kaggle/kaggle_entrypoint.py`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Code Fix, Full Test Suite & Deployment Preflight Verification)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Remediate all operational blockers preventing Kaggle GPU execution for independent Baseline R training across Folds 2, 3, and 4. Fix trainer resume variable reference defects, implement a dedicated Kaggle runner invoking the generalized Baseline R trainer (prohibiting C0/C1 fine-tuning runners and Fold 1 weights), reuse proven private Kaggle staging and Hugging Face persistence components with per-epoch SHA-256 readback, enforce strict per-fold test isolation and credential hygiene, and verify deployment readiness via mocked integration tests and non-training preflight.

---

## 2. Inputs & Dependencies
- **Base Commit:** `bd174ab7857235f9bcd1c600cec5bd17790d7cf4` on branch `review/rq1-trn-03a-cross-fold-planning`.
- **Generalized Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py`.
- **Kaggle Deployment Orchestrator:** `deploy/kaggle/kaggle_entrypoint.py`.
- **Hugging Face Mirroring Utility:** `audit_scripts/kaggle_checkpoint_sync.py`.
- **Fold Configuration Files:** `configs/rq1_trn_03a_fold{2,3,4}.yaml`.
- **Authorized Packages:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip`.
- **Master Policy:** `AGENTS.md`.

---

## 3. Work Performed & Remediations

### 3.1 Fixed Trainer Resume Bug (`audit_scripts/train_fold1_ctc_binary.py`)
- **Root Cause:** In the trainer's resume block, `fixed_budget_saved = (outdir / "fixed_budget_checkpoint.pt").exists()` attempted to resolve `outdir` before `outdir = Path(cfg["output_dir"])` was initialized, causing an `UnboundLocalError` whenever `--resume` was passed.
- **Remediation:**
  - Initialized `drive_out` and `outdir` immediately following configuration parsing, before any resumption logic executes.
  - Added optional `--output-dir` CLI override for flexible runner routing.
  - Added optional `fixed_budget_saved: bool = False` argument to `checkpoint()` signature and preserved state in payload dictionary.
  - Supported remote mirror arguments (`--remote-mirror-backend`, `--remote-mirror-target`, `--remote-mirror-token`, `--remote-rel-prefix`) directly inside the baseline trainer to allow atomic per-epoch cloud synchronization.

### 3.2 Dedicated Fold 2–4 Kaggle Runner (`audit_scripts/kaggle_fold_train_exec.py`)
- Implemented a dedicated execution driver specifically for cross-fold Baseline R training:
  - Invokes the generalized Baseline R trainer (`audit_scripts/train_fold1_ctc_binary.py`), strictly rejecting the C0/C1 fine-tuning driver (`train_rq1_trn_02b.py`).
  - **Fresh Pretrained Initialization Guarantee:** Validates that no initial checkpoint (such as Fold 1 `best.pt`) is loaded or provided, ensuring training begins strictly from `facebook/wav2vec2-base-960h` with frozen backbone and freshly initialized heads.
  - Reuses proven GPU hardware verification (`verify_kaggle_gpu_device`), requiring Nvidia Tesla T4.
  - Validates split manifest SHA-256 (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`).
  - Staging & test isolation: copies the fold's authorized package to `/kaggle/working/staged/`, verifies SHA-256, and confirms that none of that fold's 6 test speakers exist in the archive.
  - Checkpoint conflict guard: prevents accidental overwrite of pre-existing checkpoints unless `--resume`.
  - Classifies final training endpoint, requiring `fixed_budget_checkpoint.pt` (3,600 updates), `best.pt`, `last.pt`, and `history.json`.

### 3.3 Kaggle Entrypoint Integration (`deploy/kaggle/kaggle_entrypoint.py`)
- Extended entrypoint modes to support `--mode train_fold2`, `train_fold3`, `train_fold4`, and `train_fold` with `--fold`.
- Configured default remote mirror destination to `minkhoaa/langfens-rq1-cross-fold-checkpoints` for cross-fold baseline training.

### 3.4 Genuine Remote Persistence & Credential Hygiene
- Configured atomic per-epoch mirroring of `last.pt`, `best.pt`, `fixed_budget_checkpoint.pt`, and `history.json` to private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` under subfolders `fold2/`, `fold3/`, `fold4/`.
- Reused `KaggleRemoteMirror` with live write-readback-delete probe and post-upload SHA-256 readback via `hf_hub_download`.
- Enforced credential hygiene: API tokens are read dynamically from environment variables (`HF_TOKEN`) or Kaggle Secrets (`UserSecretsClient().get_secret("HF_TOKEN")`), never printed to stdout or embedded in notebooks.

### 3.5 Automated Integration Test Suite (`tests/test_rq1_trn_03a_kaggle_deployment.py`)
- Authored 8 new integration tests covering:
  - Resume bug resolution (proves no `UnboundLocalError` occurs and state restores properly).
  - Fresh pretrained initialization enforcement (proves fail-closed rejection of Fold 1 checkpoints).
  - Package discovery, staging, and SHA-256 verification across Folds 2, 3, 4.
  - Test speaker leakage failure path (proves fail-closed `PermissionError` if test speaker is present).
  - SHA-256 corruption failure path (proves fail-closed `ValueError` if package is corrupted).
  - Endpoint classification invariants (`COMPLETED_3600_UPDATES` vs `INCOMPLETE` / `FAILED`).
  - Non-training preflight dry-run across all three folds.
  - Kaggle entrypoint CLI argument dispatch.

---

## 4. Results & Verification
- **Automated Regression Suite:** All 258 tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - 258 passed, 4 skipped in 138.24s (100% green).
- **Deployment Integration Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` passes 8 tests in 18.45s.
- **Deployment Preflight Dry-Run:**
  - Fold 2: `audit_scripts/kaggle_fold_train_exec.py --fold 2 --dry-run` $\implies$ **PASS**
  - Fold 3: `audit_scripts/kaggle_fold_train_exec.py --fold 3 --dry-run` $\implies$ **PASS**
  - Fold 4: `audit_scripts/kaggle_fold_train_exec.py --fold 4 --dry-run` $\implies$ **PASS**
- **Test Isolation Maintained:** Zero test audio or TextGrid files accessed, extracted, staged, or evaluated.

---

## 5. Output Artifacts
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`
- **Updated Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py` (resume bug fixed, remote mirror supported)
- **Updated Entrypoint:** `deploy/kaggle/kaggle_entrypoint.py`
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py`
- **Step Checkpoint:** `manifests/RQ1-TRN-03A/checkpoints/04-kaggle-deployment-blockers-closure.md`

---

## 6. Restart / Recovery Instructions
- **Active Branch:** `review/rq1-trn-03a-cross-fold-planning`
- **Current State:** Deployment blockers closed; preflight verified; zero GPU compute launched.
- **Next Step:** Obtain explicit operator authorization to upload staging packages to private Kaggle datasets and launch Kaggle GPU training kernels.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:**
  - Zero local training or backward passes.
  - Zero test evaluation (all test partitions remain 100% un-evaluated).
  - Zero GPU jobs launched without operator authorization.
  - Zero pretrained-to-Fold1 checkpoint transfer.
- **Pending Approvals:** Operator authorization to stage datasets and execute Kaggle GPU runs.
