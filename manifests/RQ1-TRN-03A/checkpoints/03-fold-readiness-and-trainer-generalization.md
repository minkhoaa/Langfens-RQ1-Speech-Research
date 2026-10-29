# LANGFENS — RQ1-TRN-03A / 03-Fold Readiness & Trainer Generalization

**Status:** `COMPLETED`

**Recorded at:** 2026-10-29T17:30:00+07:00

**Evidence source:** Commits `e954fa784d1bc8684ed10adc55b6ed4d2b62335d`, `configs/rq1_trn_03a_fold{2,3,4}.yaml`, `audit_scripts/build_fold_packages.py`, `audit_scripts/train_fold1_ctc_binary.py`, `audit_scripts/evaluate_fold1_ctc_binary.py`, `audit_scripts/evaluate_protocol_z.py`, `tests/test_rq1_trn_03a_fold_readiness_integration.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Build, Full Test Suite & Partition Integrity Verification)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Implement and verify training and evaluation readiness for Folds 2, 3, and 4 before remote GPU execution. Generalize baseline trainer and evaluators (Protocol O and Protocol Z) to support all 4 folds dynamically without altering Fold 1 numerical behavior; implement deterministic TRAIN/VAL-only zip builders for Folds 2–4 with physical exclusion of test partitions; author pinned configurations with exact Baseline R hyperparameter parity; enforce dual checkpoint semantics; audit remote Kaggle dataset staging and Hugging Face checkpoint mirroring destinations with SHA-256 readback; and add integration tests.

---

## 2. Inputs & Dependencies
- **Base Commit:** `e954fa784d1bc8684ed10adc55b6ed4d2b62335d` on branch `review/rq1-trn-03a-cross-fold-planning`.
- **Master Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`).
- **Raw Master Archive:** `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` (SHA-256 `490d5a43...`).
- **Baseline Configuration Blueprint:** `configs/rq1_trn_02a_fold1.yaml`.
- **Master Policy:** `AGENTS.md`.

---

## 3. Work Performed

### 3.1 Trainer Generalization (`audit_scripts/train_fold1_ctc_binary.py`)
- Removed hardcoded fold index `1` and hardcoded split counts (`train=2399`, `val=300`).
- Implemented dynamic fold resolution from configuration (`fold_idx = int(cfg.get("fold", 1))`) and optional CLI override (`--fold`).
- Implemented dynamic manifest inspection reading exact expected utterance counts for `train` and `val` for the active fold, asserting exact dataset sizes.
- Enforced dual checkpoint semantics: saves `fixed_budget_checkpoint.pt` at update 3,600 (Epoch 6) while continuing training to allow validation-selected `best.pt` capture under early stopping patience 5.
- Preserved bit-identical Fold 1 execution when `fold: 1` is configured.

### 3.2 Evaluator Generalization (`evaluate_fold1_ctc_binary.py` & `evaluate_protocol_z.py`)
- **Protocol O Evaluator:**
  - Generalized `--fold` support (inferred from config or CLI).
  - Generalized speaker boundary detection: dynamically finds transition index where `speaker` changes, correctly producing `[150]` for Fold 1 (`NJS` $\to$ `RRBI`) and matching transition indices for Folds 2, 3, and 4.
  - Generalized within-speaker cyclic derangement (`within_speaker_cyclic`), ensuring zero cross-speaker audio leakage and zero self-matches for all folds.
  - Generalized per-speaker AP reporting to dynamic validation speaker keys (`f"{spk.lower()}_ap"`), keeping `"njs_ap"` and `"rrbi_ap"` for Fold 1.
- **Protocol Z Evaluator:**
  - Added `--fold` argument (default `1`).
  - Generalized validation/test speaker assertions against the manifest for any fold.
  - Generalized zero-oracle and gold reference dataset instantiation for the active fold.
  - Generalized `per_speaker_data` tracking dictionary across all active validation speakers.

### 3.3 Fold Package Builder & Physical Test Exclusion (`audit_scripts/build_fold_packages.py`)
- Implemented deterministic packaging utility extracting strictly the 18 authorized speakers ($S_{\text{train}}^{(k)} \cup S_{\text{val}}^{(k)}$) from `l2arctic_release_v5.0.zip`.
- For each authorized speaker, packs strictly the 150 gold `.wav` and `.TextGrid` files listed in `l2arctic_4fold_splits.json`.
- Implemented deterministic ZipInfo timestamps (`2026-01-01 00:00:00`) and outer stored compression matching Fold 1 baseline artifact.
- Generated and cryptographically verified all three packages:
  - **Fold 2 Package:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip`
    - Size: 767,978,303 bytes | SHA-256: `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`
    - Test speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`): **100% physically absent**.
  - **Fold 3 Package:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip`
    - Size: 759,803,411 bytes | SHA-256: `e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2`
    - Test speakers (`ERMS`, `NCC`, `RRBI`, `THV`, `YBAA`, `YDCK`): **100% physically absent**.
  - **Fold 4 Package:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`
    - Size: 783,006,402 bytes | SHA-256: `653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94`
    - Test speakers (`NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA`): **100% physically absent**.

### 3.4 Pinned Fold Configurations
- Created `configs/rq1_trn_03a_fold2.yaml`, `configs/rq1_trn_03a_fold3.yaml`, and `configs/rq1_trn_03a_fold4.yaml`.
- Enforced complete hyperparameter parity: `seed: 1337`, `batch_size: 2`, `gradient_accumulation_steps: 2`, `warmup_updates: 100`, `max_epochs: 25` (15,000 updates horizon), `learning_rate: 0.001`, `weight_decay: 0.01`, `optimizer: AdamW`, `scheduler: cosine`, `loss_weights: {ctc: 1.0, binary: 1.0}`, `backbone_frozen: true`, `amp: fp16`, `grad_scaler_initial_scale: 4096`, `early_stopping_patience: 5`.

### 3.5 Kaggle Staging & Hugging Face Checkpoint Mirroring Audit
- **Kaggle Staging:** Verified that remote staging requires private Kaggle datasets (`minkhoaa/langfens-rq1-fold2-staging`, etc.) created with `is_private: true`. Each dataset contains exclusively the corresponding authorized train/val zip package.
- **Hugging Face Mirroring:** Audited `KaggleRemoteMirror` in `audit_scripts/kaggle_checkpoint_sync.py`:
  - Target private repository: `minkhoaa/langfens-rq1-cross-fold-checkpoints`.
  - Authenticated write-readback-delete probe verified before training starts.
  - Atomic per-epoch upload with synchronous SHA-256 readback via `hf_hub_download` guarantees zero ephemeral data loss.
  - Subfolder routing: `fold2/`, `fold3/`, `fold4/`.

### 3.6 Correction of RQ2 Scope Wording
- Corrected RQ2 definition to **"Multi-Agent Automated Test Generation"** across `AGENTS.md`, `INDEX.md`, `CROSS_FOLD_RESEARCH_PLAN.md`, `PROCESS_CHECKPOINT.md`, and step checkpoints.

---

## 4. Results & Verification

- **Automated Regression Suite:** All 250 tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - 250 passed, 4 skipped in 140.25s (100% green).
- **Integration Test Suite (`test_rq1_trn_03a_fold_readiness_integration.py`):**
  - `test_all_fold_configs_parity`: PASS (all 4 configs verified bit-identical).
  - `test_fold_package_isolation_and_contents`: PASS (all packages verified to contain exactly 18 authorized speakers and 0 test speakers).
  - `test_l2arctic_dataset_loader_all_folds`: PASS (loads 2,400/299 for Fold 2, 2,400/300 for Fold 3, 2,399/300 for Fold 4).
  - `test_dataset_loader_fails_closed_on_test_split`: PASS (fails closed with `KeyError` on test audio access).
  - `test_trainer_split_and_horizon_across_folds`: PASS (verifies 600 updates/epoch, 15,000 horizon, 3,600 fixed budget).
  - `test_within_speaker_cyclic_derangement_all_folds`: PASS (verifies 0 self-matches, 0 cross-speaker leaks for all folds).
- **Split Invariance:** Verified `l2arctic_4fold_splits.json` SHA-256 remains `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.

---

## 5. Output Artifacts
- **Package Builder Script:** `audit_scripts/build_fold_packages.py`
- **Authorized Fold Packages:**
  - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip`
  - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip`
  - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`
- **Fold Configuration Files:**
  - `configs/rq1_trn_03a_fold2.yaml`
  - `configs/rq1_trn_03a_fold3.yaml`
  - `configs/rq1_trn_03a_fold4.yaml`
- **Updated Drivers & Evaluators:**
  - `audit_scripts/train_fold1_ctc_binary.py` (generalized multi-fold trainer)
  - `audit_scripts/evaluate_fold1_ctc_binary.py` (generalized Protocol O evaluator)
  - `audit_scripts/evaluate_protocol_z.py` (generalized Protocol Z evaluator)
- **Integration Test Suite:** `tests/test_rq1_trn_03a_fold_readiness_integration.py`
- **Step Checkpoint:** `manifests/RQ1-TRN-03A/checkpoints/03-fold-readiness-and-trainer-generalization.md`

---

## 6. Restart / Recovery Instructions
- **Active Branch:** `review/rq1-trn-03a-cross-fold-planning`
- **Current State:** Implementation and readiness verification complete; zero GPU compute launched.
- **Next Step:** Obtain explicit operator authorization to upload staging packages to private Kaggle datasets and launch Kaggle GPU training kernels.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:**
  - Zero local training or backward passes.
  - Zero test evaluation (all test partitions remain 100% un-evaluated).
  - Zero GPU jobs launched without operator authorization.
  - Zero pretrained-to-Fold1 checkpoint transfer.
- **Pending Approvals:** Operator authorization to stage datasets and execute Kaggle GPU runs.
