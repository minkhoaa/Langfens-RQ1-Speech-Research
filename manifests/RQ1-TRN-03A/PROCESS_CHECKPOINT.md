# LANGFENS — RQ1-TRN-03A / Cross-Fold Speaker-Independent Generalization Planning (Readiness Verified)

**Status:** `READINESS_VERIFIED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-29T17:45:00+07:00

**Evidence source:** Commits `7a3b9ecf5e9efedc480b477f531ebe9318ae3415`, `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c`, `configs/rq1_trn_02a_fold1.yaml`, `audit_scripts/train_fold1_ctc_binary.py`, `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Static Planning, Baseline Code Parity Audit & Regression Suite)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Design and specify a mathematically rigorous, leak-free 4-fold cross-validation phase for Folds 2, 3, and 4 (Phase RQ1-TRN-03A). Audit speaker overlap across splits, establish an authoritative per-fold test-isolation policy, plan independent Baseline R training across folds without checkpoint reuse, enforce bit-level hyperparameter parity with Fold 1 Baseline R, define cross-fold evaluation metrics and small-sample confidence interval aggregation, and structure downstream phases (model improvement and final testing) while preserving all completed Fold 1 R/C0/C1 findings and the `NOT_SUPPORTED` whole-phone prompt dropout verdict.

---

## 2. Inputs & Dependencies
- **Source Commit:** `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c` (planning baseline on `review/rq1-trn-03a-cross-fold-planning`).
- **Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Validated Baseline Architecture:** Two-head `LangfensWav2Vec2MDD` defined in `src/langfens_prep/model.py` (frozen Wav2Vec2 backbone `facebook/wav2vec2-base-960h` rev `22aad52d...`, trainable CTC head, trainable cross-attention + binary head, disabled diagnosis head).
- **Baseline R Pinned Config & Driver:** `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py`.
- **Pretrained Weights:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).
- **Platform Credentials:** Kaggle CLI (`machine_shape: NvidiaTeslaT4`, weekly quota 30h) and Hugging Face API (`HF_TOKEN` in environment).

---

## 3. Work Performed & Protocol Corrections
1. **Preserved Fold 1 R/C0/C1 Findings:** Reaffirmed that Reference R (`best.pt`, AP 0.317021, PER 23.68%, acoustic gap 0.022299) remains the primary production baseline. Preserved the scientific verdict that Treatment C1 ($p=0.25$ whole-phone binary prompt dropout) is `NOT_SUPPORTED` ($\Delta_{\text{dropout}} = -0.007918$, 95% bootstrap CI `[-0.018328, +0.000385]`). Retracted casual unverified significance claims ($p < 10^{-5}$) on CTC PER differences.
2. **Corrected Baseline R Hyperparameter Parity Contract:**
   - Corrected master seed from `42` to `seed: 1337`, matching `configs/rq1_trn_02a_fold1.yaml`.
   - Corrected micro batch size to `batch_size: 2` and gradient accumulation to `gradient_accumulation_steps: 2` (effective batch size = 4 utterances per update).
   - Corrected warmup updates to `warmup_updates: 100` fixed updates (not 10% of steps).
   - Corrected scheduler horizon to cosine annealing over `max_epochs: 25` ($\approx 15,000$ updates), matching the learning rate decay rate of Baseline R.
   - Enforced identical early stopping semantics: patience 5 on validation AP, conditioned on `blank_fraction < 0.99`.
3. **Established Dual Checkpoint Accounting:**
   - Distinguished the **Fixed-Budget Checkpoint** (captured strictly at Epoch 6 / 3,600 updates, matching C0/C1 and Fold 1 peak selection) from the **Validation-Selected Checkpoint** (`best.pt`, selected at peak validation AP prior to early stopping).
   - Mandated saving and reporting both checkpoints to prevent confounding training exposure with model performance.
4. **Clarified Per-Fold Test Isolation & Refuted False Global Sealing:**
   - Evaluated speaker assignments across all 4 folds in `l2arctic_4fold_splits.json`.
   - Clarified that in standard 4-fold cross-validation, the test speakers of Fold 1 ($S_{\text{test}}^{(1)}$) legitimately and necessarily serve as training speakers for Folds 2, 3, and 4.
   - Refuted erroneous claims that all 4 test sets are globally sealed from staging archives.
   - Defined strict per-fold isolation: For model $M_k$, $S_{\text{test}}^{(k)}$ is strictly excluded from $M_k$'s training set, validation set, early stopping / checkpoint selection, threshold calibration, and remote staging archive (`authorized_RQ1-TRN_fold{k}_train_val_only.zip`).
   - All test sets ($S_{\text{test}}^{(1..4)}$) remain 100% un-evaluated during the cross-fold validation phase, preventing model selection leakage.
5. **Preserved Deterministic Within-Speaker Shift-1 Acoustic Protocol:**
   - Explicitly mandated `method="within_speaker_cyclic"` via `generate_derangement_indices()`, ensuring that mismatched audio is permuted deterministically within the same human speaker's utterances.
6. **Clarified Confusion Metrics & Pedagogical Interpretations:**
   - Explicitly defined $\text{Recall} = \frac{TP}{TP + FN}$ and $\text{FNR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$ (missed error rate).
   - Explicitly defined $\text{Specificity} = \frac{TN}{TN + FP}$ and $\text{FPR} = \frac{FP}{TN + FP} = 1 - \text{Specificity}$ (false alarm rate).
   - Disambiguated mathematical formulas from ambiguous biometric FAR/FRR terminology.
7. **Documented Small-Sample Statistical Limitations:**
   - Documented that Macro Student's $t$ CI with $df=3$ produces wide intervals due to small $N=4$ fold sample size.
   - Documented that cluster bootstrapping over $G=8$ validation speakers has known small-cluster bias ($G < 30$).
   - Mandated reporting both macro, micro, and disaggregated per-speaker metrics.
8. **Corrected Thesis Scope Boundaries:**
   - Confirmed that SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback belong to RQ1.
   - Confirmed that RQ2 is reserved exclusively for Multi-Agent Automated Test Generation.
9. **Added Comprehensive Protocol Regression Test Suite:**
   - Implemented `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py` (8 tests, 100% passing).
10. **Completed Step 03 Fold Readiness Implementation & Verification:**
   - Generalized baseline trainer (`audit_scripts/train_fold1_ctc_binary.py`) to support `cfg["fold"] = 2, 3, 4` dynamically with manifest-derived split-size assertions and dual checkpointing (`fixed_budget_checkpoint.pt` at update 3,600 and validation `best.pt`).
   - Generalized Protocol O and Protocol Z evaluators (`evaluate_fold1_ctc_binary.py` and `evaluate_protocol_z.py`) with dynamic speaker transition detection and within-speaker cyclic derangement across all 4 folds.
   - Authored fold configurations `configs/rq1_trn_03a_fold{2,3,4}.yaml` with exact bit-level hyperparameter parity.
   - Implemented deterministic packaging utility `audit_scripts/build_fold_packages.py` and built all 3 authorized packages (`authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip`), physically excluding test speakers.
   - Added integration test suite `tests/test_rq1_trn_03a_fold_readiness_integration.py` (6 tests).
11. **Completed Step 04 Kaggle Deployment Blockers Closure:**
   - Remediated trainer resume bug where `outdir` was referenced before initialization in `audit_scripts/train_fold1_ctc_binary.py`. Added `fixed_budget_saved: bool = False` to `checkpoint()` signature.
   - Implemented dedicated Kaggle runner `audit_scripts/kaggle_fold_train_exec.py` invoking the generalized Baseline R trainer (prohibiting C0/C1 fine-tuning runner and Fold 1 weights).
   - Enforced fresh pretrained initialization guarantee (`verify_fresh_pretrained_initialization`).
   - Reused `KaggleRemoteMirror` for atomic per-epoch checkpoint upload to `minkhoaa/langfens-rq1-cross-fold-checkpoints` with SHA-256 readback probe.
   - Updated `deploy/kaggle/kaggle_entrypoint.py` to support cross-fold training modes (`train_fold2`, `train_fold3`, `train_fold4`).
   - Implemented comprehensive deployment test suite `tests/test_rq1_trn_03a_kaggle_deployment.py` (8 tests).
   - Executed full repository regression test suite: `258 passed, 4 skipped in 138.24s`.
   - Executed deployment preflight dry-run across all three folds with 100% success.
12. **Completed Step 04B Kaggle Durability & Recovery Closure:**
   - Enforced immutable fixed-budget checkpoint lock at strictly 3,600 updates and Epoch 6 (`is_fixed_budget_endpoint=True`), preventing later epoch weights from overwriting verified 3,600-update artifacts.
   - Implemented `validate_resume_checkpoint()` validating state dicts, fold identity, seed, model ancestry, and strictly rejecting cross-fold resume.
   - Implemented recovery of `fixed_budget_checkpoint.pt` from `remote_mirror` when resuming past Epoch 6.
   - Mandated verified private remote persistence (`huggingface`) for production Kaggle execution, rejecting empty targets or disabled verification.
   - Enforced per-epoch `history.json` upload with SHA-256 readback.
   - Removed credential-file fallbacks and CLI token arguments; used environment variables / Kaggle Secrets only (`resolve_cross_fold_hf_token()`).
   - Implemented executable mocked checkpoint recovery tests and isolated speaker-leakage tests in `tests/test_rq1_trn_03a_kaggle_deployment.py` (11 tests).
   - Full repository regression suite passed: `261 passed, 4 skipped in 140.23s`.

---

## 4. Results & Verification
- **Split Manifest Invariance:** Verified `l2arctic_4fold_splits.json` SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- **Automated Split Suite:** `python3 audit_scripts/validate_splits.py` passed all 4 core checks with 0 errors.
- **Automated Regression Suite:** All 261 tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - 261 passed, 4 skipped in 140.23s (100% green).
- **Integration Test Suite (`test_rq1_trn_03a_fold_readiness_integration.py`):**
  - `test_all_fold_configs_parity`: PASS (all 4 configs verified bit-identical).
  - `test_fold_package_isolation_and_contents`: PASS (all packages verified to contain exactly 18 authorized speakers and 0 test speakers).
  - `test_l2arctic_dataset_loader_all_folds`: PASS (loads 2,400/299 for Fold 2, 2,400/300 for Fold 3, 2,399/300 for Fold 4).
  - `test_dataset_loader_fails_closed_on_test_split`: PASS (fails closed with `KeyError` on test audio access).
  - `test_trainer_split_and_horizon_across_folds`: PASS (verifies 600 updates/epoch, 15,000 horizon, 3,600 fixed budget).
  - `test_within_speaker_cyclic_derangement_all_folds`: PASS (verifies 0 self-matches, 0 cross-speaker leaks for all folds).
- **Per-Fold Isolation Proved:** For every fold $k$, $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$, $S_{\text{val}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$, and $S_{\text{train}}^{(k)} \cap S_{\text{val}}^{(k)} = \emptyset$.
- **Test Isolation Maintained:** Zero test audio or TextGrid files accessed, extracted, staged, or evaluated.

---

## 5. Output Artifacts
- **Corrected Research Plan:** `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`
- **Phase Overview Checkpoint:** `manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md`
- **Step Checkpoint 01:** `manifests/RQ1-TRN-03A/checkpoints/01-cross-fold-research-plan-and-test-isolation.md`
- **Step Checkpoint 02:** `manifests/RQ1-TRN-03A/checkpoints/02-cross-fold-protocol-parity-and-isolation-corrections.md`
- **Step Checkpoint 03:** `manifests/RQ1-TRN-03A/checkpoints/03-fold-readiness-and-trainer-generalization.md`
- **Step Checkpoint 04:** `manifests/RQ1-TRN-03A/checkpoints/04-kaggle-deployment-blockers-closure.md`
- **Step Checkpoint 04B:** `manifests/RQ1-TRN-03A/checkpoints/04b-kaggle-durability-and-recovery-closure.md`
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`
- **Package Builder Script:** `audit_scripts/build_fold_packages.py`
- **Fold Configuration Files:** `configs/rq1_trn_03a_fold{2,3,4}.yaml`
- **Authorized Fold Packages:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip`
- **Integration Test Suites:**
  - `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py`
  - `tests/test_rq1_trn_03a_fold_readiness_integration.py`
  - `tests/test_rq1_trn_03a_kaggle_deployment.py`
- **Policy Corrections:** `AGENTS.md` (Per-fold test sealing rule and baseline parity contract)
- **Updated Global Registry:** `docs/process-checkpoints/INDEX.md`

---

## 6. Restart / Recovery Instructions
- **Active Branch:** `review/rq1-trn-03a-cross-fold-planning`
- **Current State:** Protocol corrected; awaiting operator authorization to begin implementation.
- **Restart / Recovery Instructions:**
  - All configurations, packages, trainers, and evaluators are implemented and verified ready.
  - Zero GPU compute launched; awaiting operator authorization to launch Kaggle GPU training kernels.
  - Next step upon approval: upload packages to private Kaggle datasets and launch Fold 2 training run.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:**
  - DO NOT run model training or backward passes on local CPU/GPU (`AGENTS.md`).
  - DO NOT access, unseal, or stage test split files for any fold.
  - DO NOT warm-start Folds 2, 3, or 4 from Fold 1 `best.pt`.
  - DO NOT alter the frozen model architecture or loss contracts.
- **Pending Approvals:** Operator authorization packet detailed in Section 10 of `CROSS_FOLD_RESEARCH_PLAN.md`.
