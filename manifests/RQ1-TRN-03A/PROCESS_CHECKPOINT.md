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
13. **Completed Step 04C Final Launch Gate Correction:**
   - Corrected invalid commit SHA (`2200050`) in Kaggle launch instructions, authoritatively pinning verified Git HEAD.
   - Strengthened resume metadata, counter, and provenance validation: enforced fail-closed checks on fold identity, manifest digest (`7d6bfdf6...`), git commit SHA, complete pinned hyperparameters (`seed`, `batch_size`, `accum`, `warmup`, `lr`, `weight_decay`, `loss_weights`, `backbone_frozen`, `amp`, `early_stopping_patience`, `grad_scaler_initial_scale`, `pretrained_model`, `pretrained_revision`), update counters (`micro >= updates * accum`), history monotonicity/matching, complete RNG states (python, numpy, torch), and forbidden transfer keys (`initial_checkpoint`, `baseline_checkpoint`, `finetune_from`).
   - Separated scheduler horizon (25 epochs, ~15,000 updates) from Kaggle execution cap (10 completed epochs, max 6,000 updates); documented historical Fold 1 distinction; enforced 10-epoch execution cap fail-closed across resumed runs; locked 3,600-update / Epoch 6 primary endpoint.
   - Removed CLI token argument path from cross-fold entrypoint execution modes; kept legacy C0/C1 compatibility strictly isolated.
   - Enforced strict environment/Kaggle Secrets credential path (`strict_secrets_only=True`) and verified remote persistence write-readback-delete probe for child trainer.
   - Handled Kaggle dataset mounting and auto-extraction: supported outer zip/bin/dat, auto-detected 18 inner speaker archives with immediate test speaker leak audit, and implemented bit-exact reconstruction and SHA-256 verification.
   - Expanded deployment test suite in `tests/test_rq1_trn_03a_kaggle_deployment.py` from 11 to 20 tests (covering missing metadata, mismatched hyperparameters, resume beyond cap, valid recovery at various epochs, auto-extracted zip staging/leak rejection, and token CLI rejection).
   - Full repository regression suite passed: `270 passed, 4 skipped in 129.08s` (100% green).
14. **Completed Step 05 Dedicated Kaggle Fold 2 Deployment Package:**
   - Preserved historical C0/C1 deployment directories (`deploy/kaggle/`, `deploy/kaggle_c1/`) and metadata completely unchanged.
   - Authored dedicated Fold 2 kernel metadata (`deploy/kaggle_fold2/kernel-metadata.json`) targeting Tesla T4x2 GPU, private execution, internet enabled, and private dataset source `minkhoaa/langfens-rq1-fold2-staging`.
   - Authored dedicated Fold 2 dataset metadata (`deploy/kaggle_fold2/dataset-metadata.json`) under CC BY-NC 4.0 with explicit active test speaker exclusion.
   - Authored 7-phase execution notebook (`deploy/kaggle_fold2/kaggle_notebook.ipynb`) pinning verified Git commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`, acquiring `HF_TOKEN` strictly via Kaggle Secrets, running preflight staging/persistence checks in `--mode verify_fold2`, and gating training behind `EXECUTE_TRAINING = False` with zero disk token searches and zero CLI token arguments.
   - Expanded deployment test suite to 24 tests (274 repository tests total) proving metadata validity, notebook hygiene, and C0/C1 preservation.
15. **Completed Step 06 Fold 2 Remote Staging & Live Preflight Execution:**
   - Under explicit operator authorization, uploaded authorized Fold 2 train/val package to private Kaggle dataset `minkhoaa/langfens-rq1-fold2-staging` as immutable binary package `authorized_RQ1-TRN_fold2_train_val_only.bin` (767,978,303 bytes, SHA-256 `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`). Active test partition speakers (`BWC, HKK, HQTV, MBMPS, SKA, SVBI`) 100% physically absent.
   - Pushed dedicated private kernel `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` to Kaggle GPU runtime with `EXECUTE_TRAINING = False` and `PREFLIGHT_PASSED = False`.
   - Live hardware execution independently verified dual Tesla T4 GPUs (`NVIDIA-SMI 580.178.04`, CUDA 13.0, 15,360 MiB per GPU), PyTorch 2.11.0, Transformers 5.16.1.
   - Credential fail-closed guardrail in Cell 2 operated as designed: threw `RuntimeError` due to missing `HF_TOKEN` secret attachment in the Kaggle UI, halting execution safely before staging unpacking or training.
   - Recorded preflight report in `manifests/RQ1-TRN-03A/checkpoints/06-fold2-remote-staging-and-preflight-report.md`; determined NO-GO for training pending secret attachment and verified write-readback-delete probe.
16. **Completed Step 08 Permanent Kaggle Workflow & Fold 2 Smoke Readiness:**
   - Formally codified permanent Kaggle workflow in `AGENTS.md`: CLI deploy with `kaggle kernels push -p <pkg> --no-run`, Kaggle Web UI secret attachment (`HF_TOKEN`) and "Save & Run All", non-interactive CLI monitoring (`status`/`output`), and verified atomic checkpoint mirroring on Hugging Face.
   - Assigned unique, collision-free smoke slug `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke` and matching title `LANGFENS RQ1 Fold 2 Baseline R T4x2 (Smoke)`, fully preserving preflight (`minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight`) and official training (`minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`) kernel packages.
   - Verified all 4 operational invariants: pinned immutable base commit `cc56b82ca1a9a6ceeca85929e07db4cac8537348`, 1-epoch execution cap (`--execution-cap-epochs 1`), isolated run ID `fold2_smoke_1epoch` (`/kaggle/working/checkpoints/fold2/fold2_smoke_1epoch/` and remote prefix `fold2/fold2_smoke_1epoch/`), and private dataset `minkhoaa/langfens-rq1-fold2-staging`.
   - Added automated regression tests proving 3-way package isolation and `AGENTS.md` policy compliance (285 tests passed, 1 skipped).

---

## 4. Results & Verification
- **Split Manifest Invariance:** Verified `l2arctic_4fold_splits.json` SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- **Automated Split Suite:** `python3 audit_scripts/validate_splits.py` passed all 4 core checks with 0 errors.
- **Automated Regression Suite:** All 285 tests pass (`PYTHONPATH=. .venv-cuda/bin/pytest tests/`):
  - 285 passed, 1 skipped, 1 warning in 172.19s (100% green).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` passes 32/32 tests (100% green).
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
- **Step Checkpoint 04C:** `manifests/RQ1-TRN-03A/checkpoints/04c-final-launch-gate-correction.md`
- **Step Checkpoint 05:** `manifests/RQ1-TRN-03A/checkpoints/05-dedicated-kaggle-fold2-deployment-package.md`
- **Step Checkpoint 06:** `manifests/RQ1-TRN-03A/checkpoints/06-fold2-remote-staging-and-preflight-report.md`
- **Step Checkpoint 07:** `manifests/RQ1-TRN-03A/checkpoints/07-simplified-kaggle-cli-deployment.md`
- **Step Checkpoint 08:** `manifests/RQ1-TRN-03A/checkpoints/08-permanent-kaggle-workflow-and-fold2-smoke-readiness.md`
- **Step Checkpoint 09:** `manifests/RQ1-TRN-03A/checkpoints/09-official-fold2-baseline-r-training-launch-gate.md`
- **Kaggle Fold 2 Preflight Package:** `deploy/kaggle_fold2_preflight/` (`kernel-metadata.json`, `dataset-metadata.json`, `kaggle_preflight.ipynb`)
- **Kaggle Fold 2 Smoke Package:** `deploy/kaggle_fold2_smoke/` (`kernel-metadata.json`, `dataset-metadata.json`, `kaggle_smoke.ipynb`)
- **Kaggle Fold 2 Training Package:** `deploy/kaggle_fold2_train/` (`kernel-metadata.json`, `dataset-metadata.json`, `kaggle_train.ipynb`)
- **Kaggle Fold 2 Deployment Package (Preserved):** `deploy/kaggle_fold2/` (`kernel-metadata.json`, `dataset-metadata.json`, `kaggle_notebook.ipynb`)
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`
- **Package Builder Script:** `audit_scripts/build_fold_packages.py`
- **Fold Configuration Files:** `configs/rq1_trn_03a_fold{2,3,4}.yaml`
- **Authorized Fold Packages:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip`
- **Integration Test Suites:**
  - `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py`
  - `tests/test_rq1_trn_03a_fold_readiness_integration.py`
  - `tests/test_rq1_trn_03a_kaggle_deployment.py`
- **Policy Corrections:** `AGENTS.md` (Permanent Kaggle workflow, per-fold test sealing rule, baseline parity contract)
- **Updated Global Registry:** `docs/process-checkpoints/INDEX.md`

---

## 6. Restart / Recovery Instructions
- **Active Branch:** `review/rq1-trn-03a-cross-fold-planning`
- **Current State:** Step 09 (Official Fold 2 Baseline R Training) successfully completed and cryptographically verified (`COMPLETED; INDEPENDENTLY_VERIFIED`). Smoke run passed 1 epoch; official training run completed 7 epochs (4,200 updates total) on Dual Tesla T4 GPUs before early stopping at patience 5; fixed budget locked at Epoch 6 / 3,600 updates (`fixed_budget_checkpoint.pt`, SHA-256 `ab4c589445feac...`); best validation AP `0.2170` at Epoch 2 (`best.pt`, SHA-256 `b9032ee9ba8d...`); all artifacts mirrored to private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` under `fold2/baseline_r_fold2/`.
- **Next Phase Dispatch:** Step 10 — Fold 3 Remote Staging & Training Preparation (`configs/rq1_trn_03a_fold3.yaml`, `authorized_RQ1-TRN_fold3_train_val_only.zip`).

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:**
  - DO NOT run model training or backward passes on local CPU/GPU (`AGENTS.md`).
  - DO NOT access, unseal, or stage test split files for any fold.
  - DO NOT warm-start Folds 2, 3, or 4 from Fold 1 `best.pt`.
  - DO NOT alter the frozen model architecture or loss contracts.
- **Pending Approvals:** Operator authorization packet detailed in Section 10 of `CROSS_FOLD_RESEARCH_PLAN.md`.
