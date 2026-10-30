# LANGFENS — RQ1-TRN-03A / 05-Dedicated Kaggle Fold 2 Deployment Package

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-29T23:00:00+07:00

**Evidence source:** Commits `407cf04e83704d9e83953e9309e3c7dca8d7154e` (base) -> `ddd090ab0501a5e186e8da6181f185c7bb1e1eb0` (implementation), `deploy/kaggle_fold2/kernel-metadata.json`, `deploy/kaggle_fold2/dataset-metadata.json`, `deploy/kaggle_fold2/kaggle_notebook.ipynb`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Deployment Package Audit & Test Suite, 274/274 tests green)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Prepare a dedicated, private Kaggle execution package for Fold 2 Baseline R training (`deploy/kaggle_fold2/`) without launching GPU execution or uploading assets to remote platforms.
1. Preserve historical C0 and C1 deployment packages (`deploy/kaggle/`, `deploy/kaggle_c1/`) and metadata completely unchanged.
2. Author dedicated Fold 2 kernel metadata (`deploy/kaggle_fold2/kernel-metadata.json`) targeting Tesla T4 $\times$ 2 (`NvidiaTeslaT4`), private visibility under CC BY-NC 4.0, internet enabled for private Hugging Face mirroring, and private dataset source `minkhoaa/langfens-rq1-fold2-staging`.
3. Author dedicated Fold 2 dataset metadata (`deploy/kaggle_fold2/dataset-metadata.json`) specifying private staging under CC BY-NC 4.0 with explicit exclusion of active test partition speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`).
4. Author a dedicated 7-phase execution notebook (`deploy/kaggle_fold2/kaggle_notebook.ipynb`) pinning verified Git commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`, resolving `HF_TOKEN` strictly via Kaggle Secrets (`UserSecretsClient`), running preflight staging and readiness checks via `--mode verify_fold2`, gating training behind `EXECUTE_TRAINING = False`, and eliminating all token file candidate searches and CLI token arguments.
5. Verify compatibility with the 25-epoch cosine scheduler horizon (~15,000 updates), 10-epoch execution cap (max 6,000 updates), and strictly locked Epoch 6 / 3,600-update fixed-budget endpoint.
6. Author automated regression tests validating the deployment metadata, notebook cell structure, and preservation of historical deployments.
7. Prepare the exact Kaggle CLI commands for private dataset creation and live preflight push, stopping strictly before remote execution.

---

## 2. Inputs & Dependencies
- **Base Commit:** `407cf04e83704d9e83953e9309e3c7dca8d7154e` on branch `review/rq1-trn-03a-cross-fold-planning`.
- **Implementation Commit:** `ddd090ab0501a5e186e8da6181f185c7bb1e1eb0`.
- **Fold 2 Staging Package:** `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip` (SHA-256 `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`).
- **Fold 2 Pinned Config:** `configs/rq1_trn_03a_fold2.yaml`.
- **Kaggle Execution Coordinator:** `deploy/kaggle/kaggle_entrypoint.py`.
- **Kaggle Fold Runner:** `audit_scripts/kaggle_fold_train_exec.py`.
- **Baseline Trainer:** `audit_scripts/train_fold1_ctc_binary.py`.
- **Master Policy:** `AGENTS.md`.

---

## 3. Implemented Deployment Specifications

### 3.1 Dedicated Architecture & Preservation of Historical Deployments
- Created dedicated directory `deploy/kaggle_fold2/` completely isolated from historical deployment directories.
- Audited `deploy/kaggle/` (Control Arm C0) and `deploy/kaggle_c1/` (Treatment Arm C1): verified zero modifications, retaining their exact configuration, metadata, and notebook files unchanged.

### 3.2 Kernel Metadata Specification (`deploy/kaggle_fold2/kernel-metadata.json`)
- **Kernel ID:** `minkhoaa/langfens-rq1-fold2-t4x2`
- **Title:** `LANGFENS RQ1 Fold 2 Baseline R T4x2`
- **Code File:** `kaggle_notebook.ipynb`
- **Language / Type:** `python` / `notebook`
- **Platform Invariants:**
  - `is_private: true` (Strict confidentiality under CC BY-NC 4.0; never published publicly)
  - `enable_gpu: true`
  - `enable_tpu: false`
  - `enable_internet: true` (Required for authenticated Hugging Face API mirroring)
  - `machine_shape: "NvidiaTeslaT4"` (Tesla T4 $\times$ 2 GPU accelerator)
  - `dataset_sources: ["minkhoaa/langfens-rq1-fold2-staging"]` (Attached private fold package)

### 3.3 Dataset Metadata Specification (`deploy/kaggle_fold2/dataset-metadata.json`)
- **Dataset ID:** `minkhoaa/langfens-rq1-fold2-staging`
- **Title:** `LANGFENS RQ1 Fold 2 Train/Val Staging Package`
- **License:** CC BY-NC 4.0 (`name: "CC-BY-NC-4.0"`)
- **Visibility:** `is_private: true`
- **Scope & Isolation:** Staged archive contains strictly 18 train/val speakers. Active test partition speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`) are 100% physically excluded.

### 3.4 Seven-Phase Audited Notebook Design (`deploy/kaggle_fold2/kaggle_notebook.ipynb`)
1. **Cell 0 (Markdown):** Monograph recording scientific protocol, bit-level hyperparameter contract, dual checkpoint accounting, test exclusion policy, and Kaggle Secrets authentication. Explicitly records reviewed training source commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`.
2. **Cell 1 (Code — Phase 1: Hardware & Environment Verification):**
   Executes checked `subprocess.run(["nvidia-smi"], check=True)`, verifies CUDA availability, asserts presence of Tesla T4 accelerator, and validates core dependency imports (`torch`, `yaml`, `transformers`, `huggingface_hub`).
3. **Cell 2 (Code — Phase 2: Credentials & Secrets Resolution):**
   Reads `HF_TOKEN` strictly from `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")`. Sets `os.environ["HF_TOKEN"]`. Completely eliminates all local disk candidate searches (`hf_token.txt`, `.hf_token`).
4. **Cell 3 (Code — Phase 3: Clone, Checkout & Verify Tree):**
   Clones repository into `/kaggle/working/repo` via checked subprocess, checks out pinned reviewed training source commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`, asserts bit-exact match against `git rev-parse HEAD`, asserts clean working tree (`git status --porcelain`), and configures `sys.path`.
5. **Cell 4 (Code — Phase 4: Staging Audit & Preflight Verification):**
   Audits `/kaggle/input/langfens-rq1-fold2-staging` contents. Dispatches `deploy/kaggle/kaggle_entrypoint.py --mode verify_fold2` via checked subprocess `subprocess.run(preflight_cmd, check=True)` (replacing all IPython `!python3` calls), executing archive SHA-256 verification (including automatic outer zip reconstruction if auto-extracted by Kaggle), 0 test speaker leak audit, and authenticated Hugging Face write-readback-delete probe without training. Sets `PREFLIGHT_PASSED = True` upon success.
6. **Cell 5 (Code — Phase 5: Gated Official Training Execution):**
   Protected by a dual fail-closed safety gate:
   - Requires `PREFLIGHT_PASSED == True` (raises `RuntimeError` if preflight was not executed or failed).
   - Requires `EXECUTE_TRAINING == True` (remains `False` by default; prints gate message).
   Running top-to-bottom stops safely after preflight verification. Only upon explicit operator authorization (`EXECUTE_TRAINING = True`) does it invoke `kaggle_entrypoint.py --mode train_fold2` via checked `subprocess.run(train_cmd, check=True)`. Zero `--remote-mirror-token` CLI arguments exist. Sets `TRAINING_COMPLETED = True` upon completion.
7. **Cell 6 (Code — Phase 6: Post-Training Endpoint Classification):**
   Invokes `classify_fold_training_endpoint` on `/kaggle/working/checkpoints/fold2/baseline_r_fold2` validating invariant completion: `fixed_budget_checkpoint.pt` (Epoch 6 / 3,600 updates), `best.pt`, `last.pt`, and `history.json` ($\ge 6$, $\le 10$ epochs).

---

## 4. Results & Verification
- **Automated Regression Suite:** All 274 repository tests pass (`PYTHONPATH=. .venv/bin/pytest tests/`):
  - `274 passed, 4 skipped, 1 warning in 157.53s` (100% green).
- **Deployment Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` passes all 24 tests:
  - `test_kaggle_fold2_kernel_metadata`: PASS
  - `test_kaggle_fold2_dataset_metadata`: PASS
  - `test_kaggle_fold2_notebook_structure_and_hygiene`: PASS
  - `test_historical_c0_c1_deployments_intact`: PASS
  - All 20 previous durability, recovery, endpoint, and entrypoint tests: PASS
- **Hygiene Verification:**
  - Zero disk token files referenced in `deploy/kaggle_fold2/`.
  - Zero `--remote-mirror-token` CLI arguments in `deploy/kaggle_fold2/`.
  - Historical `deploy/kaggle/` and `deploy/kaggle_c1/` verified clean and intact.

---

## 5. Output Artifacts
- **Dedicated Kernel Metadata:** `deploy/kaggle_fold2/kernel-metadata.json`
- **Dedicated Dataset Metadata:** `deploy/kaggle_fold2/dataset-metadata.json`
- **Dedicated Gated Execution Notebook:** `deploy/kaggle_fold2/kaggle_notebook.ipynb`
- **Updated Test Suite:** `tests/test_rq1_trn_03a_kaggle_deployment.py` (24 comprehensive tests)
- **Step Checkpoint:** `manifests/RQ1-TRN-03A/checkpoints/05-dedicated-kaggle-fold2-deployment-package.md`

---

## 6. Prepared Remote Upload & Live Preflight Commands (Awaiting Operator Authorization)

In compliance with `AGENTS.md` ("Never start training without explicit user authorization. STOP before execution"), the following commands are prepared for operator execution upon formal approval:

### Step 1: Upload Private Fold 2 Staging Package to Kaggle
```bash
# Prepare isolated staging directory
mkdir -p /tmp/kaggle_fold2_staging
cp deploy/kaggle_fold2/dataset-metadata.json /tmp/kaggle_fold2_staging/
cp datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip /tmp/kaggle_fold2_staging/

# Push private dataset under CC BY-NC 4.0 (requires Kaggle CLI credentials):
kaggle datasets create -p /tmp/kaggle_fold2_staging -r zip
```

### Step 2: Push Dedicated Fold 2 Kernel for Live Preflight Verification
```bash
# Push kernel to Kaggle without training (notebook defaults to verify_fold2 with EXECUTE_TRAINING=False):
kaggle kernels push -p deploy/kaggle_fold2/

# Monitor execution status:
kaggle kernels status minkhoaa/langfens-rq1-fold2-t4x2
kaggle kernels output minkhoaa/langfens-rq1-fold2-t4x2
```

### Step 3: Official Fold 2 Training Launch (Requires Operator Review of Preflight Output):
Set `EXECUTE_TRAINING = True` in Cell 5 of `deploy/kaggle_fold2/kaggle_notebook.ipynb`, push kernel via `kaggle kernels push -p deploy/kaggle_fold2/`, and monitor training completion.

---

## 7. Remaining External Platform Blockers & Strict Safety Boundaries
- **External Platform Blockers:**
  1. Operator authorization required to stage dataset `authorized_RQ1-TRN_fold2_train_val_only.zip` to private Kaggle dataset `minkhoaa/langfens-rq1-fold2-staging`.
  2. Operator authorization required to push and execute Kaggle GPU kernel `minkhoaa/langfens-rq1-fold2-t4x2`.
  3. Kaggle Secrets credential verification on live Kaggle kernel (`HF_TOKEN` attached under notebook Add-ons -> Secrets).
- **Safety Boundaries (Strict Compliance with `AGENTS.md`):**
  - Zero local training, zero GPU execution, zero backward passes executed on local machine.
  - Zero TEST partition audio, TextGrid, or labels unsealed or evaluated (Fold 2 test speakers `BWC, HKK, HQTV, MBMPS, SKA, SVBI` remain 100% sealed).
  - Baseline R model architecture and multi-task loss contracts kept completely unchanged.
  - Historical C0/C1 deployment assets remain 100% unchanged.
  - Fold 2 training will NOT start automatically. Execution halted awaiting explicit operator approval.
