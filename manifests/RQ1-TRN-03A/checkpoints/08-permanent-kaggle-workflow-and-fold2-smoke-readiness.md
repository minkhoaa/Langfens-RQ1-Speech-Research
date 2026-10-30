# LANGFENS — RQ1-TRN-03A / 08-Permanent Kaggle Workflow and Fold 2 Smoke Readiness

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-30T14:45:00+07:00

**Evidence source:** `AGENTS.md`, `deploy/kaggle_fold2_smoke/`, `deploy/kaggle_fold2_preflight/`, `deploy/kaggle_fold2_train/`, `audit_scripts/build_simplified_kaggle_deployment.py`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, 285/285 passed, 1 skipped)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

This checkpoint finalizes the permanent operational workflow for remote Kaggle execution and prepares the Fold 2 smoke execution package:
1. **Permanent Kaggle Workflow Formally Enacted in `AGENTS.md`:**
   - **CLI deploy:** `kaggle kernels push -p <package-dir> --no-run` staging code and metadata in the cloud without triggering premature background execution.
   - **Web UI Authorization & Launch:** Notebook editor opened in Kaggle Web UI, `HF_TOKEN` secret attached under **Add-ons -> Secrets**, and headless execution initiated via **"Save & Run All"** (Commit notebook).
   - **CLI Monitoring & Durable Mirrored Persistence:** Execution status and logs monitored non-interactively via `kaggle kernels status <slug>` and `kaggle kernels output <slug>`. Checkpoints and metrics are atomically mirrored to private Hugging Face persistent storage (`minkhoaa/langfens-rq1-cross-fold-checkpoints`) with cryptographic SHA-256 readback.
2. **Dedicated Fold 2 Smoke Kernel ID & Title Separation:**
   - Allocated unique, non-colliding slug `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke` and matching title `LANGFENS RQ1 Fold 2 Baseline R T4x2 (Smoke)`.
   - Preserves preflight kernel (`minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight`) and official training kernel (`minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`) intact without metadata collisions.
3. **Four Verified Operational Invariants:**
   - **Pinned Training Base Commit:** `cc56b82ca1a9a6ceeca85929e07db4cac8537348` (containing execution-cap and run-id support in `kaggle_fold_train_exec.py` and `kaggle_entrypoint.py`).
   - **1-Epoch Execution Cap:** `--execution-cap-epochs 1` capped strictly to 1 completed epoch (~600 updates), preventing quota overruns during validation of remote persistence.
   - **Isolated Run ID:** `fold2_smoke_1epoch` writing checkpoints locally to `/kaggle/working/checkpoints/fold2/fold2_smoke_1epoch/` and remotely to Hugging Face prefix `fold2/fold2_smoke_1epoch/`, completely isolated from official Baseline R runs (`fold2_baseline_r`).
   - **Private Dataset Staging:** Bound strictly to authorized private dataset `minkhoaa/langfens-rq1-fold2-staging` (SHA-256 `112cc9e0...`, 18 train/val speakers, 6 active test speakers strictly excluded, CC BY-NC 4.0).
4. **Zero Safety Boundary Violations:**
   - Zero GPU jobs dispatched during local authoring.
   - Zero TEST partition audio or labels accessed.
   - Zero credentials or tokens exposed in files or CLI invocations.
   - Zero scientific hyperparameters or loss formulations altered.

---

## 2. Three-Package Deployment Registry

| Dimension | Preflight Package (`deploy/kaggle_fold2_preflight/`) | Smoke Package (`deploy/kaggle_fold2_smoke/`) | Official Training Package (`deploy/kaggle_fold2_train/`) |
|---|---|---|---|
| **Kernel ID** | `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight` | `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke` | `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` |
| **Kernel Title** | `LANGFENS RQ1 Fold 2 Baseline R T4x2 (Preflight)` | `LANGFENS RQ1 Fold 2 Baseline R T4x2 (Smoke)` | `LANGFENS RQ1 Fold 2 Baseline R T4x2` |
| **Notebook File** | `kaggle_preflight.ipynb` | `kaggle_smoke.ipynb` | `kaggle_train.ipynb` |
| **Pinned Commit** | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` | `cc56b82ca1a9a6ceeca85929e07db4cac8537348` |
| **Execution Mode** | Preflight only (`--mode verify_fold2`) | Mandatory preflight gate + 1-epoch smoke | Mandatory preflight gate + 10-epoch Baseline R |
| **Execution Cap** | 0 epochs (0 updates) | 1 completed epoch (~600 updates) | 10 completed epochs (max 6,000 updates; fixed budget at Epoch 6 / 3,600 updates) |
| **Run ID** | N/A | `fold2_smoke_1epoch` | `fold2_baseline_r` |
| **Local Checkpoint Path** | None | `/kaggle/working/checkpoints/fold2/fold2_smoke_1epoch/` | `/kaggle/working/checkpoints/fold2/fold2_baseline_r/` |
| **Remote Mirror Prefix** | Probe verification only (`_probe_write_test`) | `fold2/fold2_smoke_1epoch/` | `fold2/fold2_baseline_r/` |
| **Dataset Source** | `minkhoaa/langfens-rq1-fold2-staging` | `minkhoaa/langfens-rq1-fold2-staging` | `minkhoaa/langfens-rq1-fold2-staging` |
| **Machine Shape** | `NvidiaTeslaT4` (T4 $\times$ 2) | `NvidiaTeslaT4` (T4 $\times$ 2) | `NvidiaTeslaT4` (T4 $\times$ 2) |
| **Kaggle Kernel URL** | `https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight` | `https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke` | `https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` |

---

## 3. Operational Launch Protocol for Fold 2 Smoke Execution

### Step 1: Deploy Smoke Kernel via Kaggle CLI
Push the dedicated smoke package without starting immediate background execution:
```bash
kaggle kernels push -p deploy/kaggle_fold2_smoke/ --no-run
```
Expected output confirmation:
```text
Kernel version 1 successfully created.
```

### Step 2: Grant HF_TOKEN & Launch Headless Run via Kaggle Web UI
1. Open the smoke kernel in your browser:
   `https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke`
2. In the notebook editor, verify the runtime settings:
   - **Accelerator:** GPU T4 $\times$ 2
   - **Internet:** On
3. Navigate to menu: **Add-ons -> Secrets**.
4. Check / enable the user secret **`HF_TOKEN`** for this kernel.
5. Click **"Save & Run All"** (or **Commit** button) to initiate remote headless background execution.

### Step 3: Monitor Smoke Execution via Kaggle CLI
Monitor kernel execution non-interactively from local terminal:
```bash
kaggle kernels status minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke
```
Wait until status transitions from `running` to `complete`.

### Step 4: Retrieve Logs & Verify Verification Indicators
Fetch logs and verify Phase 6 completion:
```bash
kaggle kernels output minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke
```
Look for the definitive completion indicator in the output:
```text
================================================================================
[SMOKE EXECUTION COMPLETED SUCCESSFULLY]
Epoch: 1
Updates: 600
...
================================================================================
```

### Step 5: Verify Hugging Face Remote Mirror
Verify remote checkpoint presence and SHA-256 integrity on the Hugging Face model repository:
- Repository: `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- Verified path prefix: `fold2/fold2_smoke_1epoch/`
- Expected mirrored files:
  - `epoch_001.pt` (atomic epoch snapshot)
  - `last.pt` (link/copy of epoch 1)
  - `best.pt` (initial validation peak)
  - `history.json` (1-epoch metrics history)
  - `training_endpoint_report.json`

---

## 4. Verification Evidence & Test Ledger

- **Suite Execution:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/`
- **Result:** **285 passed, 1 skipped, 1 warning in 172.19s** (100% green).
- **Key Regression Tests Verified:**
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold2_smoke_package`: PASS. Proves smoke notebook enforces preflight gate first, 1-epoch execution cap, `fold2_smoke_1epoch` run ID, strict Kaggle Secrets, and post-execution artifact verification.
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold2_three_entrypoints_isolation`: PASS. Proves preflight, smoke, and train packages possess mutually exclusive kernel IDs and matching titles without collision.
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_workflow_agents_policy`: PASS. Proves `AGENTS.md` explicitly codifies the CLI push `--no-run` $\rightarrow$ Web UI secret grant $\rightarrow$ CLI monitor lifecycle.
  - `tests/test_rq1_trn_03a_kaggle_deployment.py::test_kaggle_fold_train_exec_smoke_and_cap_args`: PASS. Proves execution cap and smoke run ID arguments are accepted by the training driver.
