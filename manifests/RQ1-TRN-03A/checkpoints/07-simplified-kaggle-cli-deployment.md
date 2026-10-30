# LANGFENS — RQ1-TRN-03A / 07-Simplified Kaggle CLI Deployment

**Status:** `COMPLETED; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-30T11:45:00+07:00

**Evidence source:** `deploy/kaggle_fold2_preflight/`, `deploy/kaggle_fold2_train/`, `audit_scripts/build_simplified_kaggle_deployment.py`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, 278/278 passed)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

In response to preflight verification operational learnings, the Fold 2 Kaggle deployment architecture has been simplified to provide a **CLI-first deployment with minimal orchestration**:
1. **Elimination of Notebook Execution Flags:** Replaced the single-notebook pattern (`EXECUTE_TRAINING = False / True`) with two dedicated, self-contained entrypoints:
   - **Preflight Entrypoint (`deploy/kaggle_fold2_preflight/`):** Runs staging, dataset SHA, manifest SHA, test-speaker isolation, and live Hugging Face write-readback-delete probe checks. Terminated immediately upon preflight pass; zero training code or execution flags.
   - **Training Entrypoint (`deploy/kaggle_fold2_train/`):** Runs preflight verification first as a mandatory fail-closed gate; if preflight passes, immediately launches official Baseline R training on Fold 2, mirrors checkpoints per epoch with SHA-256 readback, and verifies endpoint invariants.
2. **Independent CLI Executability:** Each entrypoint package contains its own `kernel-metadata.json`, `dataset-metadata.json`, and notebook, independently dispatchable via standard `kaggle kernels push -p <path>`.
3. **Fail-Closed Guardrails Preserved:**
   - Training entrypoint strictly mandates preflight pass before training dispatch (`subprocess.run(preflight_cmd, check=True)`).
   - `HF_TOKEN` resolved strictly via `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")`; zero disk token files, zero CLI token arguments.
   - Pinned training source base commit preserved: `407cf04e83704d9e83953e9309e3c7dca8d7154e`.
   - Authorized Fold 2 dataset preserved: `minkhoaa/langfens-rq1-fold2-staging` (SHA-256 `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`, 18 train/val speakers, 6 test speakers excluded).
   - Remote persistent target preserved: `minkhoaa/langfens-rq1-cross-fold-checkpoints`.
4. **Preservation of Existing Deployment:** Existing deployment package (`deploy/kaggle_fold2/`) and historical deployments (`deploy/kaggle/`, `deploy/kaggle_c1/`) remain 100% bit-intact and unmodified.
5. **Zero GPU Invocations During Development:** No GPU jobs launched, no remote datasets modified, no TEST data accessed.

---

## 2. Deployment Architecture Comparison

| Dimension | Previous Single-Notebook Design (`deploy/kaggle_fold2/`) | Simplified CLI-First Design (`deploy/kaggle_fold2_{preflight,train}/`) |
|---|---|---|
| **Entrypoints** | 1 monolithic notebook (`kaggle_notebook.ipynb`) | 2 dedicated packages (`preflight/` and `train/`) |
| **Gating Mechanism** | In-notebook variable flag (`EXECUTE_TRAINING = False`) | Entrypoint separation: preflight package has 0 training code |
| **CLI Preflight Push** | `kaggle kernels push -p deploy/kaggle_fold2/` (requires `EXECUTE_TRAINING=False`) | `kaggle kernels push -p deploy/kaggle_fold2_preflight/` |
| **CLI Training Push** | Required manually editing notebook JSON to toggle `EXECUTE_TRAINING = True` | `kaggle kernels push -p deploy/kaggle_fold2_train/` |
| **Preflight Failure in Training** | Dependent on cell order state | Built into training Phase 4 as checked prerequisite subprocess |
| **Secrets Resolution** | Kaggle Secrets `HF_TOKEN` | Kaggle Secrets `HF_TOKEN` (identical) |
| **Disk Token Fallbacks** | Prohibited | Prohibited (identical) |
| **CLI Token Arguments** | Prohibited | Prohibited (identical) |

---

## 3. Package Registry & Metadata

### 3.1 Preflight Entrypoint Package (`deploy/kaggle_fold2_preflight/`)
- **Metadata (`kernel-metadata.json`):**
  - `id`: `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`
  - `title`: `LANGFENS RQ1 Fold 2 Baseline R T4x2 (Preflight)`
  - `code_file`: `kaggle_preflight.ipynb`
  - `kernel_type`: `notebook`
  - `machine_shape`: `NvidiaTeslaT4` (T4 $\times$ 2)
  - `is_private`: `true`
  - `enable_gpu`: `true`, `enable_internet`: `true`
  - `dataset_sources`: `["minkhoaa/langfens-rq1-fold2-staging"]`
- **Notebook (`kaggle_preflight.ipynb`, 5 cells):**
  - Cell 0 (Markdown): Protocol, hardware target, preflight-only invariants.
  - Cell 1 (Code): Hardware check (`nvidia-smi`, Tesla T4 assertion, CUDA FP16).
  - Cell 2 (Code): Credentials resolution (`UserSecretsClient().get_secret("HF_TOKEN")`).
  - Cell 3 (Code): Git clone & checkout pinned commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`.
  - Cell 4 (Code): Preflight verification (`kaggle_entrypoint.py --mode verify_fold2`). Emits `[PREFLIGHT] Preflight verification PASSED successfully.`

### 3.2 Training Entrypoint Package (`deploy/kaggle_fold2_train/`)
- **Metadata (`kernel-metadata.json`):**
  - `id`: `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`
  - `title`: `LANGFENS RQ1 Fold 2 Baseline R T4x2`
  - `code_file`: `kaggle_train.ipynb`
  - `kernel_type`: `notebook`
  - `machine_shape`: `NvidiaTeslaT4` (T4 $\times$ 2)
  - `is_private`: `true`
  - `enable_gpu`: `true`, `enable_internet`: `true`
  - `dataset_sources`: `["minkhoaa/langfens-rq1-fold2-staging"]`
- **Notebook (`kaggle_train.ipynb`, 7 cells):**
  - Cell 0 (Markdown): Protocol, fresh pretrained initialization, dual checkpoint accounting.
  - Cell 1 (Code): Hardware check (`nvidia-smi`, Tesla T4 assertion, CUDA FP16).
  - Cell 2 (Code): Credentials resolution (`UserSecretsClient().get_secret("HF_TOKEN")`).
  - Cell 3 (Code): Git clone & checkout pinned commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`.
  - Cell 4 (Code): Mandatory preflight gate (`kaggle_entrypoint.py --mode verify_fold2`). Checked subprocess; halts immediately if preflight fails.
  - Cell 5 (Code): Official training execution (`kaggle_entrypoint.py --mode train_fold2`). Runs 25-epoch horizon, 10-epoch execution cap, atomic per-epoch HF mirror.
  - Cell 6 (Code): Endpoint verification (`classify_fold_training_endpoint`, asserting `COMPLETED_3600_UPDATES`).

---

## 4. Concise Operational CLI Commands

### Phase A: Re-verify Preflight via Kaggle CLI
Once `HF_TOKEN` is attached in the Kaggle UI under Add-ons -> Secrets:
```bash
# Push dedicated preflight entrypoint (zero training risk)
kaggle kernels push -p deploy/kaggle_fold2_preflight/

# Monitor preflight execution
kaggle kernels status minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2

# Retrieve preflight log and verify PASS
kaggle kernels output minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2
```
Expected success log indicator:
```text
[PREFLIGHT] Preflight verification PASSED successfully.
[PREFLIGHT] Zero training, optimizer updates, or checkpoint generation occurred.
```

### Phase B: Launch Official Training (Upon Separate Operator Authorization)
```bash
# Push dedicated training entrypoint (runs preflight first, fails closed, then trains)
kaggle kernels push -p deploy/kaggle_fold2_train/

# Monitor training progress (10 epochs / ~3 hours on Tesla T4x2)
kaggle kernels status minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2

# Retrieve final training outputs and verify COMPLETED_3600_UPDATES
kaggle kernels output minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2
```

---

## 5. Verification & Test Evidence
- Generator executed: `audit_scripts/build_simplified_kaggle_deployment.py`
- Full test suite passed: `PYTHONPATH=. .venv/bin/pytest tests/`
  - Total: **278 passed, 4 skipped, 1 warning in 138.12s** (100% green).
  - 4 new regression tests covering preflight package, training package, entrypoint separation, and preservation of existing deployment passed.
- Working tree clean across existing deployments: `deploy/kaggle_fold2/` has 0 changes.
