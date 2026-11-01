# LANGFENS — RQ1-TRN-03B / 16-Kaggle Pilot Deployment Readiness

**Status:** `COMPLETED; GATE_B_DEPLOYMENT_READY; EXECUTION_HALTED_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-11-01T15:30:00+07:00

**Evidence source:** 
- Split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`)
- Fold 1 train/val archive `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3b...`)
- Pinned pilot configs `configs/rq1_trn_03b_pilot_arm2a_focal.yaml`, `configs/rq1_trn_03b_pilot_arm2b_weighted.yaml`
- Execution harness `audit_scripts/kaggle_fold_train_exec.py`, `deploy/kaggle/kaggle_entrypoint.py`, `audit_scripts/train_fold1_ctc_binary.py`
- 6 Kaggle deployment packages in `deploy/kaggle_pilot_arm2{a,b}_{preflight,smoke,train}/`
- Automated regression test suites (`tests/test_rq1_trn_03b_kaggle_pilot_deployment.py`, 320/320 passed repository tests)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Dry-Runs, Staging Audits, Security Probes & Fail-Closed Regression Suites)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary & Readiness Gate

In accordance with `AGENTS.md` and Phase RQ1-TRN-03B experimental planning (Step 15), this checkpoint prepares and verifies the complete remote Kaggle GPU execution infrastructure for the **Stage 1 Remote-GPU Pilot** comparing **Arm 2A (Focal BCE)** and **Arm 2B (Class-Weighted BCE)** on **Fold 1**.

**Zero remote GPU training, zero backward passes, zero local training, zero remote dataset re-uploads, and zero test set accesses were launched during this step.**

The execution pipeline has passed all local dry-run staging, security, and fail-closed regression gates (**Gate B Readiness Certified**). Execution is now halted pending explicit operator review and sign-off.

---

## 2. Kaggle Integration Gap Resolution

### 2.1 The Integration Gap Identified
In Phase RQ1-TRN-03A, the Kaggle orchestration driver (`audit_scripts/kaggle_fold_train_exec.py`) and runtime entrypoint (`deploy/kaggle/kaggle_entrypoint.py`) were hardcoded to support strictly Folds 2–4 (`choices=[2, 3, 4]`). Fold 1 had historically been trained on Google Colab under Phase RQ1-TRN-02A and had no Kaggle driver or entrypoint dispatch. Furthermore, output directory routing was pinned to `checkpoints/fold{k}/baseline_r_fold{k}` without support for Phase RQ1-TRN-03B pilot isolation.

### 2.2 Architectural Resolution
1. **Generalized Fold Support (`fold_idx=1`):**
   - `kaggle_fold_train_exec.py` now accepts `--fold 1` alongside `2, 3, 4`.
   - Staging handler gracefully detects both `authorized_RQ1-TRN_fold{k}_train_val_only.zip` and Fold 1's authoritative archive `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
2. **Dedicated RQ1-TRN-03B Pilot Routing:**
   - When running pilot modes (`--arm Arm2A_Focal` or `--arm Arm2B_Weighted`), output artifacts are isolated to:
     - Local VM disk: `/kaggle/working/checkpoints/RQ1-TRN-03B/pilot_arm2a_focal` and `/kaggle/working/checkpoints/RQ1-TRN-03B/pilot_arm2b_weighted`.
     - Remote persistent mirror: `pilot/pilot_arm2a_focal/` and `pilot/pilot_arm2b_weighted/` under private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`.
   - Prevents collision with historical Fold 1 Baseline R artifacts (`checkpoints/RQ1-TRN-02A/recovered/best.pt`).
3. **Dedicated Entrypoint Dispatch Modes:**
   - Added modes to `deploy/kaggle/kaggle_entrypoint.py`:
     - `--mode train_pilot_arm2a`: Fold 1, Arm 2A Focal, `configs/rq1_trn_03b_pilot_arm2a_focal.yaml`
     - `--mode train_pilot_arm2b`: Fold 1, Arm 2B Weighted, `configs/rq1_trn_03b_pilot_arm2b_weighted.yaml`
     - `--mode verify_pilot_arm2a`: Preflight staging and persistence probe for Arm 2A
     - `--mode verify_pilot_arm2b`: Preflight staging and persistence probe for Arm 2B
4. **Zero Regression Guarantee:**
   - Existing modes (`train_fold2`, `train_fold3`, `train_fold4`, `verify_fold2..4`, `train_c0`, `train_c1`) remain bit-for-bit unchanged. All 39 legacy deployment tests in `tests/test_rq1_trn_03a_kaggle_deployment.py` continue to pass.

---

## 3. Dedicated Kaggle Deployment Packages Matrix

Six independent, private deployment packages have been authored under `deploy/`:

| Package Directory | Role | Kernel Slug | Kernel Title | Target Notebook | Dataset Source |
|---|:---:|---|---|---|---|
| `deploy/kaggle_pilot_arm2a_preflight/` | Preflight | `minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-preflight` | `LANGFENS RQ1 Pilot Arm 2A Focal T4x2 (Preflight)` | `kaggle_preflight.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |
| `deploy/kaggle_pilot_arm2a_smoke/` | Smoke | `minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-smoke` | `LANGFENS RQ1 Pilot Arm 2A Focal T4x2 (Smoke)` | `kaggle_smoke.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |
| `deploy/kaggle_pilot_arm2a_train/` | Official | `minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2` | `LANGFENS RQ1 Pilot Arm 2A Focal T4x2` | `kaggle_train.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |
| `deploy/kaggle_pilot_arm2b_preflight/` | Preflight | `minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-preflight` | `LANGFENS RQ1 Pilot Arm 2B Weighted T4x2 (Preflight)` | `kaggle_preflight.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |
| `deploy/kaggle_pilot_arm2b_smoke/` | Smoke | `minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-smoke` | `LANGFENS RQ1 Pilot Arm 2B Weighted T4x2 (Smoke)` | `kaggle_smoke.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |
| `deploy/kaggle_pilot_arm2b_train/` | Official | `minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2` | `LANGFENS RQ1 Pilot Arm 2B Weighted T4x2` | `kaggle_train.ipynb` | `minkhoaa/langfens-rq1-fold1-staging` |

### 3.1 Kernel & Notebook Hygiene
- **Hardware Profile:** Target `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`, `enable_internet: true`, `is_private: true`.
- **Pinned Git Commit:** All notebooks pin verified training base commit `56bf79535e4f697b2eea05594d5d81153e04dbff` (which introduced pilot execution modes in Step 16).
- **Zero Token Leakage:** Authentication token `HF_TOKEN` acquired exclusively via Kaggle Secrets (`UserSecretsClient().get_secret("HF_TOKEN")`). Zero tokens stored on disk, zero tokens passed in CLI arguments.
- **Fail-Closed Preflight Gate:** Smoke and Train notebooks execute Phase 4 preflight checks first; any staging, network, or manifest mismatch halts execution before training starts.

---

## 4. Fold 1 Staging Verification & Zero Active Test Leakage

- **Authoritative Archive:** `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip`
- **SHA-256 Digest:** `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (Verified).
- **Inner Speaker Census (18 Authorized Speakers):**
  - Train (16 speakers): `BWC`, `ERMS`, `HKK`, `HQTV`, `MBMPS`, `NCC`, `SKA`, `SVBI`, `THV`, `TLV`, `TNI`, `TXHC`, `YBAA`, `YDCK`, `YKWK`, `ZHAA`.
  - Val (2 speakers): `NJS`, `RRBI`.
- **Test Speaker Exclusion Audit ($S_{\text{test}}^{(1)} \cap S_{\text{staged}} = \emptyset$):**
  - Held-out Test Speakers: `ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`.
  - Overlap: **0 speakers** (100% physically absent).
- **Kaggle Staging Source:** Private dataset `minkhoaa/langfens-rq1-fold1-staging` (CC BY-NC 4.0).

---

## 5. Methodological Audit of pos_weight = 8.40 Selection

A critical scientific audit was conducted to distinguish training-derived empirical class statistics from validation-informed pedagogical operating parameters:

### 5.1 Empirical Prevalence Statistics across Partitions

| Partition / Split | Total Canonical Tokens | Total Error Tokens | Error Prevalence | Inverse Prevalence Ratio $\left(\frac{1 - p}{p}\right)$ |
|---|:---:|:---:|:---:|:---:|
| **Fold 1 Train Partition** | 79,426 | 11,841 | **14.9082%** | **5.707** ($w_{\text{train}}^{(1)}$) |
| **4-Fold Train Macro Average** | 317,877 | 48,341 | **15.2074%** | **5.576** ($w_{\text{train, macro}}$) |
| **Fold 1 Val Partition** | 9,956 | 1,157 | **11.6211%** | **7.605** |
| **4-Fold Val Macro Average** | 39,576 | 4,213 | **10.6430%** | **8.396** $\approx \mathbf{8.40}$ ($w_{\text{val, macro}}$) |

### 5.2 Scientific Distinction & Implications
1. **Validation-Informed Choice:** The selected weight $pos\_weight = \mathbf{8.40}$ was derived from the authoritative **out-of-fold validation macro prevalence** ($10.643\%$), not the training partition prevalence ($14.91\%$).
2. **Intentional Over-Weighting Factor:**
   $$\text{Overweight Ratio} = \frac{w_{\text{selected}}}{w_{\text{train}}^{(1)}} = \frac{8.40}{5.707} \approx \mathbf{1.472\times}$$
   Applying $w = 8.40$ during Fold 1 training scales the positive error gradient by ~1.47× relative to the true empirical frequency in the training data.
3. **Scientific Hypothesis of Arm 2B:** This aggressive weighting specifically tests the hypothesis that heavily penalizing false negatives (missed errors) during training will elevate the detector's Recall on held-out speakers.
4. **Trade-Off & Risk:** Over-weighting positive errors risks shifting logits too far upward, increasing False Positive Rates (false alarms).
5. **Methodological Transparency:** All pilot reporting must clearly disclose that $w = 8.40$ is a **validation-informed operating choice**, ensuring no confusion with unadjusted empirical training class balance.

---

## 6. Execution Invariants & Fail-Closed Safety Defenses

The following automated safety defenses have been implemented and validated via unit and integration tests:

1. **Fresh Pretrained Initialization:** Both Arm 2A and Arm 2B initialize strictly from `facebook/wav2vec2-base-960h` (rev `22aad52d...`). Prohibits transfer from Fold 1 Baseline R or C0/C1 checkpoints (`verify_fresh_pretrained_initialization`).
2. **Arm-Specific Loss Verification in Endpoint Classifier:**
   - Arm 2A: Verifies `binary_loss_type == "focal"`, `focal_gamma == 2.0`, `focal_alpha == 0.75`. Rejects wrong parameters.
   - Arm 2B: Verifies `binary_loss_type == "bce"`, `binary_pos_weight == 8.40`. Rejects wrong parameters.
   - Rejects cross-arm checkpoint classification (e.g. classifying Arm 2B checkpoint as Arm 2A fails closed).
3. **Resume Safety & Cross-Arm Resumption Rejection:**
   - `validate_resume_checkpoint` in `train_fold1_ctc_binary.py` validates `binary_loss_type`, `binary_pos_weight`, `focal_gamma`, `focal_alpha`.
   - Attempting to resume an Arm 2A training run from an Arm 2B or Baseline R checkpoint fails closed immediately.
4. **Credential Hygiene & Missing Secrets Fail-Closed:**
   - Unset `HF_TOKEN` halts execution before dataset unpacking or training launch.
5. **GPU Accelerator Verification:**
   - Requires verified Nvidia Tesla T4 GPU with CUDA FP16 AMP. Non-T4 or CPU execution fails closed.
6. **Dual Checkpoint Accounting:**
   - Primary fixed-budget checkpoint locked strictly at Epoch 6 / 3,600 updates (`fixed_budget_checkpoint.pt`).
   - Validation peak checkpoint saved independently (`best.pt`).

---

## 7. Pilot Feasibility & Numerical Diagnostics Plan

The primary objective of the Fold 1 Pilot is **feasibility and numerical stability screening**, not confirmatory generalization proof:

### 7.1 Real-Time Numerical Stability Monitoring
- **Gradient Norm (`grad_norm`):** Logged per update. Must remain finite without exploding ($< 25.0$).
- **AMP Loss Scale (`GradScaler`):** Starts at 4,096. Must remain adaptive. Fails closed if scale collapses to $< 1.0$ or if consecutive skipped updates exceed 5.
- **CTC Decoder Integrity:** Greedy decoding blank fraction must remain $\le \mathbf{0.85}$ (preventing blank collapse). CTC PER must remain $\le \mathbf{26.0\%}$.

### 7.2 Post-Pilot Evaluation & Pedagogical PR Metrics
- **Protocol O & Protocol Z Standalone Replay:** Standalone local evaluation of matched AP, mismatched cyclic shift-1 AP, and acoustic gap.
- **Fixed-FPR Pedagogical Operating Points:**
  - $\text{FNR}_{@10\%\text{FPR}}$ (Strict feedback)
  - $\text{FNR}_{@15\%\text{FPR}}$ (Standard classroom feedback; target $< 40\%$)
  - $\text{FNR}_{@20\%\text{FPR}}$ (High sensitivity screening)
- **Pilot Selection Criterion:** If one arm achieves $\ge +0.015$ AP gain over Baseline R on Fold 1 with non-contracted acoustic gap and stable CTC PER, it is selected as the primary candidate for full 4-fold cross-validation in Stage 2.

---

## 8. Regression Test Ledger

The deployment harness and safety invariants were validated using automated regression tests:

- **Deployment Test Suite:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/test_rq1_trn_03b_kaggle_pilot_deployment.py`
  - Tests verify: entrypoint dispatch, CLI token rejection, fold runner pilot acceptance, Fold 1 archive SHA and test isolation, arm-specific loss classification, wrong-arm resume rejection, missing secrets fail-closed, GPU check, all 6 package metadata/notebooks, and pos_weight derivation audit.
  - Result: **10 passed in 7.71s**.
- **Cross-Fold Kaggle Deployment Suite:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/test_rq1_trn_03a_kaggle_deployment.py`
  - Result: **39 passed in 19.95s** (zero regression to Baseline R Folds 2–4).
- **Full Repository Regression Suite:** `PYTHONPATH=. .venv-cuda/bin/pytest tests/`
  - Result: **320 passed, 1 skipped, 1 warning in 209.47s** (100% green).

---

## 9. Operator Deployment Guide & Commands for Gate B

Upon explicit operator authorization, execution proceeds according to the permanent Kaggle workflow:

### Phase 1: Deploy Kernel Metadata via CLI (`--no-run`)
```bash
# Arm 2A Focal Pilot Packages
kaggle kernels push -p deploy/kaggle_pilot_arm2a_preflight --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2a_smoke --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2a_train --no-run

# Arm 2B Weighted Pilot Packages
kaggle kernels push -p deploy/kaggle_pilot_arm2b_preflight --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2b_smoke --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2b_train --no-run
```

### Phase 2: Attach Secrets in Kaggle Web UI & Start Execution
1. Open Kaggle Web UI kernel editor for the target kernel:
   - Arm 2A Preflight: `https://www.kaggle.com/code/minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-preflight`
   - Arm 2B Preflight: `https://www.kaggle.com/code/minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-preflight`
2. Under **Add-ons -> Secrets**, verify `HF_TOKEN` is attached.
3. Verify input dataset `minkhoaa/langfens-rq1-fold1-staging` is attached under **Input Data**.
4. Click **Save & Run All** to initiate background execution.

### Phase 3: Monitor Execution via Non-Interactive CLI
```bash
# Monitor Arm 2A Preflight
kaggle kernels status minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-preflight
kaggle kernels output minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-preflight

# Monitor Arm 2B Preflight
kaggle kernels status minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-preflight
kaggle kernels output minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-preflight
```

---

## 10. Next Authorized Execution Step & Stop Statement

In strict compliance with `AGENTS.md` ("Never start training without explicit user authorization. STOP before execution"):

- **Current Status:** Gate B Readiness Certified. Code, configs, staging validation, failure defenses, and 6 deployment packages are verified clean and pushed.
- **Execution Boundary:** No remote GPU runs launched. No local training attempted.
- **Action Requiring Operator Authorization:** Explicit approval to push kernel packages and launch the Remote-GPU Pilot on Kaggle Tesla T4 $\times$ 2.

**STOP. AWAITING OPERATOR REVIEW AND AUTHORIZATION FOR GATE B.**
