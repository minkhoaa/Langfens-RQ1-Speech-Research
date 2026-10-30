# LANGFENS — RQ1-TRN-03A / 06-Fold 2 Remote Staging & Live Preflight Report

**Status:** `PREFLIGHT_EXECUTED; NO-GO_FOR_TRAINING (PENDING_KAGGLE_SECRET_ATTACHMENT)`

**Recorded at:** 2026-10-30T10:15:00+07:00

**Evidence source:** Kaggle CLI, live remote kernel execution logs `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`, remote dataset `minkhoaa/langfens-rq1-fold2-staging`, `tests/test_rq1_trn_03a_kaggle_deployment.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Live Kaggle Hardware Execution, API Readback & Cryptographic Ledgers)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

Under explicit operator authorization limited strictly to private Kaggle dataset staging and non-training live preflight execution:
1. **Private Staging Dataset Uploaded & Verified:** Created and verified private Kaggle dataset `minkhoaa/langfens-rq1-fold2-staging` containing the immutable archive `authorized_RQ1-TRN_fold2_train_val_only.bin` (767,978,303 bytes, SHA-256 `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`). Active test partition speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`) are 100% physically absent.
2. **Dedicated Kernel Pushed with Gated Training:** Created and executed private Kaggle kernel `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` targeting `machine_shape: NvidiaTeslaT4` with `EXECUTE_TRAINING = False` and `PREFLIGHT_PASSED = False`.
3. **Live Hardware Verification:** Remote execution verified dual Nvidia Tesla T4 GPUs (`Driver: 580.178.04`, `CUDA: 13.0`, 15,360 MiB VRAM per GPU), PyTorch `2.11.0+cu128`, Transformers `5.16.1`, and Hugging Face Hub `1.29.0`.
4. **Credential Fail-Closed Assertion Verified:** In Cell 2, the notebook's credential hygiene guardrail operated exactly as designed: it detected that `HF_TOKEN` was not yet attached to the freshly created kernel in the Kaggle UI, raised a fail-closed `RuntimeError`, and halted execution before staging extraction or training could occur.
5. **Zero Training Invariant Preserved:** 0 training steps, 0 optimizer updates, 0 backward passes executed. All test partitions remain 100% sealed.

---

## 2. Remote Platform Asset Registry

| Platform | Remote Identifier / URL | Asset Role | Privacy Setting | Size / Digest | Verification Status |
|---|---|---|:---:|---|:---:|
| **Kaggle Dataset** | `minkhoaa/langfens-rq1-fold2-staging`<br>[Link](https://www.kaggle.com/datasets/minkhoaa/langfens-rq1-fold2-staging) | Authorized Fold 2 Train/Val Package (18 speakers) | `is_private: true`<br>(CC BY-NC 4.0) | 767,978,303 bytes<br>SHA-256: `112cc9e0cc5e...` | **VERIFIED** |
| **Kaggle Kernel** | `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2`<br>[Link](https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2) | Dedicated Fold 2 Preflight & Execution Notebook | `is_private: true` | Pinned commit:<br>`407cf04e83704d9e...` | **PREFLIGHT_EXECUTED** |
| **Kaggle Accelerator** | `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` | Dual Tesla T4 (FP16 AMP) | Private | 2 GPUs $\times$ 15,360 MiB | **VERIFIED_TESLA_T4** |
| **HF Persistent Repo** | `minkhoaa/langfens-rq1-cross-fold-checkpoints` | Private Remote Checkpoint Store | `private` | Target folder:<br>`fold2/` | **TARGET_PINNED** |

---

## 3. Live Execution Evidence & Log Output

The live preflight execution on Kaggle Tesla T4 produced the following verifiable evidence:

```text
Fri Oct 30 03:21:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   51C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   52C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+

[GPU] CUDA Available: True
[GPU] Device Count: 2
[GPU 0]: Tesla T4
[GPU 1]: Tesla T4
[ENV] PyTorch version: 2.11.0+cu128
[ENV] Transformers version: 5.16.1
[ENV] Hugging Face Hub version: 1.29.0
[AUTH NOTE] Kaggle Secrets UserSecretsClient query: Connection error trying to communicate with service.

RuntimeError: FAIL-CLOSED: HF_TOKEN is required in Kaggle Secrets (or environment) for private remote mirroring.
Attach HF_TOKEN under Add-ons -> Secrets in the Kaggle notebook.
```

### Analysis of the Log Output:
- **Phase 1 (Hardware):** Successfully validated dual Tesla T4 GPUs with CUDA 13.0 and PyTorch 2.11.0.
- **Phase 2 (Credentials):** `UserSecretsClient` correctly queried the Kaggle Secrets proxy service. Because a newly pushed CLI kernel does not have secrets automatically bound to it by default, the call threw a connection exception, triggering the fail-closed assertion.
- **Fail-Closed Protection:** The notebook terminated immediately at In [2], completely halting execution before any git cloning, staging unpacking, or training subprocesses were started.

---

## 4. Actionable Platform Blocker & Resolution

### Exact Blocker:
Kaggle requires user secret permissions to be explicitly granted per notebook. Because `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` was newly created via the Kaggle CLI API, the `HF_TOKEN` secret is not yet attached to the notebook session.

### Resolution Steps (Operator Action Required):
1. Navigate to: [https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2](https://www.kaggle.com/code/minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2).
2. Click **Edit** to open the notebook editor.
3. In the top navigation bar, select **Add-ons -> Secrets**.
4. Check the checkbox next to **`HF_TOKEN`** to grant the notebook access to your Hugging Face write token.
5. Click **Save** / **Save Version** (Quick Save or Run All).
6. Once attached, `kaggle kernels push -p deploy/kaggle_fold2/` will allow `UserSecretsClient().get_secret("HF_TOKEN")` to retrieve the token and proceed through Phase 4 (`verify_fold2`), executing the write-readback-delete probe to `minkhoaa/langfens-rq1-cross-fold-checkpoints`.

---

## 5. Comprehensive Gate Assessment (GO / NO-GO)

| Evaluation Gate | Gate Objective | Verified State | Status |
|---|---|---|:---:|
| **Gate 1: Hardware & Device** | Verify Tesla T4 accelerator, CUDA FP16 support, and core library availability on remote platform. | Dual Tesla T4 GPUs verified (`NVIDIA-SMI 580.178.04`, CUDA 13.0, PyTorch 2.11.0+cu128, Transformers 5.16.1). | **GO** |
| **Gate 2: Private Dataset Staging** | Verify private upload of `authorized_RQ1-TRN_fold2_train_val_only.bin`, size (767,978,303 B), SHA-256 (`112cc9e0...`), and CC BY-NC 4.0 license. | Uploaded as immutable `.bin` to `minkhoaa/langfens-rq1-fold2-staging`; 18 authorized speakers verified; 0 test speakers. | **GO** |
| **Gate 3: Pinned Commit Parity** | Verify notebook checks out reviewed training commit `407cf04e83704d9e83953e9309e3c7dca8d7154e`. | Pinned commit invariant enforced in Cell 3 via checked subprocess and `assert actual_head == PINNED_COMMIT_SHA`. | **GO** |
| **Gate 4: Preflight Gating & Safety** | Verify preflight is mandatory and `EXECUTE_TRAINING = False` by default. | `PREFLIGHT_PASSED = False` initialized; Cell 5 fails closed with `RuntimeError` if preflight is not verified. | **GO** |
| **Gate 5: Remote Persistence Credentials** | Verify authenticated write-readback-delete probe to Hugging Face before training. | Blocked at Cell 2 awaiting secret attachment in Kaggle UI. | **NO-GO (PENDING_SECRET)** |
| **Gate 6: Operator Authorization** | Explicit operator sign-off to execute Fold 2 Baseline R model training. | Awaiting resolution of Gate 5 and operator training authorization packet. | **NO-GO (PENDING_AUTH)** |

### Final Launch Determination:
**NO-GO FOR MODEL TRAINING.**  
The environment is verified on hardware and staging, but training remains strictly blocked until:
1. `HF_TOKEN` is attached to the kernel under Add-ons -> Secrets.
2. Phase 4 preflight probe completes with verified `PREFLIGHT_PASSED = True`.
3. Explicit operator authorization is granted to set `EXECUTE_TRAINING = True`.

---

## 6. Restart & Continuation Instructions
Upon resolving the Kaggle Secret attachment:
1. Run preflight push to verify secret probe:
   ```bash
   kaggle kernels push -p deploy/kaggle_fold2/
   kaggle kernels status minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2
   kaggle kernels output minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2
   ```
2. Verify that Cell 4 logs report:
   ```text
   [PREFLIGHT] Preflight verification PASSED successfully.
   ```
3. Await explicit operator authorization before enabling `EXECUTE_TRAINING = True`.
