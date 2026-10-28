# LANGFENS — RQ1-TRN-02B / 14-Kaggle C0 & C1 Official Training Completion and Paired Remote Mirroring

**Recorded:** 2026-10-29T02:30:00+07:00 (Asia/Ho_Chi_Minh)<br>
**Source examined / pinned:** `032b0b370e53a29910d6560ee330368c83a542b8` on `review/rq1-trn-02b-kaggle-official-runs` (remote branch HEAD verified)<br>
**Process state:** `COMPLETED_TRAINING_BOTH_ARMS; READY_FOR_PAIRED_EVALUATION`<br>
**Benchmark decision:** `WAIVED_BY_OPERATOR_FOR_THIS_EXECUTION_PLAN` (user explicitly waived disposable 50-update benchmark; proceeded directly to official C0 → C1).<br>
**Official Control Arm C0:** `COMPLETED_3600_UPDATES` (6 epochs, 7,200 micro-steps, 3,600 successful updates, 0 AMP skips).<br>
**Official Treatment Arm C1:** `COMPLETED_3600_UPDATES` (6 epochs, 7,200 micro-steps, 3,600 successful updates, 0 AMP skips).<br>
**Kaggle C0 Kernel:** `minkhoaa/langfens-rq1-c0-t4x2` (Version 9, status: `KernelWorkerStatus.COMPLETE`).<br>
**Kaggle C1 Kernel:** `minkhoaa/langfens-rq1-c1-t4x2` (Version 1, status: `KernelWorkerStatus.COMPLETE`).<br>
**Kaggle Dataset:** `minkhoaa/langfens-rq1-fold1-staging` (Version 3, status: `ready`, private under CC BY-NC 4.0).<br>
**Remote Persistent Store:** `minkhoaa/langfens-rq1-trn-02b-checkpoints` (Private Hugging Face repository; all epoch checkpoints mirrored with bit-identical SHA-256 readback).<br>
**TEST Partition Isolation:** Strictly sealed; zero test audio or annotations accessed, staged, or inspected.<br>

---

## 1. Executive Summary

Following operator SMS phone verification on Kaggle, physical Nvidia Tesla T4 GPU accelerators were unlocked and provisioned. Both experimental arms of the controlled acoustic conditioning ablation were successfully executed to their mandatory matched-budget endpoint:

1. **Control Arm C0 ($p=0.00$):**
   - Cold reset from Baseline R `best.pt` (`a2445c8d...`).
   - Completed strictly 6 epochs / 3,600 successful optimizer updates (0 skips).
   - Final validation AP: **0.310348** (Peak AP: **0.312170** at Epoch 4).
   - Final validation CTC PER: **22.09%** (improved from anchor 23.64%; zero guardrail trips).
   - All checkpoints (`best.pt`, `last.pt`, `fixed_budget_checkpoint.pt`, `history.json`) atomically mirrored per-epoch to Hugging Face with verified SHA-256 readback.
2. **Treatment Arm C1 ($p=0.25$ whole-phone prompt dropout):**
   - Cold reset from the **SAME Baseline R `best.pt`** (`a2445c8d...`), completely independent of C0.
   - Completed strictly 6 epochs / 3,600 successful optimizer updates (0 skips).
   - Final validation AP: **0.302332** (Peak AP: **0.320398** at Epoch 1).
   - Final validation CTC PER: **22.12%** (improved from anchor 23.64%; zero guardrail trips).
   - All checkpoints atomically mirrored per-epoch to Hugging Face with verified SHA-256 readback.

Total Kaggle GPU resource consumption: **0.53 hours** across both arms combined (29.47h remaining of 30.00h weekly quota).

---

## 2. Experimental Configuration & Scientific Integrity

Both arms strictly observed identical hyperparameters, execution environments, and compute budgets:

| Parameter / Contract | Control Arm C0 | Treatment Arm C1 | Invariant Check |
|---|:---:|:---:|:---:|
| **Platform** | Kaggle Tesla T4 × 2 | Kaggle Tesla T4 × 2 | Matched Hardware |
| **Precision** | CUDA FP16 AMP (`init_scale=4096.0`) | CUDA FP16 AMP (`init_scale=4096.0`) | Matched Numerical Regime |
| **Backbone Model** | `facebook/wav2vec2-base-960h` (rev `22aad52d...`) | `facebook/wav2vec2-base-960h` (rev `22aad52d...`) | Frozen Eval Mode |
| **Initial Weights** | Baseline R `best.pt` (`a2445c8d...`) | Baseline R `best.pt` (`a2445c8d...`) | **Same Initialization** |
| **Random Seed** | 1337 | 1337 | Matched RNG Stream |
| **Batch Size & Accumulation** | Batch 2, Accumulation 2 (Effective 4) | Batch 2, Accumulation 2 (Effective 4) | Matched Batch Contract |
| **Optimizer & Schedule** | AdamW (`lr=1e-3`, `wd=1e-2`), Cosine (100 warmup) | AdamW (`lr=1e-3`, `wd=1e-2`), Cosine (100 warmup) | Matched Schedule |
| **Intervention Probability** | **$p = 0.00$** | **$p = 0.25$** | **Ablation Contrast Variable** |
| **Intervention Target** | N/A | Whole-phone binary prompt embeddings | CTC path unperturbed |
| **Training Budget** | 6 epochs / 3,600 successful updates | 6 epochs / 3,600 successful updates | Matched Exposure |
| **Guardrails** | CTC PER degradation $\le +1.0\%$ | CTC PER degradation $\le +1.0\%$ | Zero Trips |

---

## 3. Training Dynamics & Epoch Ledger

### 3.1 Control Arm C0 Ledger (`minkhoaa/langfens-rq1-c0-t4x2`)

| Epoch | Micro Steps | Updates | Skips | Train CTC Loss | Train Binary Loss | Val Binary AP | Val CTC PER | Wall Time | Remote Mirror Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1,200 | 600 | 0 | 0.796924 | 0.363275 | 0.310654 | 0.234490 | 98.8s | `SYNCED` (`last.pt`, `best.pt`) |
| **2** | 2,400 | 1,200 | 0 | 0.780932 | 0.364137 | 0.311215 | 0.230341 | 77.0s | `SYNCED` (`last.pt`, `best.pt`) |
| **3** | 3,600 | 1,800 | 0 | 0.767545 | 0.363359 | 0.306909 | 0.227305 | 76.8s | `SYNCED` (`last.pt`) |
| **4** | 4,800 | 2,400 | 0 | 0.757461 | 0.361106 | **0.312170** | 0.222650 | 76.5s | `SYNCED` (`last.pt`, `best.pt`) |
| **5** | 6,000 | 3,000 | 0 | 0.750882 | 0.359187 | 0.309600 | 0.221131 | 77.2s | `SYNCED` (`last.pt`) |
| **6** | 7,200 | **3,600** | 0 | 0.747726 | 0.357844 | 0.310348 | **0.220929** | 76.9s | `SYNCED` (`last.pt`, `fixed_budget_checkpoint.pt`, `history.json`) |

### 3.2 Treatment Arm C1 Ledger (`minkhoaa/langfens-rq1-c1-t4x2`)

| Epoch | Micro Steps | Updates | Skips | Train CTC Loss | Train Binary Loss | Val Binary AP | Val CTC PER | Wall Time | Remote Mirror Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1,200 | 600 | 0 | 0.796986 | 0.373396 | **0.320398** | 0.234895 | 98.1s | `SYNCED` (`last.pt`, `best.pt`) |
| **2** | 2,400 | 1,200 | 0 | 0.781039 | 0.372159 | 0.310931 | 0.230037 | 75.8s | `SYNCED` (`last.pt`) |
| **3** | 3,600 | 1,800 | 0 | 0.767647 | 0.371955 | 0.302134 | 0.227305 | 75.9s | `SYNCED` (`last.pt`) |
| **4** | 4,800 | 2,400 | 0 | 0.757555 | 0.369019 | 0.301170 | 0.222953 | 75.4s | `SYNCED` (`last.pt`) |
| **5** | 6,000 | 3,000 | 0 | 0.750949 | 0.367147 | 0.302629 | 0.221435 | 76.0s | `SYNCED` (`last.pt`) |
| **6** | 7,200 | **3,600** | 0 | 0.747807 | 0.365203 | 0.302332 | **0.221233** | 75.2s | `SYNCED` (`last.pt`, `fixed_budget_checkpoint.pt`, `history.json`) |

---

## 4. Cryptographic Artifacts & Remote Mirror Ledger

All deliverable artifacts are mirrored in private persistent repository `minkhoaa/langfens-rq1-trn-02b-checkpoints` on Hugging Face:

| Artifact Path | File Size | Cryptographic SHA-256 Digest | Status |
|---|:---:|---|:---:|
| `C0/best.pt` | 412,376,647 B | `4689170265de6fd6d9e06608afc3b307f62fe0f62939e657a6a68bc28f2ab4ba` | **VERIFIED MATCH** |
| `C0/last.pt` | 412,377,159 B | `bddce015f18a2eedd3b6e0c1d2b9ec13530c9644b4380d32309d87f32b21338f` | **VERIFIED MATCH** |
| `C0/fixed_budget_checkpoint.pt` | 412,383,309 B | `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e` | **VERIFIED MATCH** |
| `C0/history.json` | 4,972 B | `1904132a24f7515e8c2cc07dbf3e1957d077ad7df5c56dadf80223c534bfe6d2` | **VERIFIED MATCH** |
| `C1/best.pt` | 412,375,943 B | `1d643b489fc7e0c1ef39fa1d35dbe25c3898749ee3b858b990ec01f46396b89a` | **VERIFIED MATCH** |
| `C1/last.pt` | 412,377,159 B | `ce583d8851078504687c5dd38f5f17e43fe2c1fa6b746f1ed2d031516de67e9b` | **VERIFIED MATCH** |
| `C1/fixed_budget_checkpoint.pt` | 412,383,309 B | `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4` | **VERIFIED MATCH** |
| `C1/history.json` | 4,974 B | `3f30812ca03c0cf70c62c2e643cca03e3bd6b6031a41806ad2a7f7db36ea3f15` | **VERIFIED MATCH** |

---

## 5. Primary Endpoint Verification & Invariants

Both `fixed_budget_checkpoint.pt` artifacts satisfy all nine mandatory endpoint invariants verified by `classify_kaggle_training_endpoint()`:
1. Top-level `successful_updates == 3600`: **PASS** (C0: 3,600, C1: 3,600).
2. Metadata `successful_updates == 3600`: **PASS**.
3. `primary_3600_updates_completed is True`: **PASS**.
4. `primary_endpoint_reached is True`: **PASS**.
5. `endpoint_status == "COMPLETED_3600_UPDATES"`: **PASS**.
6. `guardrail_violation is False`: **PASS** (Both arms maintained CTC PER well within the 24.64% trip ceiling).
7. Checkpoint `arm` matches expected arm: **PASS** (C0 verified as C0, C1 verified as C1).
8. Critical provenance digests match ledger: **PASS** (`split_sha256: 7d6bfdf6...`, `initial_checkpoint_sha256: a2445c8d...`).
9. Non-empty model, optimizer, scheduler, and scaler state dicts present: **PASS**.

---

## 6. Next Actions & Evaluation Protocol

With both C0 and C1 completed and durably stored:
1. Proceed to standalone CPU paired counterfactual evaluation via `audit_scripts/evaluate_rq1_trn_02b.py`.
2. Compute matched-exposure counterfactual contrast:
   $$\Delta \text{AP}_{\text{acoustic}} = \text{AP}(C1) - \text{AP}(C0)$$
   $$\Delta \text{PER} = \text{PER}(C1) - \text{PER}(C0)$$
3. Execute evaluation under both Protocol O (TextGrid alignment) and Protocol Z (Zero-Oracle G2P alignment).
4. Keep official TEST strictly sealed.
