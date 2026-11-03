# P-SUP B-CONTROL Artifact Recovery and Independent Verification Report

**Subagent Identity:** Agent B (Independent Recovery Specialist for B-Control Matched Control)  
**Execution Timestamp:** 2026-11-04T01:27:00+07:00 (Asia/Ho_Chi_Minh)  
**Worktree:** `/home/khoa/Projects/KLTN/psup-recover-b-control-v1`  
**Branch:** `review/psup-recover-b-control-v1`  
**Base Commit:** `40bc3770776ee1c99462d44101fdd145602b76f5`  
**Arm Under Verification:** `B-CONTROL` (Matched Binary Control, objective: `binary_collapsed`)  
**Overall Disposition:** **VERIFIED (PASS)** — Bit-for-bit cryptographic recovery, complete payload state validation, and strict boundary enforcement confirmed across all target artifacts.

---

## 1. Executive Summary

Agent B has independently recovered, cryptographically validated, and structurally inspected all training and evaluation artifacts for the RQ1 P-SUP Matched Control arm (`B-CONTROL`, `binary_collapsed`). 

Key findings:
1. **Remote Source Integrity & Revision Accounting:**
   - Both Kaggle kernel execution (`minkhoaa/langfens-rq1-psup-b-control-t4x2`, status `COMPLETE`) and Hugging Face private mirror (`minkhoaa/langfens-rq1-cross-fold-checkpoints`, prefix `psup/b_control/`, private=True) were accessed using pre-existing authentication without credential exposure.
   - Pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d` remains strictly authentic and verified for `psup/b_control/`.
   - The repository HEAD advanced (`d8806b9...` and `6d1ec32...`) as the parent orchestrator mirrored archival benchmark predictions under `benchmarks/psup-v1/`; the `psup/b_control/` subtree was verified bit-identical and unchanged across these commits.
2. **Download Operational Accounting:**
   - An early bulk download attempt from Kaggle output was cleanly terminated via SIGINT by the parent orchestrator (PID 183900) to prevent downloading the whole cloned repository tree (`repo/`).
   - Bounded retrieval (`--file-pattern ".*\.(pt|json)$"` and pinned HF downloads) was subsequently used. This interrupted download is an accounted operational download boundary filter, NOT an artifact integrity failure.
3. **Cryptographic Parity:** All core artifacts (`fixed_budget_3600.pt`, `best.pt`, `training_summary.json`, `history.json`) exhibit 100% bit-for-bit SHA-256 match between Hugging Face LFS pointers, Kaggle kernel outputs, local pre-existing candidates, and recovered scratch files.
4. **Missing Artifact Recovered:** The checkpoint `last.pt` (SHA-256 `aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5`), which was absent from the local candidate directory `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control`, was successfully recovered from Hugging Face and Kaggle.
5. **Pre-Existing Candidates Preserved:** Local candidate files in `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control` were inspected read-only and preserved completely unmodified.
6. **Payload State Completeness:** Safe CPU tensor inspection (`weights_only=True` allowlist) verified 266,667 head parameters (all finite, 0 NaN/Inf), frozen `facebook/wav2vec2-base-960h` backbone, complete AdamW optimizer states (16 tracked parameter tensors), cosine scheduler state, GradScaler state (scale=8192.0), and full 4-component RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA for 2 devices).
7. **Update Accounting:**
   - **This Process:** 0 updates, 0 backward passes, 0 optimizer steps, 0 model mutations (strictly forensic recovery).
   - **Historical Training:** Exactly 3,600 successful updates, 0 skipped updates, 7,200 micro-steps (effective batch size 4), completed in 561.20s on Kaggle Tesla T4 x 2.
   - **Endpoints Verified:** `best.pt` was captured at Epoch 4 (update 2,250, SELECT AP 0.467607); `fixed_budget_3600.pt` was captured at Epoch 7 (update 3,600, SELECT AP 0.464117, `is_fixed_budget_endpoint=True`).
8. **Semantic Disclosure (Subtype Unidentifiability):** Because `binary_collapsed` trains exclusively against binary error labels $(y \ne 0)$ via collapsed logit $\text{logsumexp}(z_1, z_2) - z_0$, output logits $z_1$ and $z_2$ receive identical gradient updates. The model does NOT learn distinct omission or substitution probabilities. Consequently, `s_omit`, `s_sub`, `probabilities`, and `omission_ap_dedicated` are strictly `null` (unsupported). Omission AP can only be evaluated in legacy mode using generic binary error scores (`s_error`).

---

## 2. Remote Sources & Independent Pinning

### 2.1 Hugging Face Remote Mirror
- **Repository Slug:** `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- **Repository Type:** `model`
- **Privacy Setting:** `is_private = True` (authenticated API probe confirms repository is strictly private under CC BY-NC 4.0)
- **Pinned Commit SHA for `psup/b_control/`:** `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
- **Current Repository HEAD:** `6d1ec3217c94730cc590b935c45e523dc5259f39` (incorporating parent benchmark mirrors under `benchmarks/psup-v1/`). Subtree comparison between `4aa9f2...` and `6d1ec3...` confirms `psup/b_control/` is 100% bit-identical.
- **Repository Path Prefix:** `psup/b_control/`
- **Commit History for `psup/b_control/`:**
  - `4aa9f2a68f0709889eacacf170b6446d3c45c73d` (2026-11-03 16:08:03 UTC): `Upload psup/b_control/best.pt with huggingface_hub`
  - `3bbf2c21db6c4f99605383fbc55d106bffaf5d62` (2026-11-03 16:07:53 UTC): `Upload psup/b_control/training_summary.json with huggingface_hub`
  - `9893c61ae10795a91590c208ed0493dddc0efca7` (2026-11-03 16:07:51 UTC): `Upload psup/b_control/last.pt with huggingface_hub`
  - `0ddee4a3b65adeffea2fc10d4c988ef65200ad05` (2026-11-03 16:07:35 UTC): `Upload psup/b_control/fixed_budget_checkpoint.pt with huggingface_hub`
  - `f4d8e33f997f7910b406c9689b33c6e23b9d5f1b` (2026-11-03 16:07:33 UTC): `Upload psup/b_control/fixed_budget_3600.pt with huggingface_hub`
  - `0ba9369b4f56ce7d08972b0af4454a4ec6877535` (2026-11-03 16:06:54 UTC): `Upload psup/b_control/history.json with huggingface_hub`
- **LFS Pointer Verification:** All `.pt` model files are tracked via Git LFS with 132-byte pointer files referencing authentic SHA-256 blob payloads.

### 2.2 Kaggle Platform Run
- **Kernel Slug:** `minkhoaa/langfens-rq1-psup-b-control-t4x2`
- **Runtime Hardware:** `NvidiaTeslaT4` (Dual Tesla T4 GPUs)
- **Kernel Worker Status:** `KernelWorkerStatus.COMPLETE`
- **Source Git Commit Pinned in Notebook:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`
- **Execution Command from Verified Logs:**
  ```bash
  /usr/bin/python3 audit_scripts/train_p_sup.py \
    --arm control \
    --objective binary_collapsed \
    --split-manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
    --zip-path /kaggle/input/datasets/minkhoaa/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.bin \
    --output-dir /kaggle/working/checkpoints/RQ1-P-SUP/b_control \
    --seed 1337 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --warmup-updates 100 \
    --max-updates 3600 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --max-epochs 25 \
    --train-partition fit \
    --select-partition selection \
    --cal-partition calibration \
    --val-partition val \
    --remote-execution \
    --remote-mirror-target minkhoaa/langfens-rq1-cross-fold-checkpoints \
    --remote-mirror-backend huggingface
  ```
- **Recovered Execution Log:** `langfens-rq1-psup-b-control-t4x2.log` (380,708 bytes, SHA-256 `24f1a2a30211b29d1590fba830a8c314c3a46657db57bef50cf333afc25d8936`).
- **Recovered Source Notebook:** `langfens-rq1-psup-b-control-t4x2.ipynb`.

---

## 3. Cryptographic Verification & Parity Matrix

All recovered artifacts were staged into local scratch at `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/` (retained strictly outside Git). 

| File Name | Size (Bytes) | SHA-256 Checksum | HF LFS Match | Kaggle Match | Pre-Existing Local Match | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed_budget_3600.pt` | 3,241,295 | `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` | MATCH | MATCH | MATCH | **VERIFIED** |
| `fixed_budget_checkpoint.pt` | 3,241,295 | `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` | MATCH | MATCH | N/A (Alias) | **VERIFIED** |
| `best.pt` | 3,239,450 | `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f` | MATCH | MATCH | MATCH | **VERIFIED** |
| `last.pt` | 3,239,898 | `aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5` | MATCH | MATCH | *Absent locally* | **RECOVERED** |
| `training_summary.json` | 3,339 | `fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f` | MATCH | MATCH | MATCH | **VERIFIED** |
| `history.json` | 2,896 | `dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6` | MATCH | MATCH | MATCH | **VERIFIED** |
| `langfens-rq1-psup-b-control-t4x2.log` | 380,708 | `24f1a2a30211b29d1590fba830a8c314c3a46657db57bef50cf333afc25d8936` | N/A | MATCH (API) | N/A | **RECOVERED** |

### Pre-Existing Local Candidate Preservation
- Candidate Directory: `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control/`
- All 4 pre-existing files (`best.pt`, `fixed_budget_3600.pt`, `history.json`, `training_summary.json`) were checked read-only; timestamps (2026-11-04 00:32:16–20 +0700) and byte contents remain untouched.
- `last.pt` was confirmed missing from the pre-existing local directory; it has been recovered into scratch without modifying the preserved candidate directory.

---

## 4. Deep CPU Tensor & State Inspection

Checkpoints were audited on CPU using PyTorch 2.6 safe loading with allowlisted NumPy reconstruction globals (`weights_only=True` compliant).

### 4.1 Architecture and Head Parameters
The head module `CompactOperationHead` contains exactly 16 parameter tensors totaling **266,667 parameters**, matching the parameter budget of the treatment arm `T-OP`:
- `acoustic_projection.weight`: `[128, 768]` (98,304 parameters, float32, finite)
- `acoustic_projection.bias`: `[128]` (128 parameters, float32, finite)
- `phone_embedding.weight`: `[40, 128]` (5,120 parameters, float32, finite)
- `position_embedding.weight`: `[256, 128]` (32,768 parameters, float32, finite)
- `attention.in_proj_weight`: `[384, 128]` (49,152 parameters, float32, finite)
- `attention.in_proj_bias`: `[384]` (384 parameters, float32, finite)
- `attention.out_proj.weight`: `[128, 128]` (16,384 parameters, float32, finite)
- `attention.out_proj.bias`: `[128]` (128 parameters, float32, finite)
- `acoustic_norm.weight`: `[128]` (128 parameters, float32, finite)
- `acoustic_norm.bias`: `[128]` (128 parameters, float32, finite)
- `operation_head.0.weight`: `[128, 256]` (32,768 parameters, float32, finite)
- `operation_head.0.bias`: `[128]` (128 parameters, float32, finite)
- `operation_head.3.weight`: `[3, 128]` (384 parameters, float32, finite)
- `operation_head.3.bias`: `[3]` (3 parameters, float32, finite)
- `ctc_head.weight`: `[40, 768]` (30,720 parameters, float32, finite)
- `ctc_head.bias`: `[40]` (40 parameters, float32, finite)
- **Finiteness Audit:** 100% finite across all weights and biases (0 NaN, 0 Inf).
- **Backbone State:** Frozen `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`), unmutated.

### 4.2 State Completeness Audit
1. **Optimizer (AdamW):**
   - 1 parameter group, base learning rate 0.001, weight decay 0.01.
   - All 16 head parameter tensors have active momentum and variance tracking in `optimizer['state']`.
   - Current learning rate at update 3,600: `0.0008699219550575954` (exact cosine decay matching schedule).
   - Current learning rate at update 2,250: `0.0009494996720699363`.
2. **Scheduler:**
   - Cosine annealing with 100 warmup updates over 15,000 horizon.
   - `last_epoch` = 3,600 (fixed budget) / 2,250 (best selection).
   - `_step_count` = 3,601 / 2,251.
3. **AMP GradScaler:**
   - `scale`: `8192.0`
   - `growth_factor`: `2.0`, `backoff_factor`: `0.5`, `growth_interval`: `2000`
   - `_growth_tracker`: `1600` (at update 3,600) / `250` (at update 2,250)
4. **RNG State Completeness:**
   - Python `random`: full state tuple (3 elements)
   - NumPy: full state tuple (5 elements, MT19937)
   - PyTorch CPU: full state uint8 tensor (5,056 bytes)
   - PyTorch CUDA: 2 tensors (for dual T4 GPUs on Kaggle)

---

## 5. Update Accounting & Endpoint Verification

### 5.1 Process Boundaries (This Process vs Historical Training)
- **Updates executed in this subagent session:** **0 updates, 0 backward passes, 0 optimizer steps**.
- **Historical updates verified from in-tensor states and training logs:** Exactly **3,600 successful updates, 0 skipped updates**.

### 5.2 Training Epoch and Update History
The training run processed 8 epochs (Epoch 0 through Epoch 7), with 450 optimizer updates per epoch (effective batch size 4 = batch_size 2 $\times$ grad_accum 2):

| Epoch | Successful Updates | Skipped AMP | Attempts | Selection AP (SELECT) | Blank Fraction | Realized PER | Avg Task Loss | Avg CTC Loss | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 450 | 0 | 450 | 0.436842 | 0.842210 | 0.333749 | 0.375181 | 3.300258 | 0.0009986 |
| 1 | 900 | 0 | 900 | 0.439099 | 0.839453 | 0.291550 | 0.346947 | 1.045320 | 0.0009929 |
| 2 | 1,350 | 0 | 1,350 | 0.445090 | 0.838136 | 0.272841 | 0.345579 | 0.956034 | 0.0009827 |
| 3 | 1,800 | 0 | 1,800 | 0.465816 | 0.838809 | 0.257250 | 0.343967 | 0.904533 | 0.0009682 |
| 4 | **2,250** | 0 | 2,250 | **0.467607** | 0.840311 | 0.256522 | 0.342679 | 0.868264 | 0.0009495 |
| 5 | 2,700 | 0 | 2,700 | 0.455069 | 0.838840 | 0.247272 | 0.340464 | 0.841710 | 0.0009267 |
| 6 | 3,150 | 0 | 3,150 | 0.463814 | 0.838947 | 0.235838 | 0.340215 | 0.821270 | 0.0009001 |
| 7 | **3,600** | 0 | 3,600 | 0.464117 | 0.840893 | 0.240931 | 0.340399 | 0.804108 | 0.0008699 |

### 5.3 Endpoint Characterization
- **Best Selection Checkpoint (`best.pt`):**
  - Actual updates: **2,250** (Epoch 4)
  - Micro-steps: 4,500
  - Selection AP: **0.467607** (peak across training)
  - Early stopping patience counter: 0
  - Checkpoint flag: `is_fixed_budget_endpoint = False`
- **Fixed Budget Checkpoint (`fixed_budget_3600.pt`):**
  - Actual updates: **3,600** (Epoch 7)
  - Micro-steps: 7,200
  - Selection AP: **0.464117**
  - Early stopping patience counter: 2 (epochs 5 and 6 did not improve on epoch 4)
  - Checkpoint flag: `is_fixed_budget_endpoint = True`
- **Last Checkpoint (`last.pt`):**
  - Actual updates: **3,600** (Epoch 7)
  - Micro-steps: 7,200
  - Checkpoint flag: `is_fixed_budget_endpoint = False`
  - Bit-identical weights to `fixed_budget_3600.pt` except for the metadata endpoint flag.

---

## 6. Dataset Identity & Boundary Sealing Audit

### 6.1 Cryptographic Data Provenance
- **Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`  
  SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (Verified authentic).
- **Authorized Audio Archive:** `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` / zip  
  SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (Verified authentic).
- **Evaluation Contract V1:**  
  SHA-256: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` (Verified authentic).

### 6.2 Partition Isolation and Sealing Verification
The checkpoint metadata and Kaggle execution logs confirm exact partition boundaries:
- **FIT Partition (Training):** 1,799 utterances across 12 speakers: `BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA`.
- **SELECT Partition (Selection/best.pt):** 300 utterances across 2 speakers: `MBMPS, TLV`.
- **CAL Partition (Calibration):** 300 utterances across 2 speakers: `NCC, THV`.
- **VAL Partition (Evaluation):** 300 utterances across 2 speakers: `NJS, RRBI` (Strictly preserved unseen during training and checkpoint selection).
- **TEST Partition (Sealed):** 900 utterances across 6 speakers: `ABA, ASI, EBVS, HJK, LXC, PNV`.
- **Sealing Audit Result:** **PASS**. Zero test partition utterances were staged, decompressed, or trained upon.

---

## 7. Explicit Semantic Disclosure: Subtype Unidentifiability

A fundamental mathematical distinction separates `B-CONTROL` (`binary_collapsed`) from `T-OP` (`operation_ce`):

### 7.1 Forward Algebra Under `binary_collapsed`
In `src/langfens_prep/p_sup/objectives.py`:
$$\text{collapsed\_binary\_logit}(z) = \log\left(\sum_{k=1}^2 \exp(z_k)\right) - z_0 = \text{logsumexp}(z_1, z_2) - z_0$$
$$\mathcal{L}_{\text{task}} = \text{BCEWithLogits}(\text{collapsed\_binary\_logit}(z), \mathbb{I}(y \ne 0))$$

### 7.2 Consequences for Subtype Scoring
1. **Identical Gradients:** The derivative of $\mathcal{L}_{\text{task}}$ with respect to $z_1$ (substitution) and $z_2$ (deletion) differs only by the relative softmax weights $\frac{\exp(z_1)}{\exp(z_1)+\exp(z_2)}$. Since both represent positive errors ($\mathbb{I}(y \ne 0) = 1$), the supervision provides zero signal to separate substitutions from deletions.
2. **Unsupported Subtype Scores:** In `src/langfens_prep/p_sup/runtime.py`:
   - `s_error` = $p_1 + p_2$ (overall binary error score)
   - `s_omit` = `None`
   - `s_sub` = `None`
   - `probabilities` = `None`
   - `score_semantics` = `"binary collapse of same three logits; subtype unidentified/null; not calibrated"`
3. **Dedicated Omission AP is Null:** In `manifests/RQ1-P-SUP/evaluation/FINAL_EVALUATION_SUMMARY.json`, `omission_ap_dedicated` for B-CONTROL is explicitly recorded as `null`.
4. **Legacy Metric Evaluation:** The reported Omission AP (`0.038033` for fixed budget, `0.034692` for best selection) and Substitution AP (`0.311232` for fixed budget, `0.319735` for best selection) are computed exclusively using generic `s_error` evaluated on the respective label subsets (deletion vs correct, substitution vs correct). This mirrors the historical Baseline R E0 behavior and cannot measure dedicated subtype discrimination.

---

## 8. Preserved Worktree Receipts & Artifact References

Under the assigned worktree (`/home/khoa/Projects/KLTN/psup-recover-b-control-v1`), the following documentation receipts have been committed to disk:
1. `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/RECOVERY_RECEIPT.json`: Full machine-readable receipt covering HF commit SHAs, Kaggle execution parameters, payload update counters, and contract hashes.
2. `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/CHECKPOINT_PAYLOAD_AUDIT.json`: Detailed tensor-level inventory of all 16 head tensors, parameter counts, optimizer state, scheduler state, GradScaler state, and RNG state.
3. `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/SHA256SUMS`: Checksums for all 7 recovered files.
4. `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/PROCESS_CHECKPOINT.md`: Standardized research process checkpoint adhering to `AGENTS.md`.

---

## 9. Blockers, Commands Run, and Safe Restart

### 9.1 Blockers
- **Zero Blockers:** All remote sources, authentication keys, Hugging Face LFS payloads, Kaggle logs, and local candidate directories were accessible and verified without operational impediments.

### 9.2 Commands Executed During Verification
```bash
# 1. Verification of git status and branch
git status && git branch -a

# 2. Kaggle kernel status check
kaggle kernels status minkhoaa/langfens-rq1-psup-b-control-t4x2

# 3. Authenticated Hugging Face probe & commit inspection
python3 -c "from huggingface_hub import HfApi; api = HfApi(); ..."

# 4. Bounded retrieval of Kaggle kernel logs, notebook, and outputs
kaggle kernels logs minkhoaa/langfens-rq1-psup-b-control-t4x2
kaggle kernels pull minkhoaa/langfens-rq1-psup-b-control-t4x2 -p /home/khoa/.cache/langfens/psup-benchmark-v1/b_control/kaggle_source
kaggle kernels output -p /home/khoa/.cache/langfens/psup-benchmark-v1/b_control/kaggle minkhoaa/langfens-rq1-psup-b-control-t4x2 --file-pattern ".*\.(pt|json)$" --force

# 5. Hugging Face authenticated model download to local scratch
python3 -c "from huggingface_hub import hf_hub_download; ..."

# 6. Cryptographic SHA-256 verification across all sources
python3 -c "import hashlib, glob; ..."

# 7. Safe CPU tensor inspection under weights_only=True allowlist
python3 -c "import torch; with torch.serialization.safe_globals(...): torch.load(...)"
```

### 9.3 Safe Restart Instructions
1. For any downstream evaluation or comparative analysis, point directly to the recovered scratch directory:
   `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/`
2. Fixed budget checkpoint: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/fixed_budget_3600.pt`
3. Best selection checkpoint: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/best.pt`
4. The coordinator may ingest receipts from `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/` to update the global benchmark index.
