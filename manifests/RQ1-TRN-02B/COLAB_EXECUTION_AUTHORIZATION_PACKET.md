# LANGFENS — RQ1-TRN-02B: Colab Execution Authorization Packet

**Document ID:** `RQ1-TRN-02B-COLAB-AUTH-PACKET`  
**Status:** READY FOR OPERATOR EXECUTION AUTHORIZATION  
**Branch:** `fix/rq1-trn-02b-preflight-r2`  
**Commit SHA:** `96729fe10153bf143b9c15eada477a0c0ba0c89c`  
**Decision Gate:** `RQ1-TRN-02B-COLAB-AUTHORIZATION: READY`  
**Operational Scope:** Formal pre-execution authorization packet. STRICTLY NO GPU runtime allocation, Colab provisioning, training backward passes, or optimizer updates occurred during the preparation of this packet.

---

## 1. Executive Authorization Summary

This operational authorization packet freezes all cryptographic provenance, execution configurations, storage architectures, guardrail policies, and command sequences for the upcoming Fold 1 Controlled Acoustic Ablation experiment (Control Arm C0 vs. Treatment Arm C1).

All preflight engineering defects, interface contracts, Protocol Z zero-oracle pipelines, deterministic resumption protocols, fail-closed endpoint validations, and dual-arm CTC PER guardrails have been implemented, static-inspected, and regression-verified across **157 CPU tests (0 failures)**.

```
+--------------------------------------------------------------------------------------------------+
| FINAL OPERATIONAL GATE DECISION:                                                                 |
|                                                                                                  |
|   RQ1-TRN-02B-COLAB-AUTHORIZATION: READY                                                         |
|                                                                                                  |
| The software codebase, frozen data packages, cryptographic hashes, and fail-closed safeguards   |
| are verified and fully operational. Execution launch requires the operator's explicit runtime    |
| authorization and allocation of compute units (~15–20 compute units for 7–10 T4 GPU-hours).     |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Execution Provenance and Cryptographic Freeze

### 2.1 Approved Source Code State
- **Repository:** `minkhoaa/Langfens-RQ1-Speech-Research`
- **Approved Feature Branch:** `fix/rq1-trn-02b-preflight-r2`
- **Parent Base Commit SHA:** `6b4cf47fba8cb75ff338540f1dd07f7c8d267d14`
- **Working Tree State:** Clean (verified via `git status` and `git diff --check`).

### 2.2 Frozen Data and Model Artifacts Ledger

| Artifact Role | File Path / Repository Identifier | Expected SHA-256 Digest / Revision | Verification Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **VERIFIED MATCH** |
| **Initial Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **VERIFIED MATCH** |
| **Pretrained Backbone** | `facebook/wav2vec2-base-960h` | Revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | **PINNED REVISION** |
| **Authorized Audio Zip** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **VERIFIED MATCH** |
| **Pinned CMUdict** | `resources/CMUDict/cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | **VERIFIED MATCH** |
| **L2-ARCTIC Prompts** | `resources/L2-ARCTIC/prompts.json` | `87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e` | **VERIFIED MATCH** |

### 2.3 Partition and Speaker Boundaries
- **Fold Index:** 1
- **TRAIN Partition:** Exactly 2,399 utterances across 16 speakers (`EBVS` excluded from Fold 1 train; speakers: `ABA`, `BWC`, `ERMS`, `HJK`, `HKK`, `LXC`, `MBMD`, `NCC`, `PNV`, `RRBI`, `SKA`, `THV`, `TLV`, `TNI`, `TXHC`, `YDCK`).
- **VALIDATION Partition:** Exactly 300 utterances across 2 speakers:
  - `NJS`: Utterances 0 to 149 (150 utterances).
  - `RRBI`: Utterances 150 to 299 (150 utterances).
  - Boundary index: `[150]`.
- **SEALED TEST Partition:** Exactly 900 utterances across 6 speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
  - **Isolation Safeguard:** Zero test utterances or TextGrids exist in `authorized_RQ1-TRN-02A_fold1_train_val_only.zip`. Neither training nor evaluation drivers contain any code path capable of accessing test data.

---

## 3. Experimental Isolation & Arm Specifications

Control Arm C0 and Treatment Arm C1 share an identical training contract, optimizer configuration, initialization, and budget. Only the Binary Prompt Dropout intervention differs:

| Parameter / Contract Field | Control Arm (`C0`) | Treatment Arm (`C1`) | Invariance Status |
|---|:---:|:---:|:---:|
| **Experiment Identifier** | `RQ1-TRN-02B-C0` | `RQ1-TRN-02B-C1` | Isolated Run IDs |
| **Arm Identifier** | `C0` | `C1` | Dedicated Arm Tags |
| **Initial Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | **Identical** (SHA `a2445c8d...`) |
| **Random Seed** | `1337` | `1337` | **Identical** |
| **Binary Prompt Dropout ($p$)** | **`0.00` (None)** | **`0.25` (Full Vector)** | **Isolated Intervention** |
| **Trainable Parameter Groups** | `ctc_head`, `cross_attention`, `binary_head` | `ctc_head`, `cross_attention`, `binary_head` | **Identical** |
| **Backbone State** | Frozen (`requires_grad=False`, `.eval()`) | Frozen (`requires_grad=False`, `.eval()`) | **Identical** |
| **Diagnosis Head State** | Untrained (Zero loss, zero gradients) | Untrained (Zero loss, zero gradients) | **Identical** |
| **Multi-Task Loss Weights** | `ctc: 1.0, binary: 1.0` | `ctc: 1.0, binary: 1.0` | **Identical** |
| **Microbatch Size ($B_{\text{micro}}$)**| `2` | `2` | **Identical** |
| **Gradient Accumulation ($K$)** | `2` (Effective batch size = 4) | `2` (Effective batch size = 4) | **Identical** |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}, \text{decay}=0.01$) | AdamW ($\text{lr}=10^{-3}, \text{decay}=0.01$) | **Identical** |
| **Scheduler** | Cosine with 100-step linear warmup | Cosine with 100-step linear warmup | **Identical** |
| **Maximum Epochs** | `6` epochs | `6` epochs | **Identical** |
| **Maximum Successful Updates** | `3,600` successful updates | `3,600` successful updates | **Identical** |
| **Aggregate Combined Cap** | \multicolumn{2}{c|}{**12 arm-epochs / 7,200 successful updates combined**} | Hard Ceiling |
| **AMP Mode** | FP16 (`GradScaler` init scale 4096) | FP16 (`GradScaler` init scale 4096) | **Identical** |
| **Gradient Clipping** | Max $L_2$ norm = `1.0` | Max $L_2$ norm = `1.0` | **Identical** |

### 3.1 Checkpoint Destination Isolation and Conflict Protection
To prevent accidental reuse, contamination, or clobbering of previous completed, incomplete, or halted checkpoints:
- **Local Output Directories:**
  - Control Arm C0: `checkpoints/RQ1-TRN-02B/C0`
  - Treatment Arm C1: `checkpoints/RQ1-TRN-02B/C1`
- **Google Drive Remote Directories:**
  - Control Arm C0: `/content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C0`
  - Treatment Arm C1: `/content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C1`
- **Fail-Closed Conflict Detection:** The driver verifies that no pre-existing checkpoint artifacts (`last.pt`, `best.pt`, `fixed_budget_checkpoint.pt`, `halted_guardrail.pt`) exist in the output destination before starting a fresh run. If conflicting artifacts are detected and `--resume` is not specified, training halts immediately with `FileExistsError`. Existing checkpoints are never deleted.
- **Execution Run ID Option:** Supports optional `--run-id <identifier>` to namespace artifacts into distinct execution subdirectories if desired.

---

## 4. Operational Safety Review

### 4.1 Colab CLI Readiness and Quota Assessment
- **CLI Binary:** `/usr/bin/colab` (version `0.7.4`).
- **Authentication Readiness:** Verified via read-only command `colab usage`. OAuth2 credentials are valid and active; no interactive `colab auth` required.
- **Active Sessions:** Verified via `colab sessions`: `No active sessions found on server`. Zero idle compute units consumed.
- **Quota Assessment:**
  - Current measured balance: `0.00 compute units`.
  - Estimated experiment requirements:
    - Control Arm C0: 6 epochs $\times$ ~35–45 minutes $\approx$ **3.5–4.5 T4 GPU-hours**.
    - Treatment Arm C1: 6 epochs $\times$ ~35–45 minutes $\approx$ **3.5–4.5 T4 GPU-hours**.
    - Total Combined GPU Compute: **7.0–9.0 T4 GPU-hours**.
  - **Operator Action Required Prior to Launch:** The operator must top-up / assign compute units (~15–20 compute units for standard T4 rates on Google Colab) before executing the launch commands.

### 4.2 Local VM Staging and Verification Strategy
Per AGENTS.md, streaming training audio across Google Drive mounts is strictly prohibited due to I/O latency and network timeouts. Staging workflow:
1. Copy `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` from persistent Google Drive storage to local VM disk (`/content/data/`).
2. Verify SHA-256 digest on local VM disk matches `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`.
3. Train directly from local VM disk.

### 4.3 Atomic Drive Checkpoint Persistence
- At every epoch boundary, `save_checkpoint()` creates a temporary file (`.tmp`), flushes to disk, and executes atomic `os.replace()`.
- Checkpoints are immediately copied to persistent Google Drive.
- Remote SHA-256 checksum is computed over the Drive file and compared with the local file; any discrepancy raises `IOError`.

### 4.4 CTC PER Guardrail and Stop Conditions
Training and evaluation enforce strict fail-closed operational stop conditions:
1. **Manifest / Initial Checkpoint Mismatch:** Checksum differs from frozen ledger.
2. **Backbone Unfreezing / Diagnosis Leakage:** Wav2Vec2 parameters unfreezing or diagnosis head receiving non-zero gradients.
3. **Numerical Instability:** Non-finite loss (NaN/Inf) or 8 consecutive AMP-skipped updates.
4. **CTC PER Guardrail Degradation:** Validation greedy CTC PER exceeding the Reference R anchor (23.64%) by $>+1.0$ absolute percentage point ($>24.64\%$).
   - On violation: saves complete `halted_guardrail.pt`, preserves `last.pt` intact, and halts immediately.
5. **Hard Update Budget:** Halts immediately upon reaching 3,600 successful updates or 6 completed epochs.
6. **Teardown Discipline:** Always execute `colab stop -s <session_name>` promptly after training completes or halts.

---

## 5. Execution Plan: Step-by-Step Commands

*Notice: These commands are prepared and verified for operational use. They have NOT been executed during this gate review.*

### Phase 1: Control Arm C0 Training (Colab GPU)

```bash
# Step 1.1: Verify quota before provisioning
colab usage

# Step 1.2: Provision fresh T4 GPU runtime
colab new -s langfens-rq1-c0 --gpu T4

# Step 1.3: Clone repository and checkout approved branch on Colab VM
colab exec -s langfens-rq1-c0 -- "
  git clone https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git /content/repo &&
  cd /content/repo &&
  git checkout fix/rq1-trn-02b-preflight-r2 &&
  pip install -q -r requirements.txt
"

# Step 1.4: Stage authorized dataset to local VM disk and verify hash
colab exec -s langfens-rq1-c0 -- "
  mkdir -p /content/data &&
  cp /content/drive/MyDrive/KLTN-Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip /content/data/ &&
  echo 'b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9  /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip' | sha256sum -c -
"

# Step 1.5: Execute Control Arm C0 Training
colab exec -s langfens-rq1-c0 -- "
  cd /content/repo &&
  python3 audit_scripts/train_rq1_trn_02b.py \
    --config configs/rq1_trn_02b_c0.yaml \
    --zip-path /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
    --drive-output-dir /content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C0
"

# Step 1.6: Promptly release Colab runtime
colab stop -s langfens-rq1-c0
```

### Phase 2: Control Arm C0 Checkpoint Integrity Verification (Local Laptop CPU)

```bash
# Verify Drive sync, SHA-256 digest, and completed 3,600-update endpoint metadata
python3 -c "
import torch
from pathlib import Path
from audit_scripts.evaluate_rq1_trn_02b import validate_endpoint_readiness

ckpt_path = Path('checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt')
info = validate_endpoint_readiness(ckpt_path, 'C0', allow_incomplete=False)
print('Control C0 Checkpoint Verified:', info)
"
```

### Phase 3: Treatment Arm C1 Training (Colab GPU)

```bash
# Step 3.1: Provision fresh T4 GPU runtime
colab new -s langfens-rq1-c1 --gpu T4

# Step 3.2: Setup repository on fresh VM
colab exec -s langfens-rq1-c1 -- "
  git clone https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git /content/repo &&
  cd /content/repo &&
  git checkout fix/rq1-trn-02b-preflight-r2 &&
  pip install -q -r requirements.txt
"

# Step 3.3: Stage authorized dataset to local VM disk and verify hash
colab exec -s langfens-rq1-c1 -- "
  mkdir -p /content/data &&
  cp /content/drive/MyDrive/KLTN-Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip /content/data/ &&
  echo 'b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9  /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip' | sha256sum -c -
"

# Step 3.4: Execute Treatment Arm C1 Training
colab exec -s langfens-rq1-c1 -- "
  cd /content/repo &&
  python3 audit_scripts/train_rq1_trn_02b.py \
    --config configs/rq1_trn_02b_c1.yaml \
    --zip-path /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
    --drive-output-dir /content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C1
"

# Step 3.5: Promptly release Colab runtime
colab stop -s langfens-rq1-c1
```

### Phase 4: Treatment Arm C1 Checkpoint Integrity Verification (Local Laptop CPU)

```bash
# Verify Drive sync, SHA-256 digest, and completed 3,600-update endpoint metadata
python3 -c "
import torch
from pathlib import Path
from audit_scripts.evaluate_rq1_trn_02b import validate_endpoint_readiness

ckpt_path = Path('checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt')
info = validate_endpoint_readiness(ckpt_path, 'C1', allow_incomplete=False)
print('Treatment C1 Checkpoint Verified:', info)
"
```

### Phase 5: Paired Scientific Evaluation (Local Laptop CPU)

```bash
# Execute full paired counterfactual evaluation across Reference R, Control C0, and Treatment C1
python3 audit_scripts/evaluate_rq1_trn_02b.py \
  --ref-checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
  --c0-checkpoint checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt \
  --c1-checkpoint checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt \
  --manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
  --authorized-zip datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
  --prompts resources/L2-ARCTIC/prompts.json \
  --output-dir manifests/RQ1-TRN-02B/evaluation \
  --device cpu
```

---

## 6. Preflight Regression Test Census Summary

| Test File | Total Tests | Passed | Skipped | Status |
|---|:---:|:---:|:---:|:---:|
| `tests/test_rq1_trn_02b_preflight_r2_safety.py` | 11 | 11 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_corrections_r1.py` | 10 | 10 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_implementation_preflight.py` | 14 | 13 | 1 (GPU backward placeholder) | **PASS** |
| **Combined RQ1-TRN-02B Focused Census** | **35** | **34** | **1** | **PASS** |
| **Full Repository Census** | **161** | **157** | **4** (CUDA-only placeholders) | **PASS** |

---

## 7. Operational Authorization Gate Decision

```text
================================================================================
GATE: RQ1-TRN-02B-COLAB-AUTHORIZATION
STATUS: READY

All source-code repairs, data checksums, fail-closed checkpoint contracts,
Protocol O/Z scoring modules, and dual-arm guardrail policies are validated.

BLOCKER STATUS: Zero code/infrastructure blockers. Operational prerequisite
remains the allocation of Colab compute units (~15–20 compute units) by the
human operator prior to running Phase 1.
================================================================================
```
