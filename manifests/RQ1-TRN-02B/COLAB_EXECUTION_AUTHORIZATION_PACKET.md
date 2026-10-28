# LANGFENS — RQ1-TRN-02B: Colab Execution Authorization Packet (Revision R1)

**Document ID:** `RQ1-TRN-02B-COLAB-AUTH-PACKET`<br>
**Revision:** `R1 (Operational Corrections & Provenance Reconciliation)`<br>
**Status:** `CORRECTED_PENDING_RUNTIME_VERIFICATION`<br>
**Current Review Branch:** `fix/rq1-trn-02b-colab-auth-r1`<br>
**Parent Documentation Base Commit:** `aa6415a8d3a073a3550f209a53e1baa44461ce97`<br>
**Historical Preflight R2 Base Commit:** `8486cbb3596a0f5d2603dcdf46fd891b5c849840`<br>
**Decision Gate:** `RQ1-TRN-02B-COLAB-AUTHORIZATION: CORRECTED_PENDING_RUNTIME_VERIFICATION`<br>
**Operational Scope:** Formal pre-execution authorization packet. STRICTLY NO GPU runtime allocation, Colab provisioning, training backward passes, or optimizer updates occurred during the preparation of this packet.

---

## 1. Executive Authorization Summary

This operational authorization packet establishes all cryptographic provenance, execution configurations, storage architectures, fail-closed guardrails, and verified Google Colab CLI command sequences for the upcoming Fold 1 Controlled Acoustic Conditioning Ablation experiment (Control Arm C0 vs. Treatment Arm C1).

Revision R1 resolves four operational discrepancies identified during pre-execution audit:
1. **Scientific Provenance Correction:** Corrects the Fold 1 TRAIN speaker partition list in documentation (removing erroneously listed TEST and VAL speakers) while proving strict speaker-disjointness across TRAIN, VAL, and SEALED TEST.
2. **Commit Provenance Reconciliation:** Disentangles baseline, preflight, registry, and operational correction commits, eliminating the phantom uncommitted SHA `96729fe...` and establishing the requirement that training check out one immutable Git commit.
3. **Colab CLI Command Repair:** Replaces unsupported shell pipes (`colab exec -s S -- "..."`) with verified non-interactive Python script execution (`colab exec -s S -f script.py --env ... --timeout ...`), verified against Colab CLI v0.7.4.
4. **Google Drive Staging & Output Hierarchy:** Reconciles the authoritative Google Drive folder and file IDs under root `KLTN-Dataset` (`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`), defines non-interactive VM disk staging with SHA-256 re-hashing, and documents the prerequisite human-assisted `colab drivemount` step.

```
+--------------------------------------------------------------------------------------------------+
| FINAL OPERATIONAL GATE DECISION:                                                                 |
|                                                                                                  |
|   RQ1-TRN-02B-COLAB-AUTHORIZATION: CORRECTED_PENDING_RUNTIME_VERIFICATION                        |
|                                                                                                  |
| Codebase, staging scripts, cryptographic ledgers, and unit tests (175 passed) are verified.      |
| Full execution launch is BLOCKED pending:                                                       |
|   1. Operator top-up / allocation of Colab compute units (~15–20 compute units).                 |
|   2. Interactive Google Drive mount (colab drivemount -s <session>) by operator.                 |
|   3. Live runtime staging and storage write verification on authorized Colab VM disk.            |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Cryptographic Provenance and Split Ledger

### 2.1 Git Commit Provenance Ledger
To ensure full scientific reproducibility, all project sources and deliverables are indexed by exact, verified Git commits:

| Provenance Level | Commit SHA | Role / Description |
|---|---|---|
| **Frozen Scientific Baseline (Run)** | `caa56784b1dff6e1d2b8b93998b3f81e3a1f4861` | Fold 1 baseline training execution on Colab T4. |
| **Frozen Scientific Baseline (Eval)**| `c4a658f8b8e0b656758416d82c0be42323c21a10` | Protocol Z zero-oracle numerical closure and CPU replay. |
| **Baseline R Archival & Release** | `a200b73b90a137316817bc42df0ba6e0db2c24c5` | RQ1-ARTIFACT-01 Drive archival and private Hugging Face release. |
| **Preflight R2 Implementation** | `8486cbb3596a0f5d2603dcdf46fd891b5c849840` | Preflight safety closure, unique run-ID, fail-closed guards. |
| **Historical Packet Header** | `8486cbb3596a0f5d2603dcdf46fd891b5c849840` | Erratum: phantom SHA `96729fe...` corrected to `8486cbb`. |
| **Checkpoint Registry Base** | `aa6415a8d3a073a3550f209a53e1baa44461ce97` | Global checkpoint index backfill and policy enforcement. |
| **Current Review Branch** | `fix/rq1-trn-02b-colab-auth-r1` | Operational corrections, verified CLI scripts, and test suite. |
| **Future Execution Commit** | *To be pinned on merge* | Immutable Git commit SHA to be checked out on Colab VM. |

Future training MUST checkout the exact immutable Git commit SHA on the Colab VM, never a floating branch name.

### 2.2 Frozen Data and Model Artifacts Ledger

| Artifact Role | File Path / Repository Identifier | Canonical Google Drive ID / Revision | Expected SHA-256 Digest | Status |
|---|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | Repository source | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **VERIFIED MATCH** |
| **Production Baseline R Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | Drive: `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **VERIFIED MATCH** |
| **Authorized Audio Zip** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | Drive: `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **VERIFIED MATCH** |
| **Pretrained Backbone** | `facebook/wav2vec2-base-960h` | Hugging Face Git Revision | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | **PINNED REVISION** |
| **Pinned CMUdict** | `resources/CMUDict/cmudict.dict` | Repository source | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | **VERIFIED MATCH** |
| **L2-ARCTIC Prompts** | `resources/L2-ARCTIC/prompts.json` | Repository source | `87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e` | **VERIFIED MATCH** |

### 2.3 Partition and Speaker Boundaries (Fold 1)
Verified against authoritative split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`:

- **Fold Index:** 1
- **TRAIN Partition:** Exactly 2,399 utterances across **16 speakers**:
  ```text
  BWC, ERMS, HKK, HQTV, MBMPS, NCC, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA
  ```
  *(Erratum note: The historical packet erroneously listed test speakers ABA, HJK, LXC, PNV, validation speaker RRBI, and typo MBMD in TRAIN. This documentation defect has been corrected. The underlying frozen split JSON was unchanged.)*

- **VALIDATION Partition:** Exactly 300 utterances across **2 speakers**:
  - `NJS`: Utterances 0 to 149 (150 utterances).
  - `RRBI`: Utterances 150 to 299 (150 utterances).

- **SEALED TEST Partition:** Exactly 900 utterances across **6 speakers**:
  ```text
  ABA, ASI, EBVS, HJK, LXC, PNV
  ```

- **Mathematical Proof of Speaker Disjointness:**
  $$S_{\text{train}} \cap S_{\text{val}} = \emptyset, \quad S_{\text{train}} \cap S_{\text{test}} = \emptyset, \quad S_{\text{val}} \cap S_{\text{test}} = \emptyset$$
  Total corpus census: $16 + 2 + 6 = 24$ speakers; $2,399 + 300 + 900 = 3,599$ utterances ($\Delta = 0$).

- **Official Test Sealing Safeguard:**
  The authorized archive `fold1_train_val_only.zip` contains exactly 18 speaker directories (`BWC.zip`, `ERMS.zip`, `HKK.zip`, `HQTV.zip`, `MBMPS.zip`, `NCC.zip`, `NJS.zip`, `RRBI.zip`, `SKA.zip`, `SVBI.zip`, `THV.zip`, `TLV.zip`, `TNI.zip`, `TXHC.zip`, `YBAA.zip`, `YDCK.zip`, `YKWK.zip`, `ZHAA.zip`). Test speakers `ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV` are completely absent. Neither training nor preflight drivers contain any logic capable of accessing test data.

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
To prevent accidental reuse, contamination, or clobbering:
- **Local VM Output Directories:**
  - Control Arm C0: `checkpoints/RQ1-TRN-02B/C0`
  - Treatment Arm C1: `checkpoints/RQ1-TRN-02B/C1`
- **Persistent Google Drive Remote Directories:**
  - Control Arm C0: `<DRIVE_ROOT>/manifests/RQ1-TRN-02B/checkpoints/C0`
  - Treatment Arm C1: `<DRIVE_ROOT>/manifests/RQ1-TRN-02B/checkpoints/C1`
- **Fail-Closed Conflict Detection:** If any pre-existing checkpoint artifacts (`last.pt`, `best.pt`, `fixed_budget_checkpoint.pt`, `halted_guardrail.pt`) exist in the output destination and `--resume` is not explicitly passed, training halts immediately with `FileExistsError`.
- **Execution Run ID Option:** Supports `--run-id <identifier>` to namespace artifacts into distinct execution subdirectories.

---

## 4. Operational Infrastructure & Google Drive Architecture

### 4.1 Colab CLI Inspection & Verified Command Syntax
- **CLI Binary:** `/usr/bin/colab` (version `0.7.4`).
- **Authentication Provider:** OAuth2 (active user `tukhoa040505@gmail.com`).
- **CLI Syntax Correction:** The Colab CLI does NOT support positional shell command strings such as `colab exec -s S -- "bash cmd"`. Instead, `colab exec` executes a Python script via `-f <script.py>` inside the remote kernel, passing parameters via `--env KEY=VALUE`.
- **Execution Timeouts:** Default timeout for `colab exec` is 30.0 seconds. Long-running tasks must pass explicit `--timeout`:
  - Staging & preflight: `--timeout 1800` (30 minutes).
  - Model training: `--timeout 28800` (8 hours).

### 4.2 Resource Readiness & Quota Blocker
- **Measured Balance:** `0.00 compute units` (verified via `colab usage`).
- **Active Sessions:** `0` (verified via `colab sessions`).
- **Resource Requirement:** ~15–20 compute units for 7–10 T4 GPU-hours (C0: ~3.5–4.5h, C1: ~3.5–4.5h).
- **BLOCKER:** Google Colab cannot provision GPU runtimes when compute unit balance is 0.00. The operator must purchase or allocate compute units before provisioning.

### 4.3 Google Drive Hierarchy & Staging Architecture
All persistent assets reside under the authoritative project Google Drive root:
- **Drive Project Root:** `KLTN-Dataset` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)
- **Baseline R Folder:** `11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`
- **Source Baseline Checkpoint (`best.pt`):**
  - Drive Path: `manifests/RQ1-TRN-02A/checkpoints/best.pt`
  - File ID: `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`
  - Expected SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`
- **Source Authorized Data Archive (`fold1_train_val_only.zip`):**
  - Drive Path: `manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip`
  - File ID: `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY`
  - Expected SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`

### 4.4 Staging Workflow on Colab VM Disk
Streaming audio across a FUSE Drive mount introduces I/O latency and network timeouts. Therefore:
1. **Interactive Drive Mount:** The human operator executes `colab drivemount -s <session>` (or authorizes mount via Colab UI URL).
2. **Local Staging:** `audit_scripts/colab_stage_and_verify.py` copies the dataset archive to `/content/data/fold1_train_val_only.zip` and baseline weights to `/content/checkpoints/RQ1-TRN-02A/recovered/best.pt`.
3. **Cryptographic Validation:** Staged files are hashed on local VM disk and compared against expected SHA-256 digests. If any mismatch occurs, the corrupted file is removed and training is aborted.
4. **Local Execution:** Model reads audio directly from local VM disk (`/content/data/`).

### 4.5 Atomic Checkpoint Persistence & Logging
- Epoch boundaries: Checkpoints saved to temporary file (`.tmp`), flushed, and atomically replaced (`os.replace`).
- Immediate Drive Sync: Copied to persistent Google Drive destination. SHA-256 computed on Drive file and compared with local copy.
- Log Preservation: Training stdout/stderr captured in local log and continuously synced to `<DRIVE_ROOT>/manifests/RQ1-TRN-02B/checkpoints/<ARM>/training_<arm>.log`.

---

## 5. Verified Colab CLI Execution Plan

*Notice: These commands are verified through static analysis and mocked unit tests. They have NOT been executed during this pre-training review.*

### Phase 0: Operator Resource & Drive Mount Prerequisites (Interactive)

```bash
# 0.1 Verify compute unit balance (must be >= 15.0 units)
colab usage

# 0.2 Provision fresh T4 GPU runtime
colab new -s langfens-rq1-c0 --gpu T4

# 0.3 Mount Google Drive (INTERACTIVE OPERATOR PREREQUISITE)
# Note: colab drivemount requires terminal interaction or browser authorization.
colab drivemount -s langfens-rq1-c0

# 0.4 Install Python dependencies on remote Colab VM
colab install -s langfens-rq1-c0 -r requirements.txt
```

### Phase 1: Control Arm C0 Staging & Training (Colab GPU)

```bash
# Target Immutable Commit (Replace with final approved commit SHA)
export TARGET_COMMIT="<IMMUTABLE_GIT_COMMIT_SHA>"

# Step 1.1: Clone repo, checkout immutable commit, stage data & verify hashes
colab exec -s langfens-rq1-c0 \
  -f audit_scripts/colab_stage_and_verify.py \
  --env ARM=C0 \
  --env TARGET_COMMIT=$TARGET_COMMIT \
  --timeout 1800

# Step 1.2: Execute Control Arm C0 Training (Timeout: 8 hours)
colab exec -s langfens-rq1-c0 \
  -f audit_scripts/colab_train_exec.py \
  --env ARM=C0 \
  --env TARGET_COMMIT=$TARGET_COMMIT \
  --timeout 28800

# Step 1.3: Promptly release Colab runtime to preserve compute units
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

### Phase 3: Treatment Arm C1 Staging & Training (Colab GPU)

```bash
# Step 3.1: Provision fresh T4 GPU runtime
colab new -s langfens-rq1-c1 --gpu T4

# Step 3.2: Mount Google Drive (Operator interactive step)
colab drivemount -s langfens-rq1-c1

# Step 3.3: Install dependencies
colab install -s langfens-rq1-c1 -r requirements.txt

# Step 3.4: Stage data and verify hashes
colab exec -s langfens-rq1-c1 \
  -f audit_scripts/colab_stage_and_verify.py \
  --env ARM=C1 \
  --env TARGET_COMMIT=$TARGET_COMMIT \
  --timeout 1800

# Step 3.5: Execute Treatment Arm C1 Training (Timeout: 8 hours)
colab exec -s langfens-rq1-c1 \
  -f audit_scripts/colab_train_exec.py \
  --env ARM=C1 \
  --env TARGET_COMMIT=$TARGET_COMMIT \
  --timeout 28800

# Step 3.6: Promptly release Colab runtime
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
| `tests/test_rq1_trn_02b_colab_launch.py` | 18 | 18 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_preflight_r2_safety.py` | 11 | 11 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_corrections_r1.py` | 10 | 10 | 0 | **PASS** |
| `tests/test_rq1_trn_02b_implementation_preflight.py` | 14 | 13 | 1 (AGENTS.md GPU boundary) | **PASS** |
| **Combined RQ1-TRN-02B Focused Census** | **53** | **52** | **1** | **PASS** |
| **Full Repository Regression Census** | **179** | **175** | **4** (CUDA-only placeholders) | **PASS** |

---

## 7. Operational Authorization Gate Decision

```text
================================================================================
GATE: RQ1-TRN-02B-COLAB-AUTHORIZATION
STATUS: CORRECTED_PENDING_RUNTIME_VERIFICATION

All operational scripts, CLI syntax, staging procedures, and scientific split
provenance are corrected and verified locally (175 passed tests).

OPERATIONAL BLOCKERS:
1. Compute Quota: Colab compute unit balance is 0.00 units. The human operator
   must allocate compute units (~15-20 units) prior to provisioning.
2. Interactive Drive Mount: colab drivemount requires operator terminal interaction.
3. Live Staging Verification: Live VM disk copy and remote Drive write validation
   must be confirmed once the authorized Colab session is provisioned.
================================================================================
```
