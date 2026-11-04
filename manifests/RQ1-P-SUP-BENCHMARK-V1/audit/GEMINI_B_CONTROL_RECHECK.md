# Independent Recheck & Forensic Verification Report: Matched Control B-CONTROL (`binary_collapsed`)

**Document Authority:** Authoritative Subagent Artifact for Task Session `3dce8806-0948-45cf-a636-47a0095821ed`  
**Authoritative Output Path:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-2.md`  
**Investigator:** Agent B (Independent Parallel Forensic Auditor & B-Control Recovery Recheck Specialist)  
**Verification Date:** 2026-11-04T02:45:00Z  
**Coordinator Repo:** `/home/khoa/Projects/KLTN/Dataset`  
**Coordinator Branch / HEAD:** `review/psup-recovery-benchmark-v1` at commit `ea16b83ca906e6bbd867296ed702bdf57e677feb` (dirty working tree preserved unmodified)  
**Assigned Worktree:** `/home/khoa/Projects/KLTN/psup-recover-b-control-v1` on branch `review/psup-recover-b-control-v1` (base commit `40bc3770776ee1c99462d44101fdd145602b76f5`, untracked verification receipts preserved unmodified)  
**Execution Environment:** Read-only forensic verification executed via `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python` (Python 3.11.17, PyTorch 2.14.1+cu130, safe CPU deserialization)  
**Status:** **RECHECK_COMPLETED; CRYPTOGRAPHIC_AND_PAYLOAD_VERIFIED; SUBTYPE_UNIDENTIFIABILITY_DISCLOSED; SNAPSHOT_TIMING_RECONCILED; NOT_A_BLANKET_SCIENTIFIC_PASS**

---

## 1. Executive Recheck Summary & Mandate

Under explicit user authorization for independent artifact recovery and benchmark recheck (with strict prohibition against model training, architecture modifications, backwards passes, optimizer steps, full pytests, new experiments, literature searches, or sealed test partition access), Agent B conducted an exhaustive, read-only forensic re-audit of the recovered **P-SUP Matched Binary Control B-CONTROL** (`binary_collapsed`) artifacts.

### Key Audit Findings
1. **Zero Redundant Remote Downloads:**
   Pre-existing verified artifacts staged in local approved scratch storage (`/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/`) were inspected directly. No duplicate remote bulk downloads or unfiltered Kaggle outputs were issued.
2. **Cryptographic Parity Across All Five Required Artifacts:**
   All five primary required artifacts specified in `manifests/RQ1-P-SUP-BENCHMARK-V1/RECOVERY_MANIFEST.json` (`fixed_budget_3600.pt`, `best.pt`, `last.pt`, `training_summary.json`, `history.json`), as well as alias `fixed_budget_checkpoint.pt` and run log `langfens-rq1-psup-b-control-t4x2.log`, match bit-identically across:
   - Approved local recovery scratch directory (`/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/`),
   - Authenticated private Hugging Face model repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` under prefix `psup/b_control/` at pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`,
   - Kaggle official kernel execution output from `minkhoaa/langfens-rq1-psup-b-control-t4x2`,
   - Historical local candidates at `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control/` (which previously lacked `last.pt`).
3. **Safe CPU Payload Inspection (`weights_only=True`):**
   Using `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python` with explicit NumPy safe globals (`np._core.multiarray._reconstruct`, `np.ndarray`, `np.dtype`, `np.dtypes.UInt32DType`), all PyTorch `.pt` payloads were safely deserialized without executing unpickler arbitrary code. All 16 trainable head parameter tensors (totaling 266,667 parameters) and all 16 optimizer moment states are 100% strictly finite (0 NaN, 0 Inf).
4. **Dual Endpoint Semantics & Exact Update Counters:**
   - `fixed_budget_3600.pt`: exactly **3,600 successful updates**, **0 skipped AMP updates**, **7,200 micro-steps**, `epoch: 7`, optimizer step counter `3600` on all 16 parameter states, scheduler `last_epoch: 3600`, learning rate `0.0008699219550575954`, `is_fixed_budget_endpoint: True`.
   - `best.pt`: exactly **2,250 successful updates**, **0 skipped AMP updates**, **4,500 micro-steps**, `epoch: 4`, optimizer step counter `2250` on all 16 parameter states, scheduler `last_epoch: 2250`, learning rate `0.0009494996720699363`, early stopping patience `0`, `is_fixed_budget_endpoint: False`. Selected at peak validation AP (`0.46760702566135065`) on the internal selection partition (`MBMPS`, `TLV`).
   - `last.pt`: exactly **3,600 successful updates**, `epoch: 7`, `is_fixed_budget_endpoint: False`. Bit-identical head weights and optimizer moments to `fixed_budget_3600.pt`.
5. **Exact Root-Cause Accounting of the Embedded History Mismatch (3150 vs 3600):**
   The embedded `history` inside `fixed_budget_3600.pt` contains 7 epoch records ending at update `3150`, while payload counters, optimizer states, and scheduler states are at `3600`. Source code inspection of `src/langfens_prep/p_sup/trainer.py` (lines 1162–1186 at commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`) proves this is an exact artifact snapshot timing artifact: `fixed_budget_3600.pt` was snapshotted inside the training loop immediately when `ledger.successful` reached 3,600, *before* epoch 7's post-epoch validation evaluation executed to append the 8th history entry. Once epoch 7 validation completed, the final record was appended, which is why `last.pt`, `history.json`, and `training_summary.json` correctly contain all 8 epoch entries ending at 3,600 updates.
6. **Subtype Unidentifiability Disclosure & Gradient Precision Correction:**
   Under `objective: binary_collapsed`, output classes 1 (substitution) and 2 (omission) are collapsed via $\text{logsumexp}(z_1, z_2) - z_0$ against a binary error label $\mathbb{I}(y \ne 0)$. The gradient updates to $z_1$ and $z_2$ differ only by within-error softmax weights ($\frac{e^{z_1}}{e^{z_1}+e^{z_2}}$ vs $\frac{e^{z_2}}{e^{z_1}+e^{z_2}}$); the supervisory signal provides zero discriminating feedback distinguishing omissions from substitutions. Consequently, subtype logits are **UNIDENTIFIED**, not omission/substitution probabilities. In runtime output rows, `s_omit`, `s_sub`, `probabilities`, and `omission_ap_dedicated` are strictly `null`.
7. **Strict Boundary Adherence:**
   Zero backward passes, zero optimizer steps, and zero test partition audio/labels were accessed in this process. No blanket scientific pass is declared: candidate validation metrics must be evaluated on Fold-1 VAL under genuine protocols (O and Z), separate from the internal selection partition metrics recorded during training.

---

## 2. Artifact Registry & Checksum Verification

### 2.1 All Five Required Files (from `RECOVERY_MANIFEST.json`)
Approved Local Scratch Directory: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/`

| Artifact Filename | File Size (Bytes) | SHA-256 Checksum | Hugging Face LFS / Readback SHA-256 | Kaggle Output Match | Pre-Existing Candidate Status | Overall Status |
|---|---|---|---|---|---|---|
| `fixed_budget_3600.pt` | 3,241,295 | `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` | `e121a7b965747b49...` (LFS match) | MATCH | Bit-identical match | **VERIFIED** |
| `best.pt` | 3,239,450 | `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f` | `2dbad5a0160224f9...` (LFS match) | MATCH | Bit-identical match | **VERIFIED** |
| `last.pt` | 3,239,898 | `aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5` | `aa35c14dbf682705...` (LFS match) | MATCH | Absent locally; recovered | **RECOVERED** |
| `training_summary.json` | 3,339 | `fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f` | `fb8917eda607e2cf...` (Direct match) | MATCH | Bit-identical match | **VERIFIED** |
| `history.json` | 2,896 | `dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6` | `dc61bbff9b58f804...` (Direct match) | MATCH | Bit-identical match | **VERIFIED** |
| `fixed_budget_checkpoint.pt` | 3,241,295 | `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` | `e121a7b965747b49...` (LFS match) | MATCH | N/A (Alias of 3600.pt) | **VERIFIED** |
| `langfens-rq1-psup-b-control-t4x2.log` | 380,708 | `24f1a2a30211b29d1590fba830a8c314c3a46657db57bef50cf333afc25d8936` | N/A | MATCH (API download) | N/A | **RECOVERED** |

### 2.2 Independent Verification Script Output
Executed via `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python`:
```text
=== TEST 1: RECOVERY MANIFEST CHECKSUM & SIZE MATCH ===
PASS: fixed_budget_3600.pt (size=3241295, sha=e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201)
PASS: best.pt (size=3239450, sha=2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f)
PASS: last.pt (size=3239898, sha=aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5)
PASS: training_summary.json (size=3339, sha=fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f)
PASS: history.json (size=2896, sha=dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6)

=== TEST 2: LOCAL PRE-EXISTING CANDIDATE PARITY ===
PASS: Candidate best.pt matches recovered artifact SHA256
PASS: Candidate fixed_budget_3600.pt matches recovered artifact SHA256
PASS: Candidate history.json matches recovered artifact SHA256
PASS: Candidate training_summary.json matches recovered artifact SHA256
PASS: last.pt confirmed absent from candidate dir (recovered from remote)
```

---

## 3. Safe CPU Payload Inspection & Tensor Finiteness

### 3.1 Deserialization Protocol
In strict compliance with PyTorch security standards, checkpoints were loaded on CPU without executing arbitrary pickled code:
```python
import numpy as np
import torch
safe_globals = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
with torch.serialization.safe_globals(safe_globals):
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

### 3.2 Parameter Budget & Weight Architecture (266,667 Parameters)
Across all three checkpoints (`fixed_budget_3600.pt`, `best.pt`, `last.pt`), the model architecture is confirmed identical:
- **Total Head Parameters:** Exactly **266,667** float32 parameters across 16 tensors:
  - `acoustic_projection.weight`: shape `[128, 768]` (98,304 params)
  - `acoustic_projection.bias`: shape `[128]` (128 params)
  - `phone_embedding.weight`: shape `[40, 128]` (5,120 params)
  - `position_embedding.weight`: shape `[256, 128]` (32,768 params)
  - `attention.in_proj_weight`: shape `[384, 128]` (49,152 params)
  - `attention.in_proj_bias`: shape `[384]` (384 params)
  - `attention.out_proj.weight`: shape `[128, 128]` (16,384 params)
  - `attention.out_proj.bias`: shape `[128]` (128 params)
  - `acoustic_norm.weight`: shape `[128]` (128 params)
  - `acoustic_norm.bias`: shape `[128]` (128 params)
  - `operation_head.0.weight`: shape `[128, 256]` (32,768 params)
  - `operation_head.0.bias`: shape `[128]` (128 params)
  - `operation_head.3.weight`: shape `[3, 128]` (384 params)
  - `operation_head.3.bias`: shape `[3]` (3 params)
  - `ctc_head.weight`: shape `[40, 768]` (30,720 params)
  - `ctc_head.bias`: shape `[40]` (40 params)
- **Finiteness Check:** 100% of head parameter tensors and 100% of optimizer moment tensors (`exp_avg`, `exp_avg_sq`) are strictly finite (`torch.isfinite(t).all() == True`). Zero NaN or Inf values detected.
- **Head Weights Equivalence:** Direct tensor equality comparison (`torch.equal`) confirmed that `fixed_budget_3600.pt` and `last.pt` have 100% bit-for-bit identical head weights and optimizer moments.

---

## 4. Counter & Endpoint Semantics Verification

### 4.1 Comparative Endpoint Ledger

| Field / Attribute | `fixed_budget_3600.pt` | `best.pt` | `last.pt` | Verification Status |
|---|---|---|---|---|
| `successful_updates` | `3600` | `2250` | `3600` | **VERIFIED** |
| `micro_steps` | `7200` | `4500` | `7200` | **VERIFIED** (2 micro-steps/update) |
| `skipped_amp_attempts_total` | `0` | `0` | `0` | **VERIFIED** (Zero AMP skips) |
| `epoch` | `7` | `4` | `7` | **VERIFIED** |
| `is_fixed_budget_endpoint` | `True` | `False` | `False` | **VERIFIED** |
| Optimizer States Count | 16 | 16 | 16 | **VERIFIED** |
| Optimizer Step Counters | `[3600] * 16` | `[2250] * 16` | `[3600] * 16` | **VERIFIED** |
| Scheduler `last_epoch` | `3600` | `2250` | `3600` | **VERIFIED** |
| Scheduler Step Count (`_step_count`) | `3601` | `2251` | `3601` | **VERIFIED** |
| Optimizer Current Learning Rate | `0.0008699219550575954` | `0.0009494996720699363` | `0.0008699219550575954` | **VERIFIED** (Exact cosine decay) |
| GradScaler `scale` | `8192.0` | `8192.0` | `8192.0` | **VERIFIED** |
| GradScaler `_growth_tracker` | `1600` | `250` | `1600` | **VERIFIED** |
| RNG State Keys | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | `python`, `numpy`, `torch`, `cuda` (2 GPUs) | **VERIFIED** |
| Embedded History Length | 7 epochs (ends update 3150) | 5 epochs (ends update 2250) | 8 epochs (ends update 3600) | **RECONCILED** (Snapshot timing) |
| Selection AP (`best_ap`) | `0.46760702566135065` | `0.46760702566135065` | `0.46760702566135065` | Peak achieved at Epoch 4 |
| Early Stopping Patience | `2` | `0` | `2` | **VERIFIED** |

### 4.2 Endpoint Accounting Justification
- `best.pt` legitimately holds **2,250 successful updates** because validation selection AP reached its maximum (`0.46760702566135065`) at Epoch 4 on the `selection` partition (`MBMPS`, `TLV`). In accordance with early-stopping patience rules, the selection checkpoint preserves the weights at that peak.
- `fixed_budget_3600.pt` legitimately holds **3,600 successful updates**, representing the fixed-budget 3,600-step training horizon required by `AGENTS.md` dual-checkpoint accounting.

---

## 5. Root-Cause Analysis of Embedded Fixed History Mismatch (3150 vs 3600)

### 5.1 Observation
- In `fixed_budget_3600.pt`:
  - `successful_updates`: `3600`
  - Optimizer step counters: `3600` across all 16 parameter states
  - Scheduler `last_epoch`: `3600`
  - `history` array length: `7` (epochs 0 through 6), with `history[-1]['successful_updates'] == 3150`.
- In `last.pt`, `history.json`, and `training_summary.json`:
  - `history` array length: `8` (epochs 0 through 7), with `history[-1]['successful_updates'] == 3600`.

### 5.2 Architectural Cause in `src/langfens_prep/p_sup/trainer.py`
Inspection of the executed training script at commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`:
```python
# Lines 1162-1186: Inside the batch accumulation loop of Epoch 7:
if ledger.fixed_budget_reached and not fixed_budget_saved:
    fb_path = output_dir / "fixed_budget_3600.pt"
    save_checkpoint(
        path=fb_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        successful_updates=ledger.successful,  # 3600
        micro_steps=micro_steps,              # 7200
        best_ap=best_ap,
        patience=patience,
        skipped_amp=ledger.skipped,           # 0
        history=history,                      # <--- history at this moment only has epochs 0..6
        metadata=metadata,
        is_fixed_budget_endpoint=True,
        ...
    )
    fixed_budget_saved = True

# Lines 1195-1215: Post-epoch validation evaluation executed AFTER batch loop completes:
val_metrics = evaluate_val_set(adapter, selection_loader, device, objective)
epoch_record = {
    "epoch": epoch,
    "successful_updates": ledger.successful,
    "val_ap": val_ap,
    ...
}
history.append(epoch_record)  # <--- 8th record (epoch 7, 3600 updates) appended HERE!

# Lines 1245-1260: Final checkpoint saving:
save_checkpoint(last_path, ..., history=history, is_fixed_budget_endpoint=False)
```

### 5.3 Audit Conclusion on Mismatch
The mismatch is **not** an indication that training terminated early or that only 3,150 updates were performed. The model weights, optimizer moments, scheduler counters, and update ledgers conclusively prove that 3,600 updates were completed. The embedded history stopped at 3,150 solely because the fixed-budget checkpoint was serialized inside the inner training step loop immediately upon reaching update 3,600, prior to the end-of-epoch validation call that appended the 8th history entry.

---

## 6. Remote Provenance & Metadata Verification

### 6.1 Hugging Face Hub Checkpoint Persistence
- **Repository Slug:** `minkhoaa/langfens-rq1-cross-fold-checkpoints`
- **Repository Type:** `model`
- **Privacy Setting:** Confirmed **Private** (`private: True`, verified via authenticated `HfApi().repo_info`).
- **Pinned Git Revision for `psup/b_control/`:** `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
- **Subtree Evolution Note:** The repository HEAD advanced to `6d1ec3...` and `7a9eae...` as the parent orchestrator mirrored archival benchmark predictions under `benchmarks/psup-v1/`. The `psup/b_control/` subtree was verified bit-identical and unchanged from pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`.
- **Remote Files & LFS Pointers in `psup/b_control/`:**
  - `best.pt`: 3,239,450 bytes, LFS SHA-256 `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f`
  - `fixed_budget_3600.pt`: 3,241,295 bytes, LFS SHA-256 `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201`
  - `fixed_budget_checkpoint.pt`: 3,241,295 bytes, LFS SHA-256 `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201`
  - `last.pt`: 3,239,898 bytes, LFS SHA-256 `aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5`
  - `history.json`: 2,896 bytes, SHA-256 `dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6`
  - `training_summary.json`: 3,339 bytes, SHA-256 `fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f`
- **Integrity Verification:** 100% cryptographic parity across all files.

### 6.2 Kaggle Remote Execution
- **Kernel Slug:** `minkhoaa/langfens-rq1-psup-b-control-t4x2`
- **Status:** `KernelWorkerStatus.COMPLETE` (verified via `kaggle kernels status minkhoaa/langfens-rq1-psup-b-control-t4x2`)
- **Runtime Hardware:** `NvidiaTeslaT4` (Dual Tesla T4 GPUs, NVIDIA Driver 580.178.04, CUDA 13.0)
- **Staging Dataset:** `minkhoaa/langfens-rq1-fold1-staging` (verified private, SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`)
- **Accounted Download Boundary Filter:**
  An early bulk download command `kaggle kernels output` began streaming the full cloned git tree (`repo/`). The coordinator cleanly terminated PID 183900 via SIGINT to prevent downloading unnecessary working trees. Bounded retrieval (`--file-pattern ".*\.(pt|json)$"`) was subsequently used. This accounted operational download boundary filter is NOT an artifact integrity failure. All target payloads are complete, authentic, and verified.

---

## 7. Mathematical & Semantic Disclosure: Subtype Unidentifiability

### 7.1 Objective Algebra Under `binary_collapsed`
In `src/langfens_prep/p_sup/objectives.py`:
$$\text{collapsed\_binary\_logit}(z) = \log\left(\sum_{k=1}^2 \exp(z_k)\right) - z_0 = \text{logsumexp}(z_1, z_2) - z_0$$
$$\mathcal{L}_{\text{task}} = \text{BCEWithLogits}(\text{collapsed\_binary\_logit}(z), \mathbb{I}(y \ne 0))$$

### 7.2 Gradient Derivation and Clarification of Gradient Wording
Let $\tilde{z} = \text{collapsed\_binary\_logit}(z)$ and $t = \mathbb{I}(y \ne 0) \in \{0, 1\}$.
The scalar loss derivative with respect to $\tilde{z}$ is:
$$\frac{\partial \mathcal{L}_{\text{task}}}{\partial \tilde{z}} = \sigma(\tilde{z}) - t$$
By the chain rule:
$$\frac{\partial \tilde{z}}{\partial z_0} = -1 \implies \frac{\partial \mathcal{L}_{\text{task}}}{\partial z_0} = t - \sigma(\tilde{z})$$
For $k \in \{1, 2\}$ (where $1 = \text{substitution}$, $2 = \text{omission}$):
$$\frac{\partial \tilde{z}}{\partial z_k} = \frac{\exp(z_k)}{\exp(z_1) + \exp(z_2)} = \text{softmax}([z_1, z_2])_k$$
$$\frac{\partial \mathcal{L}_{\text{task}}}{\partial z_k} = (\sigma(\tilde{z}) - t) \cdot \frac{\exp(z_k)}{\exp(z_1) + \exp(z_2)}$$

**Precision Correction on Prior Reporting:**
Prior recovery text describing gradients as "strictly identical" was slightly imprecise: the gradients $\frac{\partial \mathcal{L}_{\text{task}}}{\partial z_1}$ and $\frac{\partial \mathcal{L}_{\text{task}}}{\partial z_2}$ differ by the internal softmax weights $\frac{\exp(z_1)}{\exp(z_1)+\exp(z_2)}$ versus $\frac{\exp(z_2)}{\exp(z_1)+\exp(z_2)}$.
However, the supervisory error signal $t = \mathbb{I}(y \ne 0)$ is **identical** for substitutions ($y=1$) and omissions ($y=2$). The supervisory feedback provides **zero discriminative gradient direction** to separate omissions from substitutions. 

### 7.3 Consequences for Runtime Scoring & Metric Semantics
In `src/langfens_prep/p_sup/runtime.py`:
- `scores["s_error"] = p[1] + p[2]` (sigmoid of collapsed binary logit)
- `scores["s_omit"] = None` (unsupported / unidentified)
- `scores["s_sub"] = None` (unsupported / unidentified)
- `probabilities = None` (unsupported / unidentified)
- `score_semantics = "binary collapse of same three logits; subtype unidentified/null; not calibrated"`
- `omission_ap_dedicated = null`

Any omission AP or substitution AP reported in historical B-CONTROL tables is evaluated strictly in legacy mode (using the generic binary error score `s_error` restricted to the respective label subsets). Dedicated omission detection is mathematically unidentifiable under this objective.

---

## 8. Source Code Execution Evidence, Discrepancies & Limitations

### 8.1 Verified Execution Commit
- **Execution Git SHA:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` ("fix(trainer): import datetime, timezone, and subprocess for checkpoint metadata").
- **Evidence in Notebook:** `langfens-rq1-psup-b-control-t4x2.ipynb` Phase 3 explicitly checks out `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` and asserts `actual_head == PINNED_COMMIT_SHA`.
- **Evidence in Payloads:** All three checkpoint metadata blocks record `git_sha: "dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f"`.

### 8.2 Discrepancies and Limitations
1. **Historical Commit Documentation Gap:**
   Earlier entries in `INDEX.md` and interim reports referenced commits `9ad1e5e` (a subsequent PyTorch 2.6 evaluator fix) or `df55f12` (the staging commit before adding metadata import fixes). The actual executed training code on Kaggle was `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`. This discrepancy is an accounted documentation gap and does not affect the validity of the recovered weights.
2. **Kaggle Kernel Visibility Metadata:**
   The Kaggle notebook metadata API reports `is_private=false` for the kernel container. However, the underlying dataset `minkhoaa/langfens-rq1-fold1-staging` and the Hugging Face checkpoint repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` are confirmed private.
3. **Training Log Granularity:**
   The remote execution logs record epoch validation summaries on the `selection` partition (`MBMPS`, `TLV`). They do not contain utterance-by-utterance decoded strings or token rows for the unseen validation set (`NJS`, `RRBI`).
4. **Validation Metric Distinction (Not a Scientific PASS):**
   The selection AP reported during training (`0.46760702566135065` for `best.pt` and `0.46411747113136137` for `fixed_budget_3600.pt`) was computed on the 300 `selection` partition utterances, NOT on Fold-1 VAL (`NJS`, `RRBI`). Candidate evaluation on Fold-1 VAL requires separate benchmark inference (under both protocol O and genuine Z) and must be independently recomputed by coordinator runners.

---

## 9. Data Partitioning, Isolation & Contract Integrity

### 9.1 Partition Disjointness
Metadata partition inspection of the recovered checkpoints confirms strict compliance with the evaluation contract:
- **Fit Partition (1,799 utterances, 12 speakers):**
  `['BWC', 'ERMS', 'HKK', 'HQTV', 'SKA', 'SVBI', 'TNI', 'TXHC', 'YBAA', 'YDCK', 'YKWK', 'ZHAA']`
- **Selection Partition (300 utterances, 2 speakers):**
  `['MBMPS', 'TLV']`
- **Calibration Partition (300 utterances, 2 speakers):**
  `['NCC', 'THV']`
- **Evaluation Partition (300 utterances, 2 speakers):**
  `['NJS', 'RRBI']` (Held strictly unseen during training)
- **Sealed Test Partition (900 utterances, 6 speakers):**
  `['ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV']` (Strictly absent from training staging archive)

All subsets are pairwise disjoint ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$). Zero test partition audio or labels were staged or accessed.

### 9.2 Binding Evaluation Hashes
- Contract SHA-256: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` (**MATCH**)
- Split Manifest SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` (**MATCH**)
- Dataset Archive SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` (**MATCH**)
- Backbone Architecture: `facebook/wav2vec2-base-960h` at revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` (**MATCH**)

---

## 10. Audit Commands Run & Evidence Summary

The following verification commands were executed read-only during this audit:
1. Manifest and Checksum Audit:
   ```bash
   /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c "..."
   # Verified all 5 files in RECOVERY_MANIFEST.json bit-identical to recovered scratch
   ```
2. Safe CPU Checkpoint Deserialization:
   ```bash
   /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c "
   import numpy as np, torch
   safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
   with torch.serialization.safe_globals(safe):
       data = torch.load('...', map_location='cpu', weights_only=True)
   "
   # Verified 16 tensors, 266,667 parameters, 100% finite, AdamW optimizer steps, scheduler steps, GradScaler, RNG
   ```
3. Remote Private Hugging Face API Probe:
   ```bash
   /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   info = api.repo_info('minkhoaa/langfens-rq1-cross-fold-checkpoints', repo_type='model')
   # Confirmed repo private=True, inspected LFS blobs for revision 4aa9f2a68f0709889eacacf170b6446d3c45c73d
   "
   ```
4. Kaggle Kernel Status Probe:
   ```bash
   kaggle kernels status minkhoaa/langfens-rq1-psup-b-control-t4x2
   # Confirmed status 'KernelWorkerStatus.COMPLETE'
   ```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Executed read-only forensic recheck of recovered B-CONTROL artifacts without modifying repository source, worktrees, or checkpoints; no training, backward passes, model fitting, new experiments, literature searches, or test partition access performed."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Verified all 5 required files actual SHA-256 against RECOVERY_MANIFEST.json; validated safe CPU torch.load weights_only=True with numpy safe globals; confirmed successful updates 3600 (fixed/last) and 2250 (best), 0 skips, finite payload tensors, objective binary_collapsed, dataset source SHA dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f, embedded history 3150 qualification, private HF/Kaggle parity, and subtype unidentifiability."
    }
  ],
  "changedFiles": [
    "/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/3dce8806-0948-45cf-a636-47a0095821ed/recheck/GEMINI_PARALLEL_RECHECK.task-2.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c \"import hashlib, json...\"",
      "result": "passed",
      "summary": "Cryptographically verified all 5 B-Control files in RECOVERY_MANIFEST.json match local scratch bit-for-bit."
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c \"import torch, numpy as np...\"",
      "result": "passed",
      "summary": "Safe CPU torch.load under weights_only=True allowlisting numpy globals; verified 266,667 finite parameters, counters, optimizer states, scheduler states, scaler, RNG, and embedded history."
    },
    {
      "command": "/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -c \"from huggingface_hub import HfApi...\"",
      "result": "passed",
      "summary": "Verified private HF repository minkhoaa/langfens-rq1-cross-fold-checkpoints and LFS blob hashes at revision 4aa9f2a68f0709889eacacf170b6446d3c45c73d."
    },
    {
      "command": "kaggle kernels status minkhoaa/langfens-rq1-psup-b-control-t4x2",
      "result": "passed",
      "summary": "Confirmed Kaggle kernel worker status is COMPLETE."
    }
  ],
  "validationOutput": [
    "fixed_budget_3600.pt: SHA-256 e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201, 3600 updates, 0 skips, embedded history ends at 3150 (snapshot timing qualification).",
    "best.pt: SHA-256 2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f, 2250 updates, 0 skips, peak selection AP 0.467607 at epoch 4.",
    "last.pt: SHA-256 aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5, 3600 updates, 0 skips, 8 history entries.",
    "training_summary.json: SHA-256 fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f.",
    "history.json: SHA-256 dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6.",
    "Subtype unidentifiability verified: binary_collapsed loss supervision treats substitution and deletion identically; subtype logits remain unidentified."
  ],
  "residualRisks": [
    "Selection partition metrics (MBMPS, TLV) evaluated during training must not be conflated with Fold-1 VAL (NJS, RRBI) benchmark metrics.",
    "Dedicated omission AP is unsupported under binary_collapsed; legacy omission metrics evaluate general binary error scores on the deletion subset."
  ],
  "noStagedFiles": true,
  "diffSummary": "Generated authoritative forensic verification report GEMINI_PARALLEL_RECHECK.task-2.md in managed subagent artifact path; zero repository files staged or altered.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "The embedded history in fixed_budget_3600.pt ending at 3150 is accounted for by the checkpoint serialization occurring in the training loop upon reaching 3600 updates, immediately before the post-epoch validation evaluation appended the 8th history entry. Parity is confirmed across all sources."
}
```
