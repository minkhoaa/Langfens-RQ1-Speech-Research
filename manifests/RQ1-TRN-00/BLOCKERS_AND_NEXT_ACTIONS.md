# LANGFENS — RQ1-TRN-00: Acceptance Gate Assessment, Blockers & Handover

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Acceptance Gates, Active Blockers Audit & Handover to RQ1-TRN-01  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Overall Phase Determination:** **`RQ1-TRN-00: PASS`**  

---

## 1. Acceptance Gates Evaluation

Every acceptance gate defined in the preflight specification was systematically evaluated against empirical code executions and verified test outcomes:

| Gate ID | Standard / Requirement | Result | Verified Scientific Evidence |
| :--- | :--- | :---: | :--- |
| `SOURCE_REPRODUCIBILITY_AUDITED` | Source code verification, hashes, clean git tree, and test execution | **PASS** | 28 automated pytest tests freshly executed with 0 failures in 31.67s; SHA-256 integrity manifest compiled; clean working tree on commit `47b2955`; zero remote pushes. |
| `ENVIRONMENT_VERIFIED` | Dependency inventory, Python runtime, GPU status & manifest digests | **PASS** | Python 3.14.8, PyTorch 2.14.1+cpu, 25 pinned packages in `requirements-lock.txt`; GPU GTX 1650 (4GB) analyzed; 100% SHA-256 match on raw archives and frozen split manifests. |
| `TRAINING_CONTRACTS_VALIDATED` | 10 training contract rules, loss masking proofs, label semantics | **PASS** | `MaskedBCEWithLogitsLoss` prevents -100 explosion; `MaskedCrossEntropyLoss` masks deletions & `err` tags; CTC blank 0 verified non-colliding; test blinded mode confirmed. |
| `BASELINE_EXPERIMENT_DESIGNED` | Comparative design of Baseline A (GOP) vs Baseline B (Wav2Vec2) | **PASS** | Detailed architectural specifications, comparative trade-off matrix, and literature-justified research order (Baseline B first, then Baseline A) finalized. |
| `COMPUTE_RESOURCES_PLANNED` | Empirical DataLoader benchmarks, staging economics & VRAM budget | **PASS** | 251.8 utts/s CPU baseline documented; 25.8× raw I/O speedup demystified to 3.25% end-to-end gain; 876 MB staging budget fit into 101 GB NVMe; 4GB VRAM batch size 2-4 planned. |
| `PROTOCOL_FROZEN` | 4-fold CV, Vietnamese LOSO/Zero-Shot, PR-AUC, bootstrap CIs & schema | **PASS** | Complete protocol freeze with strict binary primary metric ($N=119,151$), validation-only calibration, $B=1,000$ bootstrap CIs, YAML schema, and CC BY-NC 4.0 compliance. |

**Overall Gate Determination:** **`RQ1-TRN-00: PASS`**

---

## 2. Active Blockers & Environmental Considerations Audit

| Item / Finding | Current Status | Scientific Analysis & Resolution |
| :--- | :---: | :--- |
| **Python 3.14 + PyTorch CPU-Only Runtime** | Managed / Planned | **Current Preflight:** Zero impact. Preflight execution, data contracts, and micro-benchmarks run 100% on CPU.<br>**Downstream Training (RQ1-TRN-01):** Bleeding-edge Python 3.14 currently lacks pre-built PyTorch CUDA wheels. For GPU fine-tuning of Wav2Vec 2.0, provision a dedicated Python 3.11/3.12 environment with `torch+cu12x` or deploy to a containerized GPU instance. |
| **GTX 1650 VRAM Limitation (4,096 MiB)** | Resolved / Budgeted | Transformer self-attention risks CUDA OOM on batch size $\ge 8$. Resolved by enforcing micro-batch size $B=2$, gradient accumulation $N=8$, activation checkpointing, and frozen CNN feature encoder. |
| **BCE Loss -100 Padding Trap** | Resolved & Tested | Unmasked `nn.BCEWithLogitsLoss` treats -100 as real ground truth, causing severe loss corruption. Resolved via `MaskedBCEWithLogitsLoss` and explicit `target_binary_mask`. |
| **Vietnamese Phonemic Deletions (1,566 tokens)** | Resolved in Design | Heavy deletions ($45.8\%$ of corpus deletions) cause classical forced-alignment failure. Mitigated by prioritizing Baseline B (Wav2Vec2 end-to-end CTC). |

---

## 3. Handover & Actionable Directives for RQ1-TRN-01

With Phase `RQ1-TRN-00: PASS`, Project Langfens is officially certified as **modeling-ready**.

### Immediate Action Plan for Phase RQ1-TRN-01 (Baseline Acoustic Modeling):
1. **GPU Runtime Provisioning:**
   - Provision a standard Python 3.11 or 3.12 environment with CUDA-enabled PyTorch (`torch>=2.2.0+cu121`, `torchaudio>=2.2.0+cu121`) on the local GTX 1650 or cloud compute node.
2. **Ephemeral Audio Staging:**
   - Execute `AudioStagingManager` to extract all 3,599 16 kHz resampled WAVs onto local NVMe storage ($876 \text{ MB}$), eliminating ZIP decompression lock contention.
3. **Model Initialization:**
   - Load `facebook/wav2vec2-base-960h` backbone; initialize CTC linear projection (40 classes), binary detection head (1 logit), and 39-class diagnosis head.
   - Freeze the 7-layer convolutional feature encoder; enable gradient checkpointing.
4. **Execute Fold 1 Training:**
   - Train using `exp_template.yaml` on Fold 1 of `l2arctic_4fold_splits.json`.
   - Calibrate operating threshold $\theta^*$ strictly on the validation split.
   - Evaluate blinded test split and compute $95\%$ bootstrap confidence intervals across the $N=119,151$ canonical evaluation tokens.

### Immutable Boundary Reminder:
- **PREFLIGHT AND READINESS ONLY.** No acoustic model training or inference was executed in this phase. Real model training is formally authorized to commence upon human sign-off of Phase `RQ1-TRN-01`.
