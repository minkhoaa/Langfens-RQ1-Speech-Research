# RQ1-TRN-02B Acoustic Conditioning Improvement Design

**Revision:** DESIGN-R1
**Status:** Design-Only Specification.
**Execution Boundary:** STRICTLY NO implementation, training, optimizer updates, GPU allocation, or official-test partition access is authorized by this document. Any operational execution requires separate, explicit authorization.

---

## 1. Provenance and Frozen Context

This design revision is established on branch `docs/rq1-trn-02b-acoustic-conditioning-design`, based on design commit `68363673f04719eb976a5e690e0845c0164adc3c`, whose direct parent is the frozen baseline commit `c4a658ffbd5a2afd2ebdce172e47af43ed9bb63e`. All historical baseline evidence, split definitions, checkpoint weights, and evaluation artifacts are permanently frozen and documented in `manifests/RQ1-TRN-02B/BASELINE_FREEZE_LEDGER.md`. No historical RQ1-TRN-02A artifacts, scripts, or numerical records may be modified or silently overwritten.

### 1.1 Frozen Baseline Anchors

- **Fold 1 Split Contract:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
  - TRAIN: 2,399 utterances across 16 speakers.
  - VALIDATION: 300 utterances across 2 speakers (`NJS`, `RRBI`).
  - SEALED TEST: 900 utterances across 6 speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`). The official test set remains sealed, unstaged, and unextracted.
- **Production Baseline Checkpoint:** `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`).
  - Saved at epoch index 5 (display epoch 6), exactly 3,600 successful optimizer updates.
  - Baseline validation AP: `0.316062` (validation selection checkpoint replay) / `0.317021` (matched/mismatched control evaluation). These distinct historical runs are retained with their original provenance.
  - Baseline validation greedy CTC PER: `23.64%` (also reported as `23.68%` in an independent replay).
  - Original Colab training execution SHA: `caa56783113d40eac438e5f458cdb313b8932218`.
- **Pretrained Acoustic Backbone:** `facebook/wav2vec2-base-960h` (Hugging Face revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).
  - Strictly frozen (`param.requires_grad = False`, `.eval()` mode maintained during training and evaluation).
- **Disabled Diagnosis Head:** Untrained and excluded from parameter groups and forward passes.
- **Evaluation Baseline Evidence:**
  - Train-only canonical phone prior AP: `0.303063` (validation prevalence: `0.116211`).
  - Same-speaker shift-1 audio permutation AP: `0.294722` (overall decision agreement at $\tau=0.50$ is `96.76%`, while error-class agreement drops to `86.60%`, with 201 vs 172 TP and 208 vs 250 FP).
  - Protocol Z G2P zero-oracle aligned-only AP: `0.313530` across 9,919 / 9,956 valid aligned positions (`99.63%` coverage; evaluator SHA: `03854dca2dce81fbfcf1e2715eaf399af477dd62`).

---

## 2. Hypothesis and Minimal Architectural Intervention

### 2.1 Theoretical Rationale

In the dual-head RQ1 architecture (`src/langfens_prep/model.py`), the canonical phoneme embedding vector serves two distinct functional roles:
1. **Cross-Attention Queries:** Queries the frame-level acoustic representation extracted by the frozen Wav2Vec2 encoder to produce aligned acoustic context vectors $\mathbf{c}_i$.
2. **Direct Lexical Feature Shortcut:** Is directly concatenated unregularized with the aligned acoustic vector into the Binary MDD classification MLP:
   $$\mathbf{h}_i = \text{MLP}([\mathbf{c}_i \,\|\, \mathbf{e}_{\text{phone}}(y_i)])$$

Because $\mathbf{e}_{\text{phone}}(y_i)$ directly enters the classification head without regularization, the Binary MLP can readily memorize marginal phone-specific error priors rather than attending to phonetic acoustic mismatches. The core hypothesis of RQ1-TRN-02B is that selectively disrupting only this direct lexical shortcut during training will force the Binary MLP to rely more heavily on the aligned acoustic representation $\mathbf{c}_i$, thereby improving acoustic discrimination.

### 2.2 Minimal Intervention Contract

To isolate this hypothesis without introducing architectural confounders:
- **Masking Target:** Mask only the direct canonical phone embedding $\mathbf{e}_{\text{phone}}(y_i)$ entering the Binary MLP concatenation.
- **Preserved Attention Mechanism:** The canonical phone embeddings used to construct cross-attention queries are **not** masked and remain completely unperturbed. Cross-attention query projection and attention weight computations remain identical.
- **Preserved Acoustic Representation:** The aligned acoustic context vector $\mathbf{c}_i$ is untouched at the concatenation point.
- **Masking Mechanism:** Full-vector zero masking (all dimensions of embedding vector set to zero) applied independently per token position with probability $p=0.25$.
- **No Rescaling:** Mask-to-zero without inverted dropout rescaling ($1/(1-p)$), maintaining explicit counterfactual phone-absence representation during training.
- **Token Masking Scope:** Applied strictly at valid non-padding phoneme positions ($i < L_{\text{valid}}$). Padding positions remain unmasked and padded.
- **Training vs. Inference:** Active exclusively during training (`model.train()`). Strictly disabled during evaluation (`model.eval()`, where $p=0.00$).
- **Parity at $p=0$:** Setting $p=0.00$ mathematically and structurally reproduces the exact original forward pass.
- **Strict Architecture Boundary:** No learnable acoustic residual gates, no contrastive audio-phoneme loss, no new network heads, and no modifications to Wav2Vec2 or the diagnosis head. The multi-task loss remains CTC Loss + Binary BCE Loss with fixed 1.0 : 1.0 loss weights.

---

## 3. Controlled Experiment Matrix and Estimands

### 3.1 Experimental Arms

The previous design conflated fine-tuning with dropout by comparing a frozen baseline against a fine-tuned model. DESIGN-R1 corrects this defect by introducing an identical no-dropout fine-tuning control:

| Arm Identifier | Arm Name | Model Weight Initialization | Binary Prompt Masking | Optimizer / Scheduler Reset | Training Partition & Exposure | Role in Experimental Design |
|---|---|---|:---:|:---:|---|---|
| **Reference R** | Immutable Historical Baseline | RQ1-TRN-02A `best.pt` (SHA-256: `a2445c8d...`) | None ($p=0.00$) | None (0 updates) | None (Frozen at historical epoch 6) | Historical research anchor; validates metric reproducibility and protocol stability. |
| **Control C0** | No-Dropout Fine-Tuning Control | RQ1-TRN-02A `best.pt` (SHA-256: `a2445c8d...`) | None ($p=0.00$) | Re-initialized from step 0 per shared contract | Frozen Fold 1 TRAIN; matched budget (max 6 epochs / 3,600 updates) | Measures the effect of continued fine-tuning and optimizer updates without intervention. |
| **Treatment C1** | Binary Prompt Dropout Fine-Tuning | RQ1-TRN-02A `best.pt` (SHA-256: `a2445c8d...`) | Full-vector zero masking ($p=0.25$) | Re-initialized from step 0 per shared contract | Frozen Fold 1 TRAIN; matched budget (max 6 epochs / 3,600 updates) | Evaluates the causal effect of prompt embedding dropout under matched fine-tuning exposure. |

### 3.2 Formal Estimands and Attribution Rules

To ensure scientific validity and avoid misattributing general fine-tuning effects to the dropout mechanism:

1. **Primary Controlled Comparison ($\Delta_{\text{dropout}} = \text{C1} - \text{C0}$):**
   - Evaluates the incremental effect attributable specifically to the prompt embedding dropout intervention under identical additional training exposure, optimizer trajectories, and dataset passes.
   - This is the **only** contrast authorized to evaluate the dropout hypothesis.
2. **Fine-Tuning Drift Baseline ($\Delta_{\text{tuning}} = \text{C0} - \text{R}$):**
   - Quantifies performance shifts, overfitting, or calibration drift resulting strictly from additional optimizer updates on Fold 1 TRAIN without any architectural intervention.
3. **Compound Re-training Contrast ($\Delta_{\text{compound}} = \text{C1} - \text{R}$):**
   - Reflects the combined net effect of additional fine-tuning plus prompt dropout relative to the historical baseline.
   - **Methodological Rule:** $\text{C1} - \text{R}$ must **never** be characterized or reported as the causal effect of prompt dropout.
4. **Scope Exclusions:**
   - Sensitivity arm $p=0.50$ is excluded from the initial authorization.
   - Multi-seed runs (beyond seed `1337`) and cross-fold expansions (Folds 2, 3, 4) are deferred to separately authorized follow-on phases.

---

## 4. Shared Training Contract and Frozen Non-Intervention Variables

Control C0 and Treatment C1 must share an identical training environment and contract. Only the prompt dropout rate ($p=0.00$ vs $p=0.25$) may differ:

| Variable | Shared Specification | Enforcement & Verification Rule |
|---|---|---|
| **Weight Initialization** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | Verified by SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` prior to training. |
| **Trainable Parameter Groups** | `ctc_head`, `cross_attention_module`, `binary_head` | Identical parameter group names and order; Wav2Vec2 and diagnosis parameters strictly excluded. |
| **Backbone State** | Pretrained Wav2Vec2 base 960h | `requires_grad = False`, `.eval()` mode enforced during forward/backward. |
| **Diagnosis Head State** | Untrained | Zero gradients, excluded from loss computation and forward passes. |
| **Random Seed** | `1337` | Set across Python `random`, `numpy.random`, PyTorch CPU, and CUDA RNGs. |
| **Data Partition** | Frozen Fold 1 (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`) | SHA-256 verified; 2,399 train utterances, 300 validation utterances. |
| **Supervision Mode** | Oracle canonical phoneme sequences | Identical canonical prompts and alignments from frozen fold package. |
| **Optimizer** | PyTorch AdamW | $\text{lr} = 1.0 \times 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1.0 \times 10^{-8}$, $\text{weight\_decay} = 0.01$. |
| **Optimizer Reset Policy** | Cold reset from step 0 | Momentum and second-moment buffers re-initialized; historical 02A optimizer state discarded. |
| **LR Scheduler** | Cosine Annealing with Warm-up | Linear warm-up for 100 successful updates, followed by cosine decay to 0 over 3,600 updates. |
| **Batch Size & Accumulation** | Microbatch size 2, gradient accumulation 2 | Effective batch size = 4 utterances per parameter update. |
| **Loss Objectives & Weights** | Multi-task CTC + Binary BCE | $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{CTC}} + 1.0 \cdot \mathcal{L}_{\text{Binary}}$. |
| **Gradient Clipping** | Maximum $L_2$ norm = 1.0 | Applied via `torch.nn.utils.clip_grad_norm_` after gradient unscaling. |
| **Mixed Precision (AMP)** | PyTorch AMP (FP16) | `torch.cuda.amp.autocast(dtype=torch.float16)` and `torch.cuda.amp.GradScaler()`. |
| **Dataloader Sequence** | Deterministic sample shuffling | Dataloader generator seeded to ensure identical utterance presentation order in C0 and C1. |
| **Mask RNG Stream** | Isolated PyTorch Generator | Prompt dropout masks in C1 drawn from an independent RNG stream so dataloader shuffle is unaffected. |
| **Validation Metrics & Policy** | Protocol O Binary AP and CTC PER | Evaluated at every epoch boundary using identical evaluation code and tie-breaking. |

### 4.1 Reproducibility and Nondeterminism Auditing

Both arms will execute with deterministic PyTorch configurations (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`). However, speech ML workloads involving CTC loss (`torch.nn.CTCLoss`) and atomic GPU scatter/gather operations possess inherent non-deterministic GPU kernel paths.

To ensure complete scientific auditability, each run will record:
- Full Python, NumPy, PyTorch CPU, and CUDA RNG states at startup and epoch boundaries.
- Operating system, kernel version, Python version, PyTorch version, torchvision/torchaudio versions.
- NVIDIA driver version, CUDA toolkit version, cuDNN version, and GPU hardware device properties (T4 UUID).
- Bitwise reproducibility is targeted for dataloader indexing and CPU masking, while GPU kernel non-determinisms are documented in the run ledger.

---

## 5. Corrected Compute Budget and Optimizer Update Arithmetic

### 5.1 Step-by-Step Update Arithmetic

The prior design document asserted that Fold 1 training involved "11 epochs / 3,600 successful updates". This statement was mathematically contradictory. The verified arithmetic for Fold 1 is as follows:

1. **TRAIN Partition Size:** Exactly 2,399 utterances.
2. **Microbatch Size:** 2 utterances per forward microbatch.
3. **Microbatches per Epoch:**
   $$\text{Microbatches/Epoch} = \left\lceil \frac{2,399}{2} \right\rceil = 1,200 \text{ microbatches}$$
   (comprising 1,199 microbatches of size 2 and 1 trailing microbatch of size 1).
4. **Gradient Accumulation:** 2 microbatches per optimizer step.
5. **Optimizer Steps per Epoch:**
   $$\text{Updates/Epoch} = \frac{1,200}{2} = 600 \text{ parameter updates per epoch}$$
   (assuming no AMP gradient unscaling skips due to non-finite gradients).
6. **Epoch vs. Update Trajectory:**
   - **6 Epochs:** $6 \times 600 = 3,600 \text{ successful updates}$.
   - **11 Epochs:** $11 \times 600 = 6,600 \text{ successful updates}$.

Therefore, a budget of 3,600 updates corresponds to **6 epochs**, while an 11-epoch schedule would generate **6,600 updates**. DESIGN-R1 eliminates this inconsistency.

### 5.2 Dual-Limit Budget Enforcement

Both C0 and C1 are governed by strict per-arm and aggregate budget limits:

| Budget Level | Maximum Epochs | Hard Cap on Successful Updates | Enforcement Action upon Cap |
|---|:---:|:---:|---|
| **Per-Arm Cap (C0)** | 6 epochs | 3,600 successful updates | Training halts immediately when either limit is reached. |
| **Per-Arm Cap (C1)** | 6 epochs | 3,600 successful updates | Training halts immediately when either limit is reached. |
| **Aggregate Experiment Cap** | 12 arm-epochs | 7,200 successful updates | Total combined compute budget for RQ1-TRN-02B. |

### 5.3 Accounting for Attempted vs. Successful Parameter Updates

Because mixed-precision training employs `torch.cuda.amp.GradScaler`, gradient overflow or underflow can produce non-finite values (Inf/NaN). When this occurs:
- `scaler.step(optimizer)` skips the parameter update.
- `scaler.update()` adjusts the scale factor downwards.
- Parameter values remain unchanged for that accumulation cycle.

**Logging Requirement:**
The training loop must track and log three distinct counters:
1. `optimizer_step_attempts`: Total accumulation cycles where `scaler.step` was invoked.
2. `successful_optimizer_updates`: Accumulation cycles where gradients were finite and parameters were updated.
3. `skipped_optimizer_updates`: Accumulation cycles where updates were skipped due to Inf/NaN gradients.

The hard cap of 3,600 updates per arm applies strictly to `successful_optimizer_updates`.

### 5.4 Matched-Budget Comparison Point and Early Stopping Policy

- **Primary Matched-Budget Checkpoint:** The primary controlled comparison $\text{C1} - \text{C0}$ is evaluated at **Epoch 6** (or 3,600 successful updates, or the last common completed epoch if an authorized safety stop intervenes). Both arms must have identical data exposure.
- **Exploratory Secondary Checkpoint:** The validation-selected `best.pt` (highest validation AP) will be saved and evaluated as an exploratory secondary analysis.
- **Patience=5 Interaction:** The baseline training configuration specifies `patience = 5`. With a maximum budget of 6 epochs, early stopping cannot trigger before Epoch 6 unless validation performance declines immediately and continuously from Epoch 1.
- **Controlled Exposure Rule:** Under no circumstances will one arm be terminated early due to validation AP heuristics while the other continues training. Unequal training exposure must never masquerade as an experimental treatment effect.

---

## 6. Evaluation Contract and Scientific Limitations

### 6.1 Primary Ablation Research Question

> **Does Binary-only prompt embedding dropout ($p=0.25$) improve acoustic discrimination relative to an otherwise identical no-dropout fine-tuning control ($p=0.00$) under matched training exposure?**

### 6.2 Primary Reporting Suite (Protocol O Oracle Alignment)

Control C0 and Treatment C1 will be evaluated on the frozen Fold 1 validation partition (300 utterances; speakers `NJS` and `RRBI`) using identical token positions and scoring scripts. The evaluation report must provide:

1. **Binary AP under Matched Audio:** Precision-Recall curve area evaluated with ground-truth aligned acoustic audio.
2. **Binary AP under Deterministic Mismatched Audio:** Precision-Recall curve area evaluated under the frozen same-speaker shift-1 audio permutation policy.
3. **Matched-minus-Mismatched AP Gap ($\Delta_{\text{gap}} = \text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}}$):** Direct metric of acoustic reliance.
4. **AP against Train-Only Canonical Prior:** Performance of the baseline phone error prior (`0.303063`) evaluated on the identical validation token subset.
5. **Paired C1-minus-C0 Differences:**
   - $\Delta \text{AP}_{\text{matched}} = \text{AP}_{\text{matched}}(\text{C1}) - \text{AP}_{\text{matched}}(\text{C0})$.
   - $\Delta \text{AP}_{\text{gap}} = \Delta_{\text{gap}}(\text{C1}) - \Delta_{\text{gap}}(\text{C0})$.
6. **Class-Conditional Decision Shifts:** Confusion matrices (TP, FP, FN, TN), sensitivity, specificity, positive-class agreement, and total decision agreement evaluated at the fixed historical threshold $\tau = 0.50$ (with historical reference $\tau = 0.20$).
7. **Disaggregated Per-Speaker Results:** Independent reporting of all metrics for speaker `NJS` (150 utterances) and speaker `RRBI` (150 utterances).
8. **Greedy CTC PER Guardrail:** Validation Phoneme Error Rate under greedy decoding. Degradation exceeding 1.0 absolute PER percentage point relative to the reproducibly measured Reference R historical anchor constitutes a guardrail failure.

### 6.3 Scientific Attribution Criteria

A higher matched AP in C1 is **necessary but not sufficient** to demonstrate improved acoustic conditioning:
- If C1 achieves higher $\text{AP}_{\text{matched}}$ but exhibits an identical or smaller $\Delta_{\text{gap}}$, the improvement indicates enhanced memorization of prompt priors or altered score calibration, not increased acoustic reliance.
- Supportive evidence for the hypothesis requires:
  1. $\text{AP}_{\text{matched}}(\text{C1}) > \text{AP}_{\text{matched}}(\text{C0})$, **and**
  2. $\Delta_{\text{gap}}(\text{C1}) > \Delta_{\text{gap}}(\text{C0})$.

### 6.4 Statistical Uncertainty and Confidence Intervals

- **Utterance-Cluster Bootstrap:** 1,000 bootstrap resamples over the 300 validation utterances.
- **Stratified Resampling:** Resampling conducted within speaker strata (150 from `NJS`, 150 from `RRBI`).
- **Paired Evaluation:** Identical resampled utterance indices applied simultaneously across C0, C1, matched, and mismatched conditions to compute paired deltas.
- **Reporting:** Report 95% percentile bootstrap confidence intervals.
- **Prohibition:** Token-i.i.d. confidence intervals are strictly prohibited because token errors within an utterance are correlated.

### 6.5 Secondary Evaluation: Protocol Z (Zero-Oracle G2P Input Shift)

Protocol Z evaluates the models under realistic zero-oracle inference conditions:
- **Frontend:** CMUdict G2P phoneme conversion (frozen dictionary SHA-256: `81917843...`).
- **Zero Gold Timestamps:** No TextGrid timestamps, durations, or gold phone labels enter model forward passes.
- **Evaluation Population:** Aligned-only token positions (baseline coverage: 9,919 / 9,956 valid positions = `99.63%`).
- **Scoring:** Gold labels are utilized strictly offline after inference for scoring and metric computation.
- **Secondary Sensitivity:** Imputation strategies (Policy B1) reported strictly as labeled secondary sensitivity analyses.
- **Strict Boundary:** No Protocol Z G2P fine-tuning. Protocol Z metrics must **never** be used for checkpoint selection, hyperparameter tuning, or threshold optimization.

### 6.6 Explicit Scientific Limitations

The experimental design candidly acknowledges the following methodological limitations:
1. **Validation-Selected Initialization:** Both C0 and C1 initialize from `best.pt`, which was selected on the Fold 1 validation partition, introducing potential historical selection bias.
2. **Single-Seed Limitation:** The initial authorization executes a single seed (`1337`). Random initialization variance in the projection heads cannot be assessed until multi-seed replication is authorized.
3. **Limited Speaker Diversity:** Validation encompasses exactly two non-native speakers (`NJS`, `RRBI`).
4. **Generalization Boundary:** Paired utterance-cluster bootstrap intervals quantify utterance sampling uncertainty within these two speakers, but **do not** demonstrate cross-speaker population generalizability.
5. **Optimism Bias:** Checkpoint selection using validation AP inevitably induces optimism bias on validation metrics.
6. **Zero-Oracle Asymmetry:** Protocol Z achieves zero-oracle model inference, but offline performance measurement still relies on expert gold annotations. Results do not establish readiness for unconstrained clinical diagnosis.

---

## 7. Colab Execution Planning and Operational Safety

### 7.1 Independent Experimental Runs

To guarantee operational isolation and prevent data contamination, C0 and C1 will execute as separate, sequential Colab runs:

| Run Parameter | Control Arm (`RQ1-TRN-02B-C0`) | Treatment Arm (`RQ1-TRN-02B-C1`) |
|---|---|---|
| **Run Identifier** | `RQ1-TRN-02B-C0` | `RQ1-TRN-02B-C1` |
| **Configuration Path** | `configs/rq1_trn_02b_c0.yaml` | `configs/rq1_trn_02b_c1.yaml` |
| **Binary Mask Probability** | $p = 0.00$ | $p = 0.25$ |
| **Local VM Checkpoint Dir** | `/content/checkpoints/RQ1-TRN-02B/C0/` | `/content/checkpoints/RQ1-TRN-02B/C1/` |
| **Persistent Drive Destination** | `.../Langfens/checkpoints/RQ1-TRN-02B/C0/` | `.../Langfens/checkpoints/RQ1-TRN-02B/C1/` |
| **Log and Prediction Path** | `/content/logs/RQ1-TRN-02B/C0/` | `/content/logs/RQ1-TRN-02B/C1/` |
| **Historical Directory Isolation** | Never write to `checkpoints/RQ1-TRN-02A/` | Never write to `checkpoints/RQ1-TRN-02A/` |

### 7.2 Compute Resource Estimation and Quota Boundary

- **Hardware Platform:** Google Colab T4 GPU (16 GB VRAM).
- **Estimated Runtime per Arm:** Approximately 3 to 5 T4 GPU-hours (inclusive of VM staging, 6 training epochs, per-epoch validation, and evaluation passes).
- **Total Estimated Compute:** Approximately 6 to 10 T4 GPU-hours combined.
- **Quota Verification Rule:** Prior to launching C0, remaining compute units and authorization must be verified via non-interactive Colab CLI inspection (`colab usage`).
- **Mutual Dependency Rule:** If available quota is insufficient to complete **both** C0 and C1, allocation must not proceed. Executing C1 alone without C0 provides zero causal interpretability and is strictly prohibited.
- **No Local Fallback:** Never fall back to local CPU or local GPU training if Colab resources are unavailable. Stop and report blockers.

### 7.3 Ephemeral VM Staging and Audio Storage Hygiene

- **Audio Staging:** The reusable Fold 1 train/validation audio package must be copied from persistent Google Drive storage to the ephemeral VM disk (`/content/data/`) and its SHA-256 verified prior to training.
- **No Network Streaming:** Training dataloaders must read audio WAV files directly from the local VM NVMe/SSD. Streaming training audio across the Google Drive FUSE mount (`/content/drive/`) is strictly prohibited.
- **Official Test Sealing:** Official test partition audio and TextGrids must remain unextracted, unstaged, and inaccessible on the VM.

### 7.4 Atomic Checkpoint Persistence and Resume Continuity

- **Per-Epoch Checkpointing:** At the end of every epoch, `last.pt` and (if validation AP improves) `best.pt` must be saved locally and synced immediately to the arm's Google Drive persistent directory.
- **Atomic Sync Verification:** Checkpoint synchronization must verify file presence and SHA-256 integrity on Google Drive before the training loop proceeds to the subsequent epoch.
- **Checkpoint State Dictionary:** Each checkpoint file must contain:
  - `model_state_dict`: Model weights.
  - `optimizer_state_dict`: AdamW optimizer state.
  - `scheduler_state_dict`: Cosine scheduler state.
  - `scaler_state_dict`: AMP GradScaler state.
  - `rng_state`: Python, NumPy, PyTorch CPU, and CUDA RNG states.
  - `metadata`: Git commit SHA, config snapshot, dataset package SHA, epoch index, successful update count, attempted update count, seed, mask rate $p$, and validation metric history.

---

## 8. Operational Stop Conditions and Implementation Roadmap

### 8.1 Explicit Stop Conditions

Training or evaluation must immediately halt, release GPU resources, and alert the engineering team upon encountering any of the following conditions:

1. **Manifest or Checkpoint Mismatch:** Initial checkpoint SHA-256 differs from `a2445c8d...`, or split manifest SHA-256 differs from `7d6bfdf6...`.
2. **Data Leakage or Test Access:** Any attempt to read, stage, or evaluate utterances from the 6 sealed test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
3. **Timestamp Leakage:** TextGrid boundary timestamps entering model inference forward passes.
4. **Architectural Contamination:** Pretrained Wav2Vec2 parameters unfreezing (`requires_grad == True`), or diagnosis head receiving gradients.
5. **Numerical Instability:** Occurrence of unrecoverable NaN or Inf values in loss tensors or gradient norms.
6. **PER Guardrail Failure:** Greedy CTC PER exceeding the historical anchor by more than 1.0 absolute percentage point at any checkpoint.
7. **Persistence Failure:** Failure to write, sync, or verify SHA-256 of `last.pt` on Google Drive.
8. **Budget Cap Reached:** Optimizer reaching 3,600 successful updates or 6 completed epochs in either arm, or 7,200 updates across the project.
9. **Single-Arm Isolation:** Control C0 failing or being aborted, which renders C1 execution invalid.

### 8.2 Future Implementation Phases (Post-Design Authorization)

Execution of this design is partitioned into explicit sequential phases. Proceeding to any subsequent phase requires explicit operator authorization:

- **Phase 1: Architecture and Unit Tests (Local Laptop)**
  - Implement the optional prompt embedding dropout flag in `src/langfens_prep/model.py`.
  - Validate exact bitwise mathematical parity at $p=0.00$ against the baseline model.
  - Implement unit tests verifying: (a) padding tokens are never masked, (b) cross-attention query embeddings remain unmasked, (c) aligned acoustic vectors remain unperturbed, (d) Wav2Vec2 parameters remain frozen, and (e) diagnosis head parameters receive zero gradients.
- **Phase 2: Configuration and Entrypoint Preparation (Local Laptop)**
  - Create isolated configurations `configs/rq1_trn_02b_c0.yaml` and `configs/rq1_trn_02b_c1.yaml`.
  - Implement dedicated training driver `audit_scripts/train_rq1_trn_02b.py` enforcing dual update caps, separate attempt/successful update accounting, and atomic Drive persistence.
- **Phase 3: Preflight Manifest and Dry-Run Audit (Local Laptop)**
  - Conduct full offline preflight audit of configs, hashes, split manifests, and evaluation harnesses.
- **Phase 4: Authorized Colab Execution (Google Colab T4)**
  - Sequential execution of `RQ1-TRN-02B-C0` followed by `RQ1-TRN-02B-C1`.
- **Phase 5: Paired Evaluation and Scientific Report (Local Laptop)**
  - Execute paired Protocol O and Protocol Z evaluations and compute bootstrap confidence intervals.

---

## 9. Final Authorization Boundary

**This document represents a finalized scientific design specification.**

**EXPLICIT PROHIBITION:**
No model code changes, training scripts, optimizer updates, GPU allocations, Colab runtime launches, or official test evaluations are authorized by this document. Implementation and operational execution will commence only upon explicit, written operator instruction.
