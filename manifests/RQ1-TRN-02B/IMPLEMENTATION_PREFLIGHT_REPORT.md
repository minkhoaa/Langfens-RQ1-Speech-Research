# LANGFENS — RQ1-TRN-02B Implementation Preflight Report

**Document ID:** `RQ1-TRN-02B-IMPLEMENTATION-PREFLIGHT`  
**Status:** COMPLETE / PASS  
**Branch:** `feat/rq1-trn-02b-implementation-preflight`  
**Parent Baseline Commit:** `c4a658ffbd5a2afd2ebdce172e47af43ed9bb63e`  
**Design Reference Commit:** `428061de9d0b18ac292e0c31008047e757de6ef0`  
**Execution Scope:** Local Laptop (CPU Forward-Only). STRICTLY NO GPU training, backward passes, optimizer updates, Colab runtime launches, or official test partition evaluations occurred.

---

## 1. Acceptance Gate Decision

### **Gate Result: PASS**

The approved RQ1-TRN-02B experimental design has been fully implemented, static-inspected, and verified with 13 CPU forward-only regression tests and 136 full-suite regression tests. 

All core deliverables are complete and verified:
1. **Stage A:** Erratum in `ACOUSTIC_CONDITIONING_DESIGN.md` corrected (Fold 1 train partition confirmed as 16 speakers, 2,399 utterances).
2. **Stage B:** Controlled Binary-only prompt embedding dropout implemented in `src/langfens_prep/model.py` with 10/10 verified properties, including bitwise baseline parity at $p=0.00$, isolated RNG stream, and unperturbed cross-attention/acoustic vectors.
3. **Stage C:** Both experimental arm configs (`configs/rq1_trn_02b_c0.yaml` and `configs/rq1_trn_02b_c1.yaml`) and the dedicated training entrypoint (`audit_scripts/train_rq1_trn_02b.py`) created.
4. **Stage D:** Strict matched-budget training arithmetic enforced (6 epochs / 3,600 successful updates per arm; 7,200 max combined) with explicit attempted, successful, and skipped update accounting.
5. **Stage E:** Checkpoint schema and atomic Google Drive synchronization with remote SHA-256 verification implemented.
6. **Stage F:** Paired scientific evaluation harness created (`audit_scripts/evaluate_rq1_trn_02b.py`) implementing matched/mismatched AP, $\Delta_{\text{gap}}$, paired estimands ($\Delta_{\text{dropout}}$, $\Delta_{\text{tuning}}$, $\Delta_{\text{compound}}$), 1,000 speaker-stratified bootstrap iterations, CTC PER guardrails, confusion matrices, and Protocol Z aligned-only evaluation.
7. **Stage G:** 13 CPU forward-only regression tests implemented and passed (`tests/test_rq1_trn_02b_implementation_preflight.py`); GPU-only gradient/backward pass test explicitly marked as skipped.
8. **Stage H:** Comprehensive preflight ledger, risk analysis, and execution plan documented below.

---

## 2. Complete Source Diff Summary

### 2.1 Documentation Correction
- **`manifests/RQ1-TRN-02B/ACOUSTIC_CONDITIONING_DESIGN.md`**:
  - Corrected line 16 from "TRAIN: 2,399 utterances across 14 speakers" to "TRAIN: 2,399 utterances across 16 speakers".
  - Retained original split manifest and its verified SHA-256 (`7d6bfdf6...`).

### 2.2 Model Architecture (`src/langfens_prep/model.py`)
- **`LangfensMDDOutput`**:
  - Added optional field `intervention_mask: Optional[torch.Tensor] = None` with default `None`. Backward-compatible with existing consumers.
- **`LangfensWav2Vec2MDD.__init__`**:
  - Added `binary_prompt_dropout_p: float = 0.0`. Validates $0.0 \le p \le 1.0$.
  - Added `self.intervention_generator: Optional[torch.Generator] = None`.
- **RNG Generator Management**:
  - Added `set_intervention_generator(self, generator)` and `seed_intervention_generator(self, seed, device)`.
- **`LangfensWav2Vec2MDD.forward`**:
  - Added `binary_prompt_dropout_p: Optional[float] = None` and `intervention_generator: Optional[torch.Generator] = None`.
  - When `self.training` is False or effective $p = 0.0$: strictly bypasses masking and passes unmasked `phone_embeds` directly (100% bitwise parity with original baseline).
  - When `self.training` and effective $p > 0.0$:
    - Identifies valid phoneme tokens using `canonical_phone_lengths` (or non-zero indices).
    - Samples Bernoulli drop mask exclusively over valid token positions (`num_valid` draws). Padded positions never consume RNG draws.
    - Zeroes complete lexical embedding vector ($128$ dimensions) at selected valid positions.
    - Applies NO inverted dropout scaling ($1/(1-p)$). Kept tokens have scaling factor $1.0$.
    - Does NOT mutate `phone_embeds` in place; creates a new tensor `binary_phone_embeds`.
    - Aligned acoustic representations (`aligned_acoustic`) and cross-attention queries are untouched.
    - Masked embeddings enter `self.binary_head(torch.cat([aligned_acoustic, binary_phone_embeds], dim=-1))`.

### 2.3 Training Entrypoint (`audit_scripts/train_rq1_trn_02b.py`)
- Dedicated standalone driver isolated from RQ1-TRN-02A.
- Validates split manifest SHA-256 and initial checkpoint SHA-256 before doing any work.
- Loads exact weights from `checkpoints/RQ1-TRN-02A/recovered/best.pt`.
- Cold reset of AdamW optimizer, cosine annealing scheduler (100 warmup, 3,600 updates), and AMP GradScaler.
- Freezes Wav2Vec2 backbone (`requires_grad = False`, `.eval()` mode).
- Freezes diagnosis head (`requires_grad = False`, zero loss, zero gradients asserted).
- Implements `UpdateBudgetTracker`: explicit counters for `step_attempts`, `successful_updates`, `skipped_updates`, and `consecutive_skips`.
- Primary endpoint termination: stops training when `successful_updates >= 3600` or `epoch >= 6`.
- Atomic local checkpoint saving and Google Drive synchronization with remote SHA-256 verification.
- Enforces execution safety: requires CUDA on Colab runtime; halts with explicit error if run on local CPU without `--dry-run`.

### 2.4 Configuration Artifacts
- `configs/rq1_trn_02b_c0.yaml`: Control arm configuration ($p=0.00$).
- `configs/rq1_trn_02b_c1.yaml`: Treatment arm configuration ($p=0.25$).

### 2.5 Evaluation Harness (`audit_scripts/evaluate_rq1_trn_02b.py`)
- Independent paired evaluator executing on validation partition (speakers `NJS` and `RRBI`).
- Computes:
  - Matched-audio Binary AP.
  - Same-speaker shift-1 derangement audio Binary AP.
  - Matched-minus-mismatched AP gap ($\Delta_{\text{gap}}$).
  - Train-only canonical empirical prior AP.
  - C1 - C0 primary causal contrast ($\Delta_{\text{dropout}}$).
  - C0 - R fine-tuning drift ($\Delta_{\text{tuning}}$).
  - C1 - R compound re-training contrast ($\Delta_{\text{compound}}$).
  - Confusion matrices at $\tau = 0.50$ and $\tau = 0.20$ (TP, FP, FN, TN, precision, recall, specificity, F1, accuracy).
  - Greedy CTC PER guardrail (< 1.0% degradation vs Reference R historical anchor 23.64%).
  - 1,000 speaker-stratified paired utterance-cluster bootstrap iterations with 95% percentile CIs.
  - Parameter immutability verification (SHA-256 weights fingerprint before and after evaluation).
  - Outputs saved to `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json`.

---

## 3. Configuration Comparison: C0 versus C1

| Parameter / Field | Control Arm (`C0`) | Treatment Arm (`C1`) | Scientific Matching Status |
|---|:---:|:---:|:---:|
| **Experiment ID** | `RQ1-TRN-02B-C0` | `RQ1-TRN-02B-C1` | Distinct isolated run identifier |
| **Arm Identifier** | `C0` | `C1` | Controlled ablation condition |
| **Random Seed** | `1337` | `1337` | **Identical** |
| **Fold Index** | `1` | `1` | **Identical** |
| **Policy** | `strict` | `strict` | **Identical** |
| **Manifest Path** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | **Identical** (SHA-256 `7d6bfdf6...`) |
| **Initial Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | **Identical** (SHA-256 `a2445c8d...`) |
| **Pretrained Model Backbone** | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | **Identical** |
| **Pretrained Revision** | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | **Identical** |
| **Binary Prompt Dropout $p$** | **`0.00`** | **`0.25`** | **Isolated Experimental Treatment** |
| **Multi-task Loss Weights** | `ctc: 1.0, binary: 1.0` | `ctc: 1.0, binary: 1.0` | **Identical** |
| **Backbone Frozen** | `true` | `true` | **Identical** |
| **Diagnosis Active** | `false` | `false` | **Identical** |
| **Microbatch Size** | `2` | `2` | **Identical** |
| **Gradient Accumulation** | `2` | `2` | **Identical** |
| **Effective Batch Size** | `4` utterances | `4` utterances | **Identical** |
| **Optimizer** | `AdamW` | `AdamW` | **Identical** |
| **Learning Rate** | `0.001` ($1.0 \times 10^{-3}$) | `0.001` ($1.0 \times 10^{-3}$) | **Identical** |
| **Weight Decay** | `0.01` | `0.01` | **Identical** |
| **Scheduler** | `cosine` with warm-up | `cosine` with warm-up | **Identical** |
| **Warm-up Updates** | `100` | `100` | **Identical** |
| **Max Training Epochs** | `6` | `6` | **Identical** |
| **Max Successful Updates** | `3,600` | `3,600` | **Identical** |
| **Patience Heuristic** | `5` | `5` | **Identical** |
| **Gradient Clip Norm** | `1.0` ($L_2$) | `1.0` ($L_2$) | **Identical** |
| **AMP Mode** | `fp16` | `fp16` | **Identical** |
| **GradScaler Initial Scale** | `4096` | `4096` | **Identical** |
| **Output Directory** | `checkpoints/RQ1-TRN-02B/C0` | `checkpoints/RQ1-TRN-02B/C1` | **Isolated Directory** |
| **Google Drive Directory** | `.../manifests/RQ1-TRN-02B/checkpoints/C0` | `.../manifests/RQ1-TRN-02B/checkpoints/C1` | **Isolated Directory** |

---

## 4. Intervention Verification (10-Point Contract)

The prompt embedding dropout intervention satisfies all 10 architectural criteria:

| Requirement | Implementation Mechanism | Verification Evidence |
|---|---|---|
| **1. Per-Token Bernoulli Mask** | `drop_draws = torch.bernoulli(probs, generator=gen)` | Sampled per valid token $(b, i)$ independently at rate $p$. Verified in `test_4`. |
| **2. Complete Lexical Vector Zeroing** | `binary_phone_embeds = phone_embeds * intervention_mask.unsqueeze(-1)` | All 128 embedding dimensions zeroed when dropped (`[0, ..., 0]`). Verified in `test_3`. |
| **3. No Inverted Dropout Scaling** | Multiplied strictly by $\{0.0, 1.0\}$; no division by $(1-p)$ | Values strictly subset of $\{0.0, 1.0\}$. Counterfactual absence maintained. Verified in `test_4`. |
| **4. Mask Applied Only to Binary Head** | `binary_input = torch.cat([aligned_acoustic, binary_phone_embeds], dim=-1)` | Mask applied exclusively to vector entering `binary_head`. Verified in `test_3`. |
| **5. Preserved Attention Queries** | `cross_attention_module` called with raw `phone_embeds` prior to masking | Attention queries and attention weights bitwise identical to unmasked model. Verified in `test_3`, `test_7`. |
| **6. No In-Place Mutation** | New tensor `binary_phone_embeds` created via elementwise product | `phone_embeds` remains immutable; original tensors unmutated. Verified in `test_3`. |
| **7. Preserved Aligned Acoustics** | `aligned_acoustic` concatenated directly into `binary_input` | `aligned_acoustic` bitwise identical between $p=0.0$ and $p=0.25$. Verified in `test_3`. |
| **8. Padding Tokens Ignored** | `probs = torch.full((num_valid,), eff_p, ...)` | Sampled strictly for valid non-padding tokens. Zero RNG draws on padding. Verified in `test_4`. |
| **9. Disabled in Evaluation Mode** | Active strictly when `self.training and eff_p > 0.0` | `model.eval()` completely bypasses masking; `intervention_mask is None`. Verified in `test_6`. |
| **10. Parity at $p=0.00$** | Branch condition `eff_p > 0.0` returns unmodified `phone_embeds` | Exact bitwise logits and loss parity with original unmasked baseline. Verified in `test_1`. |

### RNG Stream Isolation Verification
- Intervention masks are drawn from an isolated PyTorch `Generator` instance (`model.intervention_generator`).
- **Direction A (Global $\to$ Intervention):** Mutating global torch RNG and numpy RNG leaves the intervention mask sequence 100% bitwise identical (`test_5`).
- **Direction B (Intervention $\to$ Global):** Sampling intervention masks does not mutate or advance the global PyTorch RNG state (`test_5`).
- Dataloader batch presentation order in C0 and C1 is identical and deterministic.

---

## 5. CPU Regression Test Results and Skipped GPU Tests

### 5.1 Preflight Test Census (`tests/test_rq1_trn_02b_implementation_preflight.py`)

Execution command:
```bash
.venv/bin/pytest -v tests/test_rq1_trn_02b_implementation_preflight.py
```

Results: **13 PASSED, 1 SKIPPED, 0 FAILED** (Duration: 4.44s)

| Test Identifier | Test Name | Status | Verified Property |
|---|---|:---:|---|
| Test 1 | `test_1_baseline_p0_forward_equivalence` | **PASS** | Exact bitwise parity at $p=0.00$ in train and eval modes |
| Test 2 | `test_2_identical_c0_c1_initial_state_dictionaries` | **PASS** | Production `best.pt` loads identically for C0 and C1 |
| Test 3 | `test_3_exact_intervention_location` | **PASS** | Aligned acoustic and attention weights identical; binary logits diverge |
| Test 4 | `test_4_token_mask_shape_and_padding_handling` | **PASS** | Mask shape $(B, L)$, binary values $\{0, 1\}$, padding tokens never dropped |
| Test 5 | `test_5_isolated_rng_reproducibility` | **PASS** | Isolated generator reproducibility; no global RNG leakage |
| Test 6 | `test_6_dropout_disabled_in_eval` | **PASS** | Masking strictly disabled when `model.eval()` is set |
| Test 7 | `test_7_unmodified_cross_attention_queries` | **PASS** | Cross-attention query projection formed from unmasked embeddings |
| Test 8 | `test_8_frozen_backbone_and_diagnosis_exclusion` | **PASS** | Backbone frozen in eval; diagnosis head frozen with zero grad |
| Test 9 | `test_9_correct_trainable_parameter_inventory` | **PASS** | CTC head, cross-attention module, binary head trainable |
| Test 10 | `test_10_correct_optimizer_and_scheduler_configuration`| **PASS** | Static audit of C0 and C1 YAML hyperparameter specs |
| Test 11 | `test_11_update_budget_counter_logic` | **PASS** | Update tracker accounting, 3,600 step cap, 8-skip exception |
| Test 12 | `test_12_evaluation_population_and_sealed_test_safeguards`| **PASS** | 16 train spks, 2 val spks (NJS, RRBI), 6 test spks sealed |
| Test 13 | `test_13_checkpoint_schema_and_resume_metadata` | **PASS** | Full payload schema, required keys, atomic persistence |
| Guardrail | `test_gpu_backward_pass_marked_for_colab` | **SKIPPED** | Explicitly marked placeholder reserved for Google Colab GPU |

### 5.2 Full Repository Regression Suite

Execution command:
```bash
PYTHONPATH=.:src .venv/bin/pytest tests/
```

Results: **136 PASSED, 4 SKIPPED, 0 FAILED** (Duration: 102.77s)
- All historical parser, audio loader, normalizer, dataset, collator, training contract, reconciliation regression, and evaluator tests continue to pass 100%.
- Zero regressions introduced into the existing codebase.

---

## 6. Dataset and Checkpoint Hash Ledger

| Artifact Role | File Path | Expected SHA-256 Digest | Preflight Verification Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **VERIFIED MATCH** |
| **Initial Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **VERIFIED MATCH** |
| **Acoustic Backbone** | `facebook/wav2vec2-base-960h` | Revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | **PINNED REVISION** |
| **CMUdict Frontend** | `cmudict-0.7b` | SHA-256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | **PINNED G2P** |

---

## 7. Update Budget Calculation and Accounting

### 7.1 Arithmetic Derivation
- **Train partition ($N_{\text{train}}$):** 2,399 utterances.
- **Microbatch size ($B_{\text{micro}}$):** 2 utterances.
- **Microbatches per epoch:** $\lceil 2,399 / 2 \rceil = 1,200$ microbatches (1,199 of size 2, 1 of size 1).
- **Gradient accumulation ($K_{\text{accum}}$):** 2 microbatches per parameter update.
- **Updates per epoch:** $1,200 / 2 = 600$ optimizer updates per completed epoch.
- **Trajectory:**
  - 1 Epoch: 600 updates.
  - 2 Epochs: 1,200 updates.
  - 3 Epochs: 1,800 updates.
  - 4 Epochs: 2,400 updates.
  - 5 Epochs: 3,000 updates.
  - **6 Epochs:** $6 \times 600 = \mathbf{3,600 \text{ successful updates}}$ (Primary Endpoint).

### 7.2 Budget Caps
- **Per-Arm Limit:** Maximum 6 epochs; hard cap of **3,600 successful optimizer updates**.
- **Aggregate Experiment Limit:** Maximum 12 arm-epochs; hard cap of **7,200 successful optimizer updates combined**.
- **Update Accounting Rule:** The 3,600 cap applies strictly to `successful_updates`. `step_attempts` and `skipped_updates` are tracked independently in checkpoint metadata and training history.
- **Primary Controlled Comparison:** Evaluated strictly at the predeclared endpoint of **Epoch 6 / 3,600 successful updates**. If AMP gradient unscaling skips cause a discrepancy between C0 and C1 update counts, this discrepancy must be reported and primary causal interpretation suspended until resolved.

---

## 8. Estimated Colab Requirements and Checkpoint Storage

### 8.1 Compute Platform & Runtime
- **GPU Architecture:** Google Colab T4 GPU (16 GB VRAM).
- **Runtime per Arm:** Approximately 3.5 to 5.0 T4 GPU-hours (including disk staging, 6 training epochs, per-epoch validation passes, and final artifact synchronization).
- **Total Combined Runtime:** Approximately **7.0 to 10.0 T4 GPU-hours**.
- **Mutual Dependency Rule:** Control C0 and Treatment C1 form an inseparable scientific pair. C1 must never be trained or interpreted without an identically matched C0 run.

### 8.2 Checkpoint Storage Ledger
- Model checkpoint file size: Approximately **393.3 MB** per checkpoint.
- Expected checkpoints per arm:
  - `last.pt`: ~394 MB (overwritten every completed epoch).
  - `best.pt`: ~394 MB (overwritten when validation AP improves).
  - `fixed_budget_checkpoint.pt`: ~394 MB (written at Epoch 6 / 3,600 updates).
- Per-arm storage: ~1.2 GB.
- **Total persistent Google Drive storage (C0 + C1):** ~**2.4 GB**.

---

## 9. Exact Future Execution Commands (Not Executed)

*Notice: These commands are documented for operational reproducibility and protocol adherence only. No Colab commands, GPU allocations, or training steps were executed during this preflight task.*

### 9.1 Colab Quota Inspection (Pre-requisite)
```bash
# Verify available compute units before allocating runtime
colab usage
```

### 9.2 Control Arm C0 Execution (Colab GPU)
```bash
# 1. Provision T4 GPU runtime
colab new -s langfens-rq1-trn-02b-c0 --gpu T4

# 2. Stage repository, fold package, and production checkpoint onto local VM disk (/content)
# ... staging commands ...

# 3. Execute Control Arm C0 Training Driver
colab run -s langfens-rq1-trn-02b-c0 -- \
  python3 audit_scripts/train_rq1_trn_02b.py \
    --config configs/rq1_trn_02b_c0.yaml \
    --zip-path /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
    --drive-output-dir /content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C0

# 4. Promptly release runtime
colab stop -s langfens-rq1-trn-02b-c0
```

### 9.3 Treatment Arm C1 Execution (Colab GPU)
```bash
# 1. Provision fresh T4 GPU runtime
colab new -s langfens-rq1-trn-02b-c1 --gpu T4

# 2. Execute Treatment Arm C1 Training Driver
colab run -s langfens-rq1-trn-02b-c1 -- \
  python3 audit_scripts/train_rq1_trn_02b.py \
    --config configs/rq1_trn_02b_c1.yaml \
    --zip-path /content/data/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
    --drive-output-dir /content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C1

# 3. Promptly release runtime
colab stop -s langfens-rq1-trn-02b-c1
```

### 9.4 Paired Scientific Evaluation Execution (Local Laptop CPU)
```bash
# Execute paired counterfactual evaluation harness
python3 audit_scripts/evaluate_rq1_trn_02b.py \
  --ref-checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
  --c0-checkpoint checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt \
  --c1-checkpoint checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt \
  --manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
  --authorized-zip datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
  --output-dir manifests/RQ1-TRN-02B/evaluation \
  --device cpu
```

---

## 10. Known Risks, Blockers, and Stop Conditions

### 10.1 Operational Stop Conditions
Training or evaluation must immediately halt, release GPU resources, and alert the team upon encountering:
1. **Hash Mismatch:** Manifest SHA-256 differs from `7d6bfdf6...` or initial checkpoint SHA-256 differs from `a2445c8d...`.
2. **Official Test Partition Contamination:** Any attempt to load, read, or stage audio/TextGrids from test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
3. **Architectural Contamination:** Pretrained Wav2Vec2 parameters unfreezing (`requires_grad == True`), or diagnosis head receiving non-zero gradients.
4. **Numerical Instability:** Non-finite (NaN/Inf) loss tensors or 8 consecutive AMP-skipped optimizer updates.
5. **Greedy CTC PER Guardrail Degradation:** Validation CTC PER exceeding the Reference R historical anchor (23.64%) by more than 1.0 absolute percentage point.
6. **Persistence Verification Failure:** Failure to write, sync, or verify remote SHA-256 of `last.pt` on Google Drive.
7. **Compute Budget Exceeded:** Optimizer reaching 3,600 successful updates or 6 completed epochs in either arm.

### 10.2 Scientific Limitations Candidly Acknowledged
1. **Validation Selection Bias:** Initial checkpoint was selected on Fold 1 validation partition.
2. **Single-Seed Limitation:** Initial phase evaluates seed `1337`; multi-seed replication requires separate follow-on authorization.
3. **Limited Speaker Diversity:** Validation partition contains exactly 2 non-native speakers (`NJS`, `RRBI`). Paired bootstrap intervals quantify sampling uncertainty within these speakers, not population generalizability across unobserved accents.

---

## 11. Final Verification Sign-Off

- **Stage A (Design Erratum):** Verified PASS.
- **Stage B (Binary-Only Dropout):** Verified PASS.
- **Stage C (Dual Arm Configurations & Driver):** Verified PASS.
- **Stage D (Training Budgets & Accounting):** Verified PASS.
- **Stage E (Checkpoint Persistence Contract):** Verified PASS.
- **Stage F (Paired Evaluation Harness):** Verified PASS.
- **Stage G (CPU Forward-Only Regression Tests):** Verified PASS (13 passed, 1 skipped).
- **Stage H (Preflight Documentation):** Verified PASS.

**Preflight Status:** READY FOR OPERATOR REVIEW AND SUBSEQUENT COLAB EXECUTION AUTHORIZATION GATE.
