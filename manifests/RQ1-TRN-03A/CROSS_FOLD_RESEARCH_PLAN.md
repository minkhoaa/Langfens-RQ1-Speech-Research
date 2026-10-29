# LANGFENS — RQ1-TRN-03A: Cross-Fold Speaker-Independent Generalization Research Plan

**Document ID:** `LANGFENS-RQ1-TRN-03A-CROSS-FOLD-RESEARCH-PLAN`<br>
**Phase Identifier:** `RQ1-TRN-03A`<br>
**Target Research Question:** RQ1 — Speaker-Independent Mispronunciation Detection & Phonetic Generalization<br>
**Author:** Senior Speech ML Research Engineer & Reproducibility Auditor<br>
**Date:** 2026-10-29<br>
**Base Commit:** `7a3b9ecf5e9efedc480b477f531ebe9318ae3415` on branch `review/rq1-trn-03a-cross-fold-planning`<br>
**Status:** `READY_FOR_OPERATOR_REVIEW` (Planning Complete; Execution Blocked Pending Operator Authorization)

---

## 1. Executive Summary & Strategic Reprioritization

With the successful completion and closure of Phase RQ1-TRN-02B (where whole-phone prompt embedding dropout was conclusively determined to be `NOT_SUPPORTED` on Fold 1), Project Langfens transitions from localized single-fold architecture ablations to its primary scientific objective: **rigorous, corpus-wide speaker-independent mispronunciation detection (MDD)**.

### 1.1 Strategic Rationale
1. **Generalization Over Hyper-Tuning:** Single-fold evaluations (Fold 1) cannot establish how well the frozen Wav2Vec2 + multi-task MDD architecture generalizes across distinct human vocal tracts, fundamental frequency ($F_0$) distributions, and diverse first-language (L1) phonological transfer profiles.
2. **Pedagogical Alignment:** Computer-Assisted Pronunciation Training (CAPT) systems fail in practice if their decision boundaries overfit to a specific development cohort. Establishing robust, out-of-fold generalization across all 24 L2-ARCTIC speakers is a prerequisite for generating trustworthy pronunciation feedback.
3. **Thesis Proposal Fulfillment:** The master engineering thesis protocol specified 4-Fold Speaker-Independent Cross-Validation as the authoritative benchmark. Completing Folds 2, 3, and 4 completes the empirical foundation of RQ1.

### 1.2 Non-Negotiable Operational Boundaries
In strict adherence to `AGENTS.md`:
- **Local Laptop Boundary:** Limited strictly to configuration design, static manifest audits, staging script preparation, unit/regression testing, and post-training CPU metric aggregation. **Zero local training or backward passes.**
- **Remote GPU Boundary:** ALL training runs (Folds 2, 3, 4) must execute on authorized remote GPU hardware (Kaggle Tesla T4 $\times$ 2 or Google Colab T4) under CUDA FP16 AMP.
- **Fail-Closed Test Sealing:** All four fold test sets ($S_{\text{test}}^{(1..4)}$) remain **100% physically sealed and unstaged** during this validation phase.
- **Execution Stop:** **NO training, NO staging package creation, and NO GPU provisioning occurs during this planning step.** The process stops immediately for operator review and explicit sign-off.

---

## 2. Preservation of Completed Fold 1 Scientific Evidence

All historical findings, code paths, and checkpoints from RQ1-TRN-02A and RQ1-TRN-02B are preserved as immutable scientific records. Under no circumstances will prior evidence be overwritten, modified, or re-run.

### 2.1 Preserved Fold 1 Artifact Ledger

| Model Arm | Canonical Identifier | Source Platform | Updates / Epochs | Matched Val AP | CTC PER | Acoustic Gap | Authoritative Status |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Reference R** | `checkpoints/RQ1-TRN-02A/recovered/best.pt`<br>SHA-256 `a2445c8d...` | Colab T4 | 3,600 / Ep 6 (of 11) | **0.317021** | 23.68% | **+0.022299** | **Authoritative Primary Production Baseline** |
| **Control C0** | HF: `langfens-rq1-trn-02b-checkpoints`<br>`C0/fixed_budget_checkpoint.pt`<br>SHA-256 `0dc1d4fb...` | Kaggle T4 $\times$ 2 | 3,600 / Ep 6 | 0.310304 | **22.02%** | +0.017543 | **Multi-Task PER Reference** (p < $10^{-5}$ PER gain) |
| **Treatment C1** | HF: `langfens-rq1-trn-02b-checkpoints`<br>`C1/fixed_budget_checkpoint.pt`<br>SHA-256 `52b0e6c4...` | Kaggle T4 $\times$ 2 | 3,600 / Ep 6 | 0.302386 | 22.01% | +0.011423 | **Hypothesis NOT_SUPPORTED** (Falsified Ablation) |

### 2.2 Reaffirmed Scientific Findings
1. **Primary Causal Verdict:** Whole-phone prompt dropout ($p=0.25$) yielded $\Delta_{\text{dropout}} = \mathbf{-0.007918}$ AP points (95% cluster bootstrap CI: `[-0.018328, +0.000385]`). The hypothesis that whole-phone prompt dropout forces acoustic reliance is **NOT_SUPPORTED on Fold 1**.
2. **Acoustic Reliance Degradation:** Rather than expanding acoustic reliance, prompt dropout caused the acoustic gap to contract ($\Delta_{\text{gap}} = -0.006120$), with the lower bound crossing zero (`[-0.003024, +0.025217]`), and caused performance on speaker `RRBI` to collapse (-0.0098 AP).
3. **Reference R Retained:** Reference R is established as the sole primary production baseline for Fold 1. It serves as the architectural and methodological blueprint for Folds 2, 3, and 4.

---

## 3. Speaker Partitioning Audit & Cross-Validation Conflict Resolution

### 3.1 Exhaustive Speaker Partitioning Matrix

The authoritative split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`) partitions the 24 L2-ARCTIC speakers into four mathematically balanced folds:

| Fold Index | Partition | Speaker Count | Male | Female | L1 Distribution | Utterances | Canonical Tokens | Error Tokens (Prevalence) |
|:---:|---|:---:|:---:|:---:|---|:---:|:---:|:---:|
| **Fold 1** | **TRAIN** | 16 | 8 | 8 | 2 Ara, 3 Chi, 2 Hin, 3 Kor, 3 Spa, 3 Vie | 2,399 | 79,426 | 11,841 (14.91%) |
| | **VAL** | 2 | 1 | 1 | `NJS` (Spa, F), `RRBI` (Hin, M) | 300 | 9,956 | 1,157 (11.62%) |
| | **TEST (SEALED)** | 6 | 3 | 3 | `ABA` (Ara), `ASI` (Hin), `EBVS` (Spa), `HJK` (Kor), `LXC` (Chi), `PNV` (Vie) | 900 | 29,769 | 4,520 (15.18%) |
| **Fold 2** | **TRAIN** | 16 | 8 | 8 | 3 Ara, 3 Chi, 2 Hin, 2 Kor, 3 Spa, 3 Vie | 2,400 | 79,480 | 11,959 (15.05%) |
| | **VAL** | 2 | 1 | 1 | `TNI` (Hin, F), `YBAA` (Ara, M) | 299 | 9,873 | 921 (9.33%) |
| | **TEST (SEALED)** | 6 | 3 | 3 | `BWC` (Chi), `HKK` (Kor), `HQTV` (Vie), `MBMPS` (Spa), `SKA` (Ara), `SVBI` (Hin) | 900 | 29,798 | 4,638 (15.56%) |
| **Fold 3** | **TRAIN** | 16 | 8 | 8 | 2 Ara, 2 Chi, 3 Hin, 3 Kor, 3 Spa, 3 Vie | 2,400 | 79,472 | 12,175 (15.32%) |
| | **VAL** | 2 | 1 | 1 | `TXHC` (Chi, M), `ZHAA` (Ara, F) | 300 | 9,881 | 1,123 (11.37%) |
| | **TEST (SEALED)** | 6 | 3 | 3 | `ERMS` (Spa), `NCC` (Chi), `RRBI` (Hin), `THV` (Vie), `YBAA` (Ara), `YDCK` (Kor) | 899 | 29,798 | 4,220 (14.16%) |
| **Fold 4** | **TRAIN** | 16 | 8 | 8 | 3 Ara, 2 Chi, 3 Hin, 2 Kor, 3 Spa, 3 Vie | 2,399 | 79,499 | 12,366 (15.55%) |
| | **VAL** | 2 | 1 | 1 | `HKK` (Kor, M), `NCC` (Chi, F) | 300 | 9,866 | 1,012 (10.26%) |
| | **TEST (SEALED)** | 6 | 3 | 3 | `NJS` (Spa), `TLV` (Vie), `TNI` (Hin), `TXHC` (Chi), `YKWK` (Kor), `ZHAA` (Ara) | 900 | 29,786 | 4,140 (13.90%) |

### 3.2 Audit of the Speaker Overlap & Staging Conflict

#### The Perceived Conflict:
In early project documentation and `AGENTS.md`, the operational instruction was drafted as:
> *"Official test sealing: Never stage or extract official test audio/labels. Maintain strict speaker-disjoint isolation."*
> *"Fold 1 contract: 2,399 train, 300 val (NJS/RRBI), 900 test SEALED."*

When operating strictly within Fold 1, the test partition comprised six specific speakers:
$$S_{\text{test}}^{(1)} = \{\text{ABA}, \text{ASI}, \text{EBVS}, \text{HJK}, \text{LXC}, \text{PNV}\}$$
If an engineer or automated check treats $S_{\text{test}}^{(1)}$ as a **global, perpetual blacklist** across all project activities, an insurmountable deadlock arises:
- **Fold 2 Training Set:** Requires $\{\text{ABA}, \text{ASI}, \text{EBVS}, \text{HJK}, \text{LXC}, \text{PNV}\}$!
- **Fold 3 Training Set:** Requires $\{\text{ABA}, \text{ASI}, \text{EBVS}, \text{HJK}, \text{LXC}, \text{PNV}\}$!
- **Fold 4 Training Set:** Requires $\{\text{ABA}, \text{ASI}, \text{EBVS}, \text{HJK}, \text{LXC}, \text{PNV}\}$!

Under an erroneous global blacklist interpretation, Folds 2, 3, and 4 could never be trained.

#### The Scientific Resolution:
In standard $K$-fold cross-validation ($K=4$), test partitions are **fold-specific**, not globally permanent:
1. **Partition Exhaustiveness:** The four test sets are mutually disjoint and their union encompasses all 24 corpus speakers:
   $$\bigcup_{k=1}^4 S_{\text{test}}^{(k)} = \mathcal{S}_{\text{all}} \quad (24 \text{ speakers}), \qquad S_{\text{test}}^{(i)} \cap S_{\text{test}}^{(j)} = \emptyset \quad \forall i \ne j$$
2. **Strict Intra-Fold Disjointness:** For any given fold $k \in \{1, 2, 3, 4\}$, the partitions are provably leak-free:
   $$S_{\text{train}}^{(k)} \cap S_{\text{val}}^{(k)} = \emptyset, \quad S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset, \quad S_{\text{val}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$$
3. **Out-of-Fold Independence:** Model $M_k$ is evaluated strictly on $S_{\text{val}}^{(k)}$ (and eventually $S_{\text{test}}^{(k)}$). Model $M_k$ is NEVER evaluated on utterances from $S_{\text{train}}^{(k)}$. When $M_2$ trains on $\{\text{ABA}, \dots\}$, it will be tested exclusively on $\{\text{BWC}, \text{HKK}, \text{HQTV}, \text{MBMPS}, \text{SKA}, \text{SVBI}\}$, which it has never seen.
4. **Clarification of AGENTS.md Rule:** The sealing mandate applies to **the active fold's test set during training and validation**. It prohibits staging or evaluating the test split of fold $k$ while building and tuning model $M_k$.

---

## 4. Authoritative Three-Tier Test-Isolation Policy

To eliminate any possibility of data leakage while enabling rigorous cross-validation, Project Langfens establishes a strict **Three-Tier Isolation Policy**:

```text
Evaluation Tiering Architecture
├── Tier 1: Cross-Fold Validation Phase (Current Plan — Phase RQ1-TRN-03A)
│   ├── Target: Train Baseline R on Folds 2, 3, 4; Validate on S_val^(k)
│   └── Isolation: ALL 4 Test Partitions (S_test^(1..4)) are 100% SEALED. Zero test staging.
├── Tier 2: Model Improvement Phase (Future — Phase RQ1-TRN-03B)
│   ├── Target: Develop novel acoustic conditioning / feedback heads across folds
│   └── Isolation: Tuned strictly on pooled 4-fold validation sets. ALL Test sets remain SEALED.
└── Tier 3: Final Independent Cross-Fold Testing (Thesis Benchmark — Phase RQ1-EVAL-01)
    ├── Target: Evaluate frozen best models M_1..M_4 on held-out S_test^(1..4)
    └── Isolation: Executed strictly once with frozen decision thresholds. Single unsealing pass.
```

### 4.1 Physical Fail-Closed Staging Implementation
To ensure test sealing cannot be bypassed by software bugs or runtime leakage:
1. **Dedicated Per-Fold Packages:** Data staging packages are generated strictly per-fold and contain **only** train and validation data:
   - Fold 1: `authorized_RQ1-TRN_fold1_train_val_only.zip` (18 speakers: 16 train + 2 val) — *Existing*
   - Fold 2: `authorized_RQ1-TRN_fold2_train_val_only.zip` (18 speakers: 16 train + 2 val) — *To be built*
   - Fold 3: `authorized_RQ1-TRN_fold3_train_val_only.zip` (18 speakers: 16 train + 2 val) — *To be built*
   - Fold 4: `authorized_RQ1-TRN_fold4_train_val_only.zip` (18 speakers: 16 train + 2 val) — *To be built*
2. **Zero Test Files on Remote Disks:** In Tier 1, the test audio and TextGrids for Fold $k$ ($S_{\text{test}}^{(k)}$) are **physically absent** from the staging zip archive. Even if an automated script attempted to load test data, it would fail closed with `FileNotFoundError`.
3. **No Access or Staging in this Phase:** During Phase RQ1-TRN-03A planning, **zero test files are unsealed, extracted, or uploaded**.

---

## 5. Independent Baseline R Training Protocol (Folds 2, 3, 4)

### 5.1 Prohibition of Checkpoint Transfer
- **Rule:** Under no circumstances may Fold 2, Fold 3, or Fold 4 be initialized or fine-tuned from Fold 1 checkpoints (`best.pt`, `last.pt`, or C0/C1).
- **Scientific Justification:** Fold 1 `best.pt` was already optimized on Fold 1 training speakers and selected on Fold 1 validation speakers (`NJS`, `RRBI`). Initializing Fold 2 from Fold 1 `best.pt` would contaminate Fold 2 with prior exposure, creating cross-fold parameter entanglement and invalidating cross-validation independence.

### 5.2 Independent Initialization Contract
Each fold model $M_k$ ($k \in \{2, 3, 4\}$) must be initialized completely from scratch:
1. **Backbone:** Original pretrained `facebook/wav2vec2-base-960h` (revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94.4M parameters), loaded directly from Hugging Face cache or local verified weights.
2. **Trainable Heads:** Fresh pseudo-random parameter initialization for:
   - CTC linear projection head ($768 \to 41$ phoneme classes).
   - Cross-attention key/query/value projections and canonical phoneme embedding matrix ($41 \to 768$).
   - Binary MDD classification MLP ($1536 \to 256 \to 1$).
3. **Frozen Heads:** 39-class substitution diagnosis head remains completely frozen, unparameterized in the optimizer, and untrained.
4. **Frozen Backbone:** Pretrained Wav2Vec2 encoder remains frozen throughout training (`requires_grad = False`, eval mode during `.train()` calls).

### 5.3 Deterministic Seeding Policy
- **Master Seed:** `42` (identical to split generation and baseline standard).
- **RNG Synchronization:** At the start of each fold training script:
  ```python
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  random.seed(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- Any fold-indexed RNG perturbation must be explicitly recorded in the training config ledger.

### 5.4 Architecture & Hyperparameter Contract Preservation
To guarantee strict comparability with Fold 1 Baseline R:

| Parameter | Specification | Preserved Justification |
|---|---|---|
| **Architecture** | `LangfensWav2Vec2MDD` (Two-Head Mode) | Bit-identical model definition in `src/langfens_prep/model.py` |
| **Backbone Mode** | Frozen `facebook/wav2vec2-base-960h` | Prevents representation collapse and saves GPU VRAM |
| **Binary Prompt Dropout** | $p = 0.00$ (Disabled) | $p=0.25$ falsified in RQ1-TRN-02B; Baseline R standard is $p=0.00$ |
| **Loss Function** | $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{ctc}} + 1.0 \cdot \mathcal{L}_{\text{binary}}$ | Multi-task objective balancing acoustic alignment and error detection |
| **Binary Loss Implementation** | `MaskedBCEWithLogitsLoss(reduction='mean')` | Masks padding tokens and ignore indices (-100) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$) | Standard Transformer optimization setup |
| **Peak Learning Rate** | $1.0 \times 10^{-3}$ for trainable heads | Proved optimal in RQ1-TRN-01B1 (Variant D, zero blank collapse) |
| **LR Schedule** | Cosine Annealing with Warmup (10% steps) | Smooth convergence trajectory |
| **Batch Size & Accumulation** | Batch size = 4 utterances, Accumulation = 1 | Matches Fold 1 exact micro-step and update dynamics |
| **Precision** | CUDA FP16 AMP (`torch.cuda.amp.GradScaler`) | Memory-efficient execution on remote T4 hardware |
| **Training Budget** | Primary: 6 epochs (3,600 updates); Max: 10 epochs | Primary fixed-budget evaluation at 3,600 updates; early stopping patience 5 |
| **CTC Length Guard** | Fail-closed skip on input length < target length | Prohibits unaligned CTC loss crashes |

---

## 6. Remote Infrastructure, Quota Management & Remote Persistence

### 6.1 Compute Platform Evaluation & Allocation
- **Primary Execution Environment:** Kaggle GPU Notebook / Script execution.
  - Accelerator: Nvidia Tesla T4 $\times$ 2 (`machine_shape: NvidiaTeslaT4`).
  - Available Quota: 30.00 hours/week (verified via `kaggle quota`).
  - Expected Runtime: ~48–55 minutes per fold run (6 epochs / 3,600 updates).
  - Total Compute Required: $3 \text{ folds} \times 52 \text{ min} \approx \mathbf{2.6 \text{ hours}}$ (< 9% of weekly quota).
- **Secondary Fallback Environment:** Google Colab GPU (Tesla T4 via official Colab CLI). Reserved as backup if Kaggle quota experiences unexpected degradation.

### 6.2 Data Staging Strategy
1. **Source of Truth:** Immutable raw dataset `l2arctic_release_v5.0.zip` on Google Drive (`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`, SHA-256 `490d5a43...`).
2. **Packaging Script:** `audit_scripts/build_fold_packages.py` will generate:
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip`
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip`
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`
3. **Cryptographic Validation:** Each zip package is hashed and logged in a staging manifest before deployment.
4. **Remote Mirroring:** Upload packages to private Kaggle dataset (`minkhoaa/langfens-rq1-fold2-staging`, etc.) or persistent Google Drive folder.
5. **Local VM Containment:** Remote runners copy the package from `/kaggle/input` to `/kaggle/working` or `/tmp`, verify the SHA-256 digest, extract locally, and stream audio directly from high-speed local disk.

### 6.3 Remote Checkpoint Durability & Atomic Readback
To safeguard against Kaggle ephemeral session termination (where unmirrored artifacts are permanently deleted):
1. **Per-Epoch Checkpointing:** After every epoch, the trainer writes `checkpoint_epoch_{e}.pt` and updates `best.pt` (if $\text{AP}_{\text{val}}$ improves) and `last.pt`.
2. **Payload Completeness:** Every checkpoint dictionary must include:
   ```python
   {
       "epoch": epoch,
       "global_step": global_step,
       "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "scheduler_state_dict": scheduler.state_dict(),
       "scaler_state_dict": scaler.state_dict(),
       "rng_state": torch.get_rng_state(),
       "cuda_rng_state": torch.cuda.get_rng_state_all(),
       "val_ap": val_ap,
       "best_val_ap": best_val_ap,
       "config": training_config,
   }
   ```
3. **Atomic Cloud Sync:** Checkpoints are mirrored via private Hugging Face API (`huggingface_hub.HfApi`) to a dedicated private repository:
   `minkhoaa/langfens-rq1-cross-fold-checkpoints` under subfolders `fold2/`, `fold3/`, `fold4/`.
4. **Readback Probe:** Immediate cryptographic digest verification or file existence check ensures no partial writes occur.

---

## 7. Cross-Fold Evaluation Metrics, Confidence Intervals & Aggregation Rules

### 7.1 Multi-Granular Evaluation Regimes per Fold
For each fold model $M_k$ evaluated on its validation split $S_{\text{val}}^{(k)}$:

1. **Protocol O (Oracle TextGrid Tokens):**
   - **Matched Binary Average Precision ($\text{AP}_{\text{matched}}^{(k)}$):** Primary ranking metric computed from sorted posterior probabilities against ground-truth error labels ($b_i \in \{0, 1\}$).
   - **Mismatched Audio AP ($\text{AP}_{\text{mismatched}}^{(k)}$):** Evaluated with randomly shuffled audio representations to measure reliance on spurious prior correlations.
   - **Acoustic Reliance Gap ($\text{Gap}^{(k)}$):**
     $$\text{Gap}^{(k)} = \text{AP}_{\text{matched}}^{(k)} - \text{AP}_{\text{mismatched}}^{(k)}$$
     A positive gap confirms that the model utilizes acoustic evidence to detect errors.
   - **Canonical Prior Margin:** Difference between model AP and the empirical error prior computed on the fold's training set ($\text{AP}_{\text{prior}}^{(k)}$).
   - **Greedy CTC Phoneme Error Rate ($\text{PER}^{(k)}$):**
     $$\text{PER}^{(k)} = \frac{S_k + D_k + I_k}{N_{\text{ref}}^{(k)}}$$
   - **Operational Threshold Metrics ($\tau \in \{0.20, 0.50\}$ and calibrated $\tau_k^*$):**
     Precision, Recall (Sensitivity), Specificity, $F_1$-Score, and False Acceptance Rate ($\text{FAR} = 1 - \text{Recall}$).

2. **Protocol Z (Zero-Oracle G2P Tokens):**
   - Realistic evaluation using CMUdict G2P reference phonemes without gold TextGrid timestamps.
   - Dynamic programming alignment matching hypothesis tokens to canonical reference.
   - Aligned-pair Average Precision ($\text{AP}_Z^{(k)}$) and Token Coverage percentage.

3. **Disaggregated Speaker-Level Breakdown:**
   - In each fold, both validation speakers ($s_{k,1}, s_{k,2}$) are evaluated individually.
   - Across four folds, this produces detailed performance profiles for **eight distinct speakers** spanning all six L1 accents.

### 7.2 Cross-Fold Aggregation Rules

To synthesize performance across the four folds into an authoritative corpus-wide benchmark:

#### 1. Macro-Averaging (Unweighted Fold Mean)
Treats each fold as an independent observation:
$$\overline{\text{AP}}_{\text{macro}} = \frac{1}{4} \sum_{k=1}^4 \text{AP}^{(k)}$$
$$\sigma_{\text{macro}} = \sqrt{\frac{1}{3} \sum_{k=1}^4 \left( \text{AP}^{(k)} - \overline{\text{AP}}_{\text{macro}} \right)^2}$$
Standard Error:
$$\text{SE}_{\text{macro}} = \frac{\sigma_{\text{macro}}}{\sqrt{4}} = \frac{\sigma_{\text{macro}}}{2}$$
Student's $t$ 95% Confidence Interval ($df = 3$, two-tailed critical value $t_{0.025, 3} = 3.182$):
$$\text{CI}_{95\%, \text{macro}} = \left[ \overline{\text{AP}}_{\text{macro}} - 3.182 \cdot \text{SE}_{\text{macro}}, \; \overline{\text{AP}}_{\text{macro}} + 3.182 \cdot \text{SE}_{\text{macro}} \right]$$

#### 2. Micro-Averaging (Pooled Token-Level Evaluation)
Pools all out-of-fold validation predictions across all four folds into a single global dataset:
$$N_{\text{val, pooled}} = \sum_{k=1}^4 N_{\text{val}}^{(k)} = 9,956 + 9,873 + 9,881 + 9,866 = \mathbf{39,576 \text{ tokens}}$$
$$E_{\text{val, pooled}} = \sum_{k=1}^4 E_{\text{val}}^{(k)} = 1,157 + 921 + 1,123 + 1,012 = \mathbf{4,213 \text{ error tokens}} \quad (10.65\% \text{ prevalence})$$
- Computes global Precision-Recall curve, global $\text{AP}_{\text{micro}}$, global ROC-AUC, and global threshold confusion matrices.
- Eliminates fold-size weighting artifacts.

#### 3. Speaker-Stratified Cluster Bootstrap (1,000 Resamples)
- Resamples entire speakers with replacement from the eight validation speakers (`NJS`, `RRBI`, `TNI`, `YBAA`, `TXHC`, `ZHAA`, `HKK`, `NCC`).
- Computes non-parametric 95% confidence intervals (`[2.5%, 97.5%]`) for macro AP, micro AP, acoustic gap, and CTC PER.
- Directly quantifies uncertainty stemming from speaker diversity.

#### 4. L1 Accent Disaggregation
Aggregates metrics by speaker native language:
- Arabic (`YBAA`, `ZHAA`)
- Chinese (`TXHC`, `NCC`)
- Hindi (`RRBI`, `TNI`)
- Korean (`HKK`)
- Spanish (`NJS`)
Reveals whether the architecture suffers from accent-specific bias or performance discrepancies.

---

## 8. Separation of Cross-Fold Validation, Model Improvement & Final Testing

To maintain scientific integrity and prevent leakage throughout the remainder of the thesis research:

```text
Detailed Phase Separation Roadmap
│
├── PHASE RQ1-TRN-03A: Baseline R Cross-Fold Training & Validation (Current Scope)
│   ├── Milestone 1: Stage Train/Val packages for Folds 2, 3, 4 (Zero test audio staged)
│   ├── Milestone 2: Independent Baseline R training on Kaggle/Colab GPU (3 runs, 3,600 updates each)
│   ├── Milestone 3: Checkpoint persistence to Hugging Face with cryptographic verification
│   ├── Milestone 4: Standalone CPU validation replay (Protocol O & Protocol Z)
│   └── Deliverable: 4-Fold Validation Baseline Report (Macro/Micro AP, Acoustic Gap, CTC PER)
│
├── PHASE RQ1-TRN-03B: Model Improvement & Pronunciation Feedback Exploration (Subsequent Scope)
│   ├── Candidate A: Cross-Attention Acoustic Regularization (feature-level dropout or contrastive gate)
│   ├── Candidate B: Auxiliary Acoustic Contrastive Loss (Task B/C alignment)
│   ├── Candidate C: Substitution Diagnosis Head Rehabilitation (acoustic-conditioned realized phone classification)
│   ├── Rule: Tuned and compared STRICTLY against Phase RQ1-TRN-03A on the pooled validation set
│   └── Constraint: Test partitions remain 100% SEALED
│
├── PHASE RQ1-EVAL-01: Final Independent 4-Fold Cross-Validation Test (Final Thesis Gate)
│   ├── Trigger: Only after final model architecture is frozen and approved by operator
│   ├── Execution: Single unsealing of S_test^(1..4) (24 speakers, 3,599 utterances, 119,151 tokens)
│   ├── Protocol: Evaluate frozen models M_1..M_4 on held-out test sets with frozen thresholds tau_k*
│   └── Deliverable: Definitive L2-ARCTIC 4-Fold Cross-Validation Benchmark for Thesis Defense
│
└── PHASE RQ2: Cross-Corpus Transfer & Educational Pronunciation Feedback
    ├── SpeechOcean762 zero-shot transfer evaluation (5,000 utterances)
    └── Pedagogical feedback generation (articulatory guidance, phoneme substitution trees)
```

---

## 9. Research Risks, Execution Costs & Acceptance Gates

### 9.1 Research Risks & Mitigation Matrix

| Identified Risk | Severity | Scientific Impact | Mitigation Protocol |
|---|:---:|---|---|
| **Inter-Fold Variance** | Medium | Validation AP may fluctuate significantly between folds due to varying L1 error rates (e.g. Vietnamese 23.5% vs Korean 9.4%). | Report both Macro and Micro AP alongside L1-stratified breakdowns and cluster bootstrap CIs. |
| **CTC Instability on New Folds** | High | Risk of blank collapse or gradient divergence on unobserved speaker distributions. | Architecture uses frozen backbone + peak lr $10^{-3}$, proven 100% stable in RQ1-TRN-01B1 and Fold 1 runs. Driver enforces fail-closed length checks. |
| **Accidental Test Set Leakage** | Critical | Invalidates academic credibility and thesis defense. | Strict physical isolation: test utterances are omitted at archive build time (`train_val_only.zip`). Remote VMs cannot leak data they do not possess. |
| **Ephemeral GPU Data Loss** | High | Kaggle session timeout causing loss of 50-minute training run. | Mandatory atomic per-epoch mirroring to Hugging Face with readback SHA-256 verification before proceeding. |
| **Speaker Memorization** | Medium | Model relies on speaker acoustic traits rather than phoneme mispronunciation. | Maintained acoustic permutation control (mismatched audio) to track the acoustic reliance gap across all folds. |

### 9.2 Execution Resource Budget

| Activity | Target Platform | Instances | Estimated Duration | Total GPU Budget | Quota Impact |
|---|---|:---:|:---:|:---:|:---:|
| **Fold Package Build** | Local Laptop CPU | 3 archives | ~3 minutes | 0.0 GPU hours | Local disk only (~3.6 GB) |
| **Remote Staging Sync** | Local $\to$ Kaggle/HF | 3 datasets | ~5 minutes | 0.0 GPU hours | Cloud storage only |
| **Fold 2 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50 minutes | ~0.83 GPU hours | ~2.8% weekly quota |
| **Fold 3 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50 minutes | ~0.83 GPU hours | ~2.8% weekly quota |
| **Fold 4 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50 minutes | ~0.83 GPU hours | ~2.8% weekly quota |
| **Cross-Fold CPU Validation** | Local Laptop CPU | 3 evaluations | ~12 minutes | 0.0 GPU hours | Local CPU inference mode |
| **Total Phase Budget** | — | **3 GPU Runs** | **~2.5 Hours** | **~2.5 GPU Hours** | **< 9% Weekly Quota** |

### 9.3 Formal Acceptance Gates

- **Gate 1: Preflight & Staging Integrity Gate (Local Laptop)**
  - Split manifest checksum confirmed: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`).
  - Staging script `audit_scripts/build_fold_packages.py` creates `fold2`, `fold3`, `fold4` packages.
  - Automated audit proves $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$ and 0 test files inside each zip.
  - Test suite passes with 0 failures (`pytest tests/`).
- **Gate 2: Independent Training & Checkpoint Persistence Gate (Remote GPU)**
  - Remote training runs for Folds 2, 3, and 4 complete 6 epochs / 3,600 updates each with 0 unhandled exceptions and 0 AMP skips.
  - Checkpoints (`best.pt`, `last.pt`, `checkpoint_epoch_*.pt`) mirrored to Hugging Face with matching SHA-256 readback.
- **Gate 3: Cross-Fold Validation & Statistical Aggregation Gate (Local Laptop)**
  - Checkpoints downloaded to local cache.
  - Standalone CPU evaluator executes Protocol O and Protocol Z on validation partitions.
  - Synthesized report computes Macro AP, Micro AP, Acoustic Gap, CTC PER, and 1,000-resample cluster bootstrap 95% CIs.
  - All test partitions remain 100% sealed.

---

## 10. Operator Action Packet & Required Decisions

In compliance with `AGENTS.md` ("Never start training without explicit user authorization. STOP before execution"):

### Exact Decisions Requiring Operator Approval:

1. **Strategic Scope Approval:**
   - Confirm reprioritization of RQ1 toward corpus-wide 4-Fold Cross-Validation Baseline Training (Phase RQ1-TRN-03A), retaining Reference R as the architectural blueprint.
2. **Test-Isolation Policy Sign-Off:**
   - Formally approve the Three-Tier Test-Isolation Policy: keeping all four test partitions ($S_{\text{test}}^{(1..4)}$) 100% sealed and unstaged during Folds 2, 3, and 4 training and validation.
3. **Staging Package Generation Approval:**
   - Authorize execution of `audit_scripts/build_fold_packages.py` to create `authorized_RQ1-TRN_fold2_train_val_only.zip`, `authorized_RQ1-TRN_fold3_train_val_only.zip`, and `authorized_RQ1-TRN_fold4_train_val_only.zip` containing strictly train and val speakers.
4. **Compute Platform Selection:**
   - Approve Kaggle GPU (Tesla T4 $\times$ 2, private kernels) as the primary execution platform (~2.5 hours total GPU budget across 3 runs).
5. **Remote Persistence Destination:**
   - Approve target private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` (or private Google Drive) for atomic per-epoch mirroring.

**STATUS: EXECUTION HALTED. AWAITING OPERATOR REVIEW AND APPROVAL OF ITEMS 1–5.**
