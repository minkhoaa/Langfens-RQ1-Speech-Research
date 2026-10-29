# LANGFENS — RQ1-TRN-03A: Cross-Fold Speaker-Independent Generalization Research Plan (Corrected Protocol)

**Document ID:** `LANGFENS-RQ1-TRN-03A-CROSS-FOLD-RESEARCH-PLAN`<br>
**Phase Identifier:** `RQ1-TRN-03A`<br>
**Target Research Question:** RQ1 — Speaker-Independent Mispronunciation Detection, Acoustic Generalization & Pronunciation Feedback<br>
**Author:** Senior Speech ML Research Engineer & Reproducibility Auditor<br>
**Date:** 2026-10-29<br>
**Base Commit:** `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c` on branch `review/rq1-trn-03a-cross-fold-planning`<br>
**Status:** `READY_FOR_OPERATOR_REVIEW` (Protocol Corrected; Execution Blocked Pending Operator Authorization)

---

## 1. Executive Summary & Strategic Reprioritization

With the successful completion and closure of Phase RQ1-TRN-02B (where whole-phone prompt embedding dropout was conclusively determined to be `NOT_SUPPORTED` on Fold 1), Project Langfens transitions from single-fold prompt-dropout ablations to its primary scientific objective: **rigorous, corpus-wide speaker-independent mispronunciation detection (MDD)** across all 24 L2-ARCTIC speakers.

### 1.1 Strategic Rationale
1. **Generalization Over Hyper-Tuning:** Single-fold evaluations (Fold 1) cannot establish how well the frozen Wav2Vec2 + multi-task MDD architecture generalizes across distinct human vocal tracts, fundamental frequency ($F_0$) distributions, and diverse first-language (L1) phonological transfer profiles.
2. **Pedagogical Alignment:** Computer-Assisted Pronunciation Training (CAPT) systems fail in practice if their decision boundaries overfit to a specific development cohort. Establishing robust, out-of-fold generalization across all 24 L2-ARCTIC speakers is a prerequisite for generating trustworthy pronunciation feedback.
3. **Thesis Proposal Fulfillment:** The master engineering thesis protocol specified 4-Fold Speaker-Independent Cross-Validation as the authoritative benchmark. Completing Folds 2, 3, and 4 establishes the empirical baseline of RQ1.
4. **Thesis Scope Demarcation:** 
   - **Research Question 1 (RQ1):** Encompasses speech acoustic foundation models, multi-task mispronunciation detection, substitution diagnosis, cross-fold speaker generalization, cross-corpus zero-shot transfer on SpeechOcean762, and phonological pronunciation feedback.
   - **Research Question 2 (RQ2):** Reserved exclusively for Multi-Agent Automated Test Generation.

### 1.2 Non-Negotiable Operational Boundaries
In strict adherence to `AGENTS.md`:
- **Local Laptop Boundary:** Limited strictly to configuration design, static manifest audits, staging script preparation, unit/regression testing, and post-training CPU metric aggregation. **Zero local training or backward passes.**
- **Remote GPU Boundary:** ALL training runs (Folds 2, 3, 4) must execute on authorized remote GPU hardware (Kaggle Tesla T4 $\times$ 2 or Google Colab T4) under CUDA FP16 AMP.
- **Fail-Closed Per-Fold Test Sealing:** For each model $M_k$, the test partition $S_{\text{test}}^{(k)}$ is strictly excluded from training, validation, early stopping, threshold calibration, and remote VM staging.
- **Execution Stop:** **NO training, NO dataset unsealing, NO staging package creation, and NO GPU provisioning occurs during this planning step.** The process stops immediately for operator review and explicit sign-off.

---

## 2. Preservation of Completed Fold 1 Scientific Evidence

All historical findings, code paths, and checkpoints from RQ1-TRN-02A and RQ1-TRN-02B are preserved as immutable scientific records. Under no circumstances will prior evidence be overwritten, modified, or re-run.

### 2.1 Preserved Fold 1 Artifact Ledger

| Model Arm | Canonical Identifier | Source Platform | Updates / Epochs | Matched Val AP | CTC PER | Acoustic Gap | Authoritative Status |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Reference R** | `checkpoints/RQ1-TRN-02A/recovered/best.pt`<br>SHA-256 `a2445c8d...` | Colab T4 | 3,600 / Ep 6 (of 11) | **0.317021** | 23.68% | **+0.022299** | **Authoritative Primary Production Baseline** |
| **Control C0** | HF: `langfens-rq1-trn-02b-checkpoints`<br>`C0/fixed_budget_checkpoint.pt`<br>SHA-256 `0dc1d4fb...` | Kaggle T4 $\times$ 2 | 3,600 / Ep 6 | 0.310304 | **22.02%** | +0.017543 | **Multi-Task PER Reference** (PER delta: -1.66 pp vs R) |
| **Treatment C1** | HF: `langfens-rq1-trn-02b-checkpoints`<br>`C1/fixed_budget_checkpoint.pt`<br>SHA-256 `52b0e6c4...` | Kaggle T4 $\times$ 2 | 3,600 / Ep 6 | 0.302386 | 22.01% | +0.011423 | **Hypothesis NOT_SUPPORTED** (Falsified Ablation) |

### 2.2 Reaffirmed Scientific Findings
1. **Primary Causal Verdict:** Whole-phone prompt dropout ($p=0.25$) yielded $\Delta_{\text{dropout}} = \text{C1} - \text{C0} = \mathbf{-0.007918}$ AP points (95% cluster bootstrap CI: `[-0.018328, +0.000385]`). The hypothesis that whole-phone prompt dropout forces acoustic reliance is **NOT_SUPPORTED on Fold 1**.
2. **Acoustic Reliance Degradation:** Rather than expanding acoustic reliance, prompt dropout caused the acoustic gap to contract ($\Delta_{\text{gap}} = -0.006120$), with the lower bound crossing zero (`[-0.003024, +0.025217]`), and caused performance on speaker `RRBI` to collapse (-0.0098 AP).
3. **Reference R Retained:** Reference R is established as the sole primary production baseline for Fold 1. It serves as the architectural and methodological blueprint for Folds 2, 3, and 4.
4. **Removal of Unsupported Significance Claims:** The CTC PER difference between C0 (22.02%) and R (23.68%) represents an empirical difference of -1.66 percentage points (2,176 edits vs 2,340 edits over 9,881 reference tokens). Prior draft text casually asserting "$p < 10^{-5}$" is retracted as methodologically unsubstantiated, as no formal non-parametric utterance-level bootstrap hypothesis test was conducted on Levenshtein edit distance.

---

## 3. Speaker Partitioning Audit & Cross-Validation Conflict Resolution

### 3.1 Exhaustive Speaker Partitioning Matrix

The authoritative split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`) partitions the 24 L2-ARCTIC speakers into four mathematically balanced folds:

| Fold Index | Partition | Speaker Count | Male | Female | L1 Distribution | Utterances | Canonical Tokens | Error Tokens (Prevalence) |
|:---:|---|:---:|:---:|:---:|---|:---:|:---:|:---:|
| **Fold 1** | **TRAIN** | 16 | 8 | 8 | 2 Ara, 3 Chi, 2 Hin, 3 Kor, 3 Spa, 3 Vie | 2,399 | 79,426 | 11,841 (14.91%) |
| | **VAL** | 2 | 1 | 1 | `NJS` (Spa, F), `RRBI` (Hin, M) | 300 | 9,956 | 1,157 (11.62%) |
| | **TEST** | 6 | 3 | 3 | `ABA` (Ara), `ASI` (Hin), `EBVS` (Spa), `HJK` (Kor), `LXC` (Chi), `PNV` (Vie) | 900 | 29,769 | 4,520 (15.18%) |
| **Fold 2** | **TRAIN** | 16 | 8 | 8 | 3 Ara, 3 Chi, 2 Hin, 2 Kor, 3 Spa, 3 Vie | 2,400 | 79,480 | 11,959 (15.05%) |
| | **VAL** | 2 | 1 | 1 | `TNI` (Hin, F), `YBAA` (Ara, M) | 299 | 9,873 | 921 (9.33%) |
| | **TEST** | 6 | 3 | 3 | `BWC` (Chi), `HKK` (Kor), `HQTV` (Vie), `MBMPS` (Spa), `SKA` (Ara), `SVBI` (Hin) | 900 | 29,798 | 4,638 (15.56%) |
| **Fold 3** | **TRAIN** | 16 | 8 | 8 | 2 Ara, 2 Chi, 3 Hin, 3 Kor, 3 Spa, 3 Vie | 2,400 | 79,472 | 12,175 (15.32%) |
| | **VAL** | 2 | 1 | 1 | `TXHC` (Chi, M), `ZHAA` (Ara, F) | 300 | 9,881 | 1,123 (11.37%) |
| | **TEST** | 6 | 3 | 3 | `ERMS` (Spa), `NCC` (Chi), `RRBI` (Hin), `THV` (Vie), `YBAA` (Ara), `YDCK` (Kor) | 899 | 29,798 | 4,220 (14.16%) |
| **Fold 4** | **TRAIN** | 16 | 8 | 8 | 3 Ara, 2 Chi, 3 Hin, 2 Kor, 3 Spa, 3 Vie | 2,399 | 79,499 | 12,366 (15.55%) |
| | **VAL** | 2 | 1 | 1 | `HKK` (Kor, M), `NCC` (Chi, F) | 300 | 9,866 | 1,012 (10.26%) |
| | **TEST** | 6 | 3 | 3 | `NJS` (Spa), `TLV` (Vie), `TNI` (Hin), `TXHC` (Chi), `YKWK` (Kor), `ZHAA` (Ara) | 900 | 29,786 | 4,140 (13.90%) |

### 3.2 Audit of the Speaker Overlap & Rejection of False Global Sealing

#### The Methodological Error of Global Sealing:
In early project documentation, a colloquial phrasing ("Official test sealing: Never stage or extract official test audio/labels") led to a severe misinterpretation: that the six speakers of Fold 1 Test ($S_{\text{test}}^{(1)} = \{\text{ABA}, \text{ASI}, \text{EBVS}, \text{HJK}, \text{LXC}, \text{PNV}\}$) were permanently blacklisted from ever being staged or trained upon across any fold.

This interpretation is fundamentally incorrect. In standard $K$-fold cross-validation ($K=4$) across a finite corpus of 24 speakers:
1. **Speakers Rotate Legitudinally Across Folds:**
   - The speakers in $S_{\text{test}}^{(1)}$ are held out for model $M_1$, but they **must** be used to train models $M_2$, $M_3$, and $M_4$.
   - The speakers in $S_{\text{test}}^{(2)}$ are held out for model $M_2$, but they **must** be used to train models $M_1$, $M_3$, and $M_4$.
   - The speakers in $S_{\text{test}}^{(3)}$ are held out for model $M_3$, but they **must** be used to train models $M_1$, $M_2$, and $M_4$.
   - The speakers in $S_{\text{test}}^{(4)}$ are held out for model $M_4$, but they **must** be used to train models $M_1$, $M_2$, and $M_3$.
2. **Mathematical Exhaustiveness:** Across the four folds, every single speaker in L2-ARCTIC appears in exactly one test set ($4 \times 6 = 24$), and the four test sets are mutually disjoint:
   $$\bigcup_{k=1}^4 S_{\text{test}}^{(k)} = \mathcal{S}_{\text{all}} \quad (24 \text{ speakers}), \qquad S_{\text{test}}^{(i)} \cap S_{\text{test}}^{(j)} = \emptyset \quad \forall i \ne j$$
3. **Refutation of Global Untouched Claims:** Any claim that all 24 speakers could be simultaneously omitted from training staging packages is mathematically impossible. A staging package for Fold $k$ must contain 18 speakers ($S_{\text{train}}^{(k)} \cup S_{\text{val}}^{(k)}$).

---

## 4. Authoritative Per-Fold Test Isolation & Three-Tier Evaluation Policy

To eliminate data leakage while enabling mathematically sound cross-validation:

```text
Evaluation Tiering Architecture
├── Tier 1: Cross-Fold Baseline Training & Validation (Phase RQ1-TRN-03A — Current Scope)
│   ├── Target: Train independent Baseline R models M_2, M_3, M_4; Validate on S_val^(k)
│   └── Isolation: For model M_k, S_test^(k) is strictly excluded from training, validation,
│                  model selection, threshold calibration, and remote staging.
│                  Test sets remain 100% un-evaluated during this phase.
├── Tier 2: Cross-Fold Model Improvement (Phase RQ1-TRN-03B — Subsequent Scope)
│   ├── Target: Explore acoustic conditioning, contrastive loss, and phoneme diagnosis
│   └── Isolation: Tuned strictly on the pooled 4-fold validation splits (39,576 tokens).
│                  Test sets remain 100% un-evaluated to prevent selection leakage.
└── Tier 3: Final Independent Cross-Fold Testing (Phase RQ1-EVAL-01 — Thesis Benchmark)
    ├── Target: Evaluate frozen final models M_1..M_4 on their respective held-out test sets S_test^(1..4)
    └── Isolation: Executed strictly once post-freeze using frozen thresholds tau_k* from validation.
```

### 4.1 Strict Per-Fold Isolation Contract
For any given fold $k \in \{1, 2, 3, 4\}$ and its corresponding model $M_k$:
1. **Training Isolation:** $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$. Model $M_k$ never computes forward or backward passes on $S_{\text{test}}^{(k)}$.
2. **Validation Isolation:** $S_{\text{val}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$. Checkpoint monitoring, early stopping, and hyperparameter checks evaluate strictly on $S_{\text{val}}^{(k)}$.
3. **Threshold Calibration Isolation:** Operating decision thresholds $\tau_k^*$ are calibrated strictly on $S_{\text{val}}^{(k)}$:
   $$\tau_k^* = \arg\max_\tau F_1^{(k)}(\tau; S_{\text{val}}^{(k)})$$
4. **Physical Staging Isolation:** The staging package for Fold $k$ (`authorized_RQ1-TRN_fold{k}_train_val_only.zip`) contains **strictly** the 18 speakers of $S_{\text{train}}^{(k)} \cup S_{\text{val}}^{(k)}$. Utterances from $S_{\text{test}}^{(k)}$ are physically omitted at zip creation time, ensuring remote VMs cannot experience accidental test leakage.
5. **Separation of Tuning and Testing:** Cross-fold validation results reported in Phase RQ1-TRN-03A are based strictly on out-of-fold validation sets ($S_{\text{val}}^{(k)}$). Test sets ($S_{\text{test}}^{(k)}$) remain completely un-evaluated until Phase RQ1-EVAL-01.

---

## 5. Independent Baseline R Training Protocol (Folds 2, 3, 4)

### 5.1 Prohibition of Checkpoint Transfer
- **Rule:** Under no circumstances may Fold 2, Fold 3, or Fold 4 be initialized or warm-started from Fold 1 checkpoints (`best.pt`, `last.pt`, C0, or C1).
- **Scientific Justification:** Fold 1 weights were already optimized on Fold 1 training speakers. Reusing Fold 1 weights for Fold 2 would contaminate Fold 2 with Fold 1 training exposure, violating statistical cross-validation independence.

### 5.2 Independent Initialization Contract
Each fold model $M_k$ ($k \in \{2, 3, 4\}$) must be initialized completely from scratch:
1. **Backbone:** Original pretrained `facebook/wav2vec2-base-960h` (revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94.4M parameters).
2. **Trainable Heads:** Fresh pseudo-random parameter initialization under `seed: 1337` for:
   - CTC linear projection head ($768 \to 41$ phoneme classes).
   - Cross-attention key/query/value projections and canonical phoneme embedding matrix ($41 \to 768$).
   - Binary MDD classification MLP ($1536 \to 256 \to 1$).
3. **Frozen Heads:** 39-class substitution diagnosis head remains completely frozen, unparameterized in the optimizer, and untrained.
4. **Frozen Backbone:** Pretrained Wav2Vec2 encoder remains frozen throughout training (`requires_grad = False`, eval mode during `.train()` calls).

### 5.3 Exact Hyperparameter Parity with Fold 1 Baseline R
All hyperparameters must be bit-identical to `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py`:

| Hyperparameter / Contract Element | Corrected Specification | Fold 1 Baseline R Invariant |
|---|---|---|
| **Master Random Seed** | `seed: 1337` | Bit-identical to Fold 1 Baseline R config |
| **Micro Batch Size** | `batch_size: 2` | 2 utterances per GPU micro-step |
| **Gradient Accumulation** | `gradient_accumulation_steps: 2` | Accumulate 2 micro-steps $\implies$ **Effective batch size = 4** |
| **Updates per Epoch** | $\approx 600$ updates | $\lceil 2,400 / 2 \rceil / 2 = 600$ updates per epoch |
| **Warmup Updates** | `warmup_updates: 100` | Fixed 100 optimizer updates with linear warmup |
| **Scheduler Horizon** | Cosine schedule over `max_epochs: 25` | Total horizon $T_{\text{max}} = 600 \times 25 = \mathbf{15,000 \text{ updates}}$ |
| **Peak Learning Rate** | `learning_rate: 0.001` (1e-3) | AdamW with $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$ |
| **Weight Decay** | `weight_decay: 0.01` | Weight decay on trainable projection heads |
| **Gradient Clipping** | `gradient_clip_norm: 1.0` | Global norm clipping |
| **Loss Formulation** | $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{ctc}} + 1.0 \cdot \mathcal{L}_{\text{binary}}$ | `MaskedBCEWithLogitsLoss` + CTC Loss |
| **Binary Prompt Dropout** | $p = 0.00$ (Disabled) | Baseline R standard; C1's $p=0.25$ was falsified |
| **Precision** | CUDA FP16 AMP | Initial scale 4,096 via `GradScaler` |
| **Early Stopping Semantics** | Patience = 5 epochs | Monitored on validation binary AP, guarded by `blank_fraction < 0.99` |

### 5.4 Checkpoint Selection Semantics: Dual Checkpoint Accounting
To prevent conflating training exposure with model performance, every fold training run will save and evaluate **two distinct checkpoints**:
1. **Fixed-Budget Checkpoint (`fixed_budget_checkpoint.pt`):**
   - Captured at strictly **Epoch 6 / 3,600 optimizer updates** (0 AMP skips).
   - Serves as the primary matched-exposure comparison point against Fold 1 Baseline R (which peaked at Epoch 6 / 3,600 updates) and C0/C1.
2. **Validation-Selected Checkpoint (`best.pt`):**
   - Checkpoint achieving the highest Validation Matched Binary AP with `blank_fraction < 0.99` prior to early stopping (patience 5, up to `max_epochs: 25`).
   - In Fold 1 Baseline R, `best.pt` coincided with Epoch 6. On Folds 2, 3, or 4, validation peak may occur at a different epoch.
3. **Endpoint Checkpoint (`last.pt`):**
   - State at training termination (e.g. when early stopping triggers).
   - In Fold 1, this occurred at Epoch 11 / 6,600 updates.

---

## 6. Remote Infrastructure, Quota Management & Remote Persistence

### 6.1 Compute Platform Allocation
- **Primary Platform:** Kaggle GPU (Tesla T4 $\times$ 2, private execution).
  - Quota: 30.00 hours/week available.
  - Expected runtime per fold: ~50 minutes for 6 epochs, up to ~80 minutes if training proceeds to 11 epochs.
  - Total compute required: $3 \text{ runs} \times 1.2 \text{ h} \approx \mathbf{3.6 \text{ GPU hours}}$ (< 12% of weekly quota).
- **Secondary Platform:** Google Colab GPU (Tesla T4). Reserved as fallback.

### 6.2 Data Staging Flow
1. **Packaging Script:** `audit_scripts/build_fold_packages.py` will generate:
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip`
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip`
   - `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip`
2. **Integrity Validation:** SHA-256 digests computed and verified locally before upload.
3. **Physical Isolation Audit:** Automated tests assert that no files belonging to $S_{\text{test}}^{(k)}$ exist in `fold{k}_train_val_only.zip`.
4. **Remote Persistence:** Atomic per-epoch checkpoint mirroring to private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints` under `fold2/`, `fold3/`, `fold4/` with cryptographic readback.

---

## 7. Cross-Fold Evaluation Metrics, Acoustic Controls & Aggregation Rules

### 7.1 Protocol O Evaluation & Deterministic Shift-1 Acoustic Control
For each fold model $M_k$ evaluated on its validation split $S_{\text{val}}^{(k)}$:

1. **Matched Audio Condition:**
   - Evaluates posterior probabilities $P(\text{Error} \mid \mathbf{x}_i, y_i)$ on matched audio-prompt pairs.
   - Computes Matched Binary Average Precision ($\text{AP}_{\text{matched}}^{(k)}$).
2. **Deterministic Within-Speaker Cyclic Shift-1 Mismatched Control:**
   - **Method:** `method="within_speaker_cyclic"` via `generate_derangement_indices()`.
   - **Mechanism:** For each speaker segment $[S_{\text{start}}, S_{\text{end}}]$ with length $L = S_{\text{end}} - S_{\text{start}}$, utterance index $i$ is assigned the audio waveform of utterance:
     $$\pi(i) = S_{\text{start}} + ((i - S_{\text{start}} + 1) \bmod L)$$
   - **Scientific Purpose:** Preserves the human speaker's vocal tract geometry, $F_0$, accent, and recording channel while completely breaking alignment with the prompt text.
   - Computes Mismatched Binary Average Precision ($\text{AP}_{\text{mismatched}}^{(k)}$).
3. **Acoustic Reliance Gap:**
   $$\text{Gap}^{(k)} = \text{AP}_{\text{matched}}^{(k)} - \text{AP}_{\text{mismatched}}^{(k)}$$
   A positive gap proves that the model utilizes acoustic evidence to detect errors rather than memorizing phonetic priors.
4. **Greedy CTC PER:**
   $$\text{PER}^{(k)} = \frac{S_k + D_k + I_k}{N_{\text{ref}}^{(k)}}$$
   Evaluated with greedy decoding (blank ID = 0, consecutive token collapse).
5. **Protocol Z (Zero-Oracle CMUdict G2P):**
   - Aligned-pair AP on CMUdict canonical phone sequences without gold TextGrid timestamps.

### 7.2 Exact Mathematical Definitions of Confusion Metrics
To eliminate ambiguity in CAPT and biometric terminology:
- **Positive Class ($b_i = 1$):** Mispronunciation / Phonetic Error.
- **Negative Class ($b_i = 0$):** Correct Pronunciation.

$$\begin{aligned}
\text{Precision} &= \frac{TP}{TP + FP} \\
\text{Recall (Sensitivity)} &= \frac{TP}{TP + FN} \\
\text{Specificity (True Negative Rate)} &= \frac{TN}{TN + FP} \\
\text{False Negative Rate (FNR)} &= \frac{FN}{TP + FN} = 1 - \text{Recall} \\
\text{False Positive Rate (FPR)} &= \frac{FP}{TN + FP} = 1 - \text{Specificity} \\
F_1\text{-Score} &= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\end{aligned}$$

**Pedagogical Interpretation:**
- $\text{FNR}$ represents the **Missed Error Rate** (pedagogically: falsely accepting an incorrect pronunciation).
- $\text{FPR}$ represents the **False Alarm Rate** (pedagogically: falsely rejecting a correct pronunciation).
- In all reports, metrics must be labeled explicitly as $\text{FNR}$ and $\text{FPR}$ with their mathematical formulas to prevent confusion with disparate biometric FAR/FRR definitions.

### 7.3 Cross-Fold Aggregation Rules & Small-Sample Limitations

#### 1. Macro-Averaging (Unweighted Fold Mean)
Treats each fold as an independent observation ($N=4$):
$$\overline{\text{AP}}_{\text{macro}} = \frac{1}{4} \sum_{k=1}^4 \text{AP}^{(k)}, \qquad \sigma_{\text{macro}} = \sqrt{\frac{1}{3} \sum_{k=1}^4 \left( \text{AP}^{(k)} - \overline{\text{AP}}_{\text{macro}} \right)^2}$$
Standard Error: $\text{SE}_{\text{macro}} = \frac{\sigma_{\text{macro}}}{\sqrt{4}} = \frac{\sigma_{\text{macro}}}{2}$.
Student's $t$ 95% Confidence Interval ($df = 3$, critical value $t_{0.025, 3} = 3.182$):
$$\text{CI}_{95\%, \text{macro}} = \left[ \overline{\text{AP}}_{\text{macro}} \pm 3.182 \cdot \text{SE}_{\text{macro}} \right]$$

#### 2. Micro-Averaging (Pooled Token-Level Evaluation)
Pools all out-of-fold validation predictions across all four folds ($N_{\text{val, pooled}} = \mathbf{39,576 \text{ tokens}}$, $E_{\text{val, pooled}} = \mathbf{4,213 \text{ error tokens}}$, prevalence $10.65\%$):
- Computes pooled Precision-Recall curve, pooled $\text{AP}_{\text{micro}}$, and pooled threshold confusion matrices.

#### 3. Critical Documentation of Small-Sample Limitations:
1. **Degrees of Freedom in Macro Statistics ($N=4$):** With only 4 folds ($df=3$), Student's $t$ critical multiplier ($3.182$) produces wide confidence intervals. This appropriately captures fold-level sampling variability, but has limited power to detect small effect sizes.
2. **Cluster Bootstrap Small-Sample Bias ($G=8$):** Resampling clusters of utterances across 8 validation speakers (`NJS`, `RRBI`, `TNI`, `YBAA`, `TXHC`, `ZHAA`, `HKK`, `NCC`) is subject to known small-cluster bias when cluster count $G < 30$ (Cameron, Gelbach & Miller, 2008). 
3. **Mandatory Disaggregated Reporting:** Because of these small-sample limitations, aggregated metrics must **always** be accompanied by disaggregated per-fold and per-speaker results. No claims of statistical certainty will be made without explicit disclosure of cluster sample size ($G=8$).

---

## 8. Research Roadmap: Separation of Validation, Model Improvement & Testing

```text
Detailed RQ1 Execution Roadmap
│
├── PHASE RQ1-TRN-03A: Baseline R Cross-Fold Training & Validation (Current Scope)
│   ├── Step 01: Plan cross-fold generalization phase and test isolation protocol (DONE)
│   ├── Step 02: Protocol audit, baseline parity correction, and regression tests (CURRENT)
│   ├── Step 03: Implement build_fold_packages.py and train configs (Pending Approval)
│   ├── Step 04: Train Baseline R on Folds 2, 3, 4 on Kaggle GPU (Pending Approval)
│   ├── Step 05: Checkpoint persistence and standalone CPU validation replay
│   └── Step 06: Synthesize 4-Fold Validation Baseline Report (Macro/Micro AP, Acoustic Gap)
│
├── PHASE RQ1-TRN-03B: Model Improvement & Phonological Feedback Exploration
│   ├── Candidate A: Cross-attention acoustic regularizers (feature-level dropout, query gates)
│   ├── Candidate B: Multi-task substitution diagnosis head rehabilitation
│   ├── Rule: Tuned strictly on pooled 4-fold validation sets (39,576 tokens)
│   └── Constraint: Test partitions remain 100% un-evaluated
│
├── PHASE RQ1-EVAL-01: Final Independent 4-Fold Cross-Validation Test
│   ├── Trigger: Only after final model architecture is frozen and approved
│   ├── Execution: One-shot evaluation on S_test^(1..4) (24 speakers, 3,599 utterances, 119,151 tokens)
│   └── Protocol: Evaluate frozen models M_1..M_4 on held-out test sets with frozen thresholds tau_k*
│
└── PHASE RQ1-CAPT: SpeechOcean762 Cross-Corpus Transfer & Educational Pronunciation Feedback
    ├── SpeechOcean762 zero-shot transfer evaluation (5,000 utterances)
    └── Pedagogical feedback generation (articulatory guidance, phoneme substitution trees)
```

*(Note: RQ2 is preserved exclusively for Multi-Agent Automated Test Generation).*

---

## 9. Research Risks, Execution Costs & Acceptance Gates

### 9.1 Research Risks & Mitigations

| Risk | Severity | Scientific Impact | Mitigation Protocol |
|---|:---:|---|---|
| **Inter-Fold AP Variance** | Medium | Validation AP may fluctuate across folds due to varying L1 error rates (e.g. Vietnamese 23.5% vs Korean 9.4%). | Report both Macro and Micro AP alongside L1-stratified breakdowns and per-speaker tables. |
| **CTC Blank Collapse on New Folds** | High | Risk of blank collapse on unobserved speaker distributions. | Hyperparameters use frozen backbone + peak lr $10^{-3}$, proven stable in RQ1-TRN-01B1 and Fold 1 runs. Driver enforces fail-closed length checks. |
| **Accidental Test Set Leakage** | Critical | Invalidates academic credibility and thesis defense. | Strict per-fold physical isolation: test audio is omitted at zip package creation time (`train_val_only.zip`). |
| **Confounding Budget with Selection** | High | Comparing models at different training updates masquerades exposure as treatment effect. | Mandatory dual checkpoint accounting: evaluate both Fixed-Budget (3,600 updates) and validation `best.pt`. |

### 9.2 Execution Budget

| Activity | Target Platform | Instances | Duration | Total GPU Budget | Quota Impact |
|---|---|:---:|:---:|:---:|:---:|
| **Fold 2 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50–80 min | ~1.2 GPU hours | ~4.0% weekly quota |
| **Fold 3 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50–80 min | ~1.2 GPU hours | ~4.0% weekly quota |
| **Fold 4 Baseline Training** | Kaggle Tesla T4 $\times$ 2 | 1 run | ~50–80 min | ~1.2 GPU hours | ~4.0% weekly quota |
| **Cross-Fold CPU Validation** | Local Laptop CPU | 3 evaluations | ~12 min | 0.0 GPU hours | Local CPU inference mode |
| **Total Phase Budget** | — | **3 GPU Runs** | **~3.6 Hours** | **~3.6 GPU Hours** | **< 12% Weekly Quota** |

### 9.3 Formal Acceptance Gates

- **Gate 1: Staging & Configuration Parity Gate (Local Laptop)**
  - Automated tests confirm `seed=1337`, `batch_size=2`, `accum=2`, `warmup=100`, `max_epochs=25`.
  - Staging script creates `fold2`, `fold3`, `fold4` packages with $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$ and 0 test files inside.
  - Test suite passes with 0 failures (`pytest tests/`).
- **Gate 2: Independent Training & Dual Checkpoint Persistence Gate (Remote GPU)**
  - Remote training runs for Folds 2, 3, and 4 complete with 0 unhandled exceptions and 0 AMP skips.
  - Fixed-budget checkpoint (Epoch 6 / 3,600 updates) and `best.pt` mirrored to Hugging Face with SHA-256 readback.
- **Gate 3: Cross-Fold Validation & Statistical Reporting Gate (Local Laptop)**
  - Standalone CPU evaluator executes Protocol O (matched and within-speaker shift-1 mismatched) and Protocol Z on validation partitions.
  - Synthesized report computes Macro AP, Micro AP, Acoustic Gap, CTC PER, FNR, FPR, and small-sample CIs.
  - All test partitions remain 100% un-evaluated.

---

## 10. Operator Action Packet & Required Decisions

In compliance with `AGENTS.md` ("Never start training without explicit user authorization. STOP before execution"):

### Exact Decisions Requiring Operator Approval:

1. **Approval of Corrected Baseline Parity Contract:**
   - Confirm baseline parameters for Folds 2, 3, 4: `seed: 1337`, `batch_size: 2`, `gradient_accumulation_steps: 2`, `warmup_updates: 100`, cosine schedule with `max_epochs: 25` horizon (~15,000 updates), early stopping patience 5 on validation AP.
2. **Approval of Dual Checkpoint Accounting:**
   - Formally approve capturing both Fixed-Budget Checkpoint (Epoch 6 / 3,600 updates) and Validation `best.pt` for each fold.
3. **Approval of Per-Fold Test Isolation & Staging Scope:**
   - Authorize implementation of `audit_scripts/build_fold_packages.py` to create `authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip` containing strictly train and val speakers, physically omitting active test speakers.
4. **Approval of Kaggle GPU Compute Allocation:**
   - Approve allocation of ~3.6 hours of Kaggle GPU quota (Tesla T4 $\times$ 2) across 3 training runs.
5. **Approval of Thesis Scope Boundaries:**
   - Confirm that SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback remain under RQ1, with RQ2 reserved exclusively for Multi-Agent Automated Test Generation.

**STATUS: EXECUTION HALTED. AWAITING OPERATOR REVIEW AND APPROVAL OF ITEMS 1–5.**
