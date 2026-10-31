# LANGFENS — RQ1-TRN-03A / 15-RQ1-TRN-03B Experimental Design & Readiness

**Status:** `COMPLETED; GATE_A_PASSED_DESIGN_READY; EXECUTION_HALTED_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-11-01T12:00:00+07:00

**Evidence source:** Split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`), Baseline R 4-fold reports (`manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json` SHA-256 `43f7de4e...`, `FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz` SHA-256 `355b6e23...`), loss modules `src/langfens_prep/losses.py`, architecture `src/langfens_prep/model.py`, trainer `audit_scripts/train_fold1_ctc_binary.py`, pinned configs `configs/rq1_trn_03b_*.yaml`, automated regression test suite (`tests/test_rq1_trn_03b_experimental_design.py`).

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Local Laptop Static Analysis, Loss Unit Tests, Invariant Audits & Parity Regressions, 310/310 passed)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary & Objective

In compliance with `AGENTS.md` and the master research plan, this checkpoint formally establishes the experimental design, mathematical loss contracts, pinned configuration ledger, and two-stage execution gates for Phase **RQ1-TRN-03B (Model Improvement & Pedagogical Operating Optimization)**.

**Zero remote GPU training, zero backward passes, zero local training, zero test set access, and zero baseline checkpoint modifications were executed during this step.**

The primary objective of Phase RQ1-TRN-03B is to evaluate hypothesis-driven, one-variable-at-a-time (OVAT) loss interventions targeting the extreme class imbalance (10.64% error prevalence) documented in Baseline R, elevating the Precision-Recall curve to achieve pedagogically viable operating points without altering the underlying neural architecture or degrading CTC acoustic grounding.

---

## 2. Closure of Residual Step 14 Issues

Before advancing to experimental readiness, five residual technical and terminological issues from Step 14 were audited, resolved, and verified via automated regression tests:

### 2.1 Actual 1,199 Validation Utterances vs. Hardcoded 1,200
- **Audited Discrepancy:** The split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` contains exactly 1,199 out-of-fold validation utterances across the four folds:
  - Fold 1 (`NJS`, `RRBI`): 300 utterances
  - Fold 2 (`TNI`, `YBAA`): 299 utterances (`TNI`: 149, `YBAA`: 150)
  - Fold 3 (`TXHC`, `ZHAA`): 300 utterances
  - Fold 4 (`HKK`, `NCC`): 300 utterances
  - Total: $300 + 299 + 300 + 300 = \mathbf{1,199 \text{ utterances}}$
- **Resolution:**
  - `audit_scripts/evaluate_cross_fold_validation.py` was refactored: removed hardcoded `1200` and replaced with dynamic `len(all_utterances_records)` (1,199).
  - `manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json` metadata `total_out_of_fold_validation_utterances` corrected to `1199` (SHA-256 updated to `43f7de4e...`).
  - Confirmed that `FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz` already tracks exactly `len(utterances) == 1199`.
  - Regression test `test_actual_1199_validation_utterances_across_all_four_folds` passes.

### 2.2 Evaluator Runtime Provenance vs. Introducing Commit
- **Audited Discrepancy:** Distinguishing the Git commit that introduced the cross-fold evaluation harness (`audit_scripts/evaluate_cross_fold_validation.py`) from subsequent execution-time runtime commit SHAs.
- **Resolution:**
  - Authoritative introducing commit certified as **`618e86d`** (`feat(rq1-trn-03a): complete Fold 4 closure and four-fold validation synthesis`).
  - Both `evaluator_source_commit: "618e86d"` and runtime `evaluator_git_sha` are permanently recorded in report and prediction record metadata, providing immutable Git ancestry.
  - Regression test `test_evaluator_runtime_provenance_and_introducing_commit` passes.

### 2.3 Stale Regression Test Counts
- **Audited Discrepancy:** Step 14 checkpoint narrative cited "298 passed", recorded before the final two prediction archive and macro prevalence assertions were added in `tests/test_rq1_trn_03a_four_fold_closure.py` (which brought passing count to 300).
- **Resolution:**
  - Stale references in Step 14 checkpoint and `PROCESS_CHECKPOINT.md` updated to authoritative count: **300 passed, 1 skipped, 1 warning**.
  - With Step 15's 10 new design regression tests, the total repository test count is now **310 passed, 1 skipped, 1 warning**.

### 2.4 Cross-Attention Query / Key / Value Terminology
- **Audited Discrepancy:** Section 4 of Step 14 draft narrative inverted cross-attention projections, stating "apply targeted dropout to the cross-attention key/value projections from canonical text, forcing the query to rely on audio frame representations".
- **Resolution:**
  - Certified against `src/langfens_prep/model.py` (`CanonicalAcousticCrossAttention`):
    - **Query ($Q$):** Canonical text prompt sequence (`canonical_queries = phone_proj(phone_embeds) + pos_embeds`).
    - **Key ($K$) & Value ($V$):** Acoustic frame representations from the frozen Wav2Vec2 backbone (`frame_features`).
  - Phrasing corrected: The canonical text queries attend over acoustic keys and values. Any acoustic regularizer modulates Keys/Values, while prompt dropout modulates Queries.
  - Regression test `test_cross_attention_query_key_value_mapping` verifies tensor shapes and mapping roles.

### 2.5 Multi-Task Vocabulary Cardinality (39-Class Diagnosis vs. 40-Class CTC)
- **Audited Discrepancy:** Prior narrative loosely referenced "40-class substitution diagnosis head" or "41 phoneme classes".
- **Resolution:**
  - Certified against `src/langfens_prep/phoneme_normalizer.py` and `src/langfens_prep/model.py`:
    - **Realized CTC Head:** **40 classes** (1 CTC blank token at index 0 + 39 standard ARPAbet phonemes at indices 1..39).
    - **Substitution Diagnosis Head:** **39 classes** (39 standard ARPAbet phonemes at indices 0..38; no blank token; active exclusively for mispronounced tokens with `ignore_index = -100`).
  - Regression test `test_vocabulary_cardinality_and_head_separation` asserts linear head dimensions.

---

## 3. Preservation of Baseline R Benchmarks & Test Sealing

All prior evidence, checkpoints, metrics, and split manifests are preserved without modification:

### 3.1 Immutable Checkpoint Ledger

| Fold | Checkpoint Identifier | Remote Persistent Store (HF) | Updates / Epoch | SHA-256 Digest | Status |
|:---:|---|---|:---:|---|:---:|
| **Fold 1** | `best.pt` | Google Drive (`10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`) | 3,600 / Ep 6 | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | Verified |
| **Fold 2** | `fixed_budget_checkpoint.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 3,600 / Ep 6 | `ab4c589445feac42636296cd36d7cf2ae7eba7360445f0084454c87fa2fef508` | Verified |
| **Fold 2** | `best.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 1,200 / Ep 2 | `b9032ee9ba8d5f8f03c8ce164237a49c9524442281b183d9ee522eb8d2b759ff` | Verified |
| **Fold 3** | `fixed_budget_checkpoint.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 3,600 / Ep 6 | `bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441` | Verified |
| **Fold 3** | `best.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 3,600 / Ep 6 | `679f15c545384b90d003d9184d42ab945c7c7adcc67029dfdef7f79c93f25eea` | Verified |
| **Fold 4** | `fixed_budget_checkpoint.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 3,600 / Ep 6 | `c502271feba097781a6b0bdfe413085271d3a03647353f272dccd29bbb1499f6` | Verified |
| **Fold 4** | `best.pt` | HF: `minkhoaa/langfens-rq1-cross-fold-checkpoints` | 1,800 / Ep 3 | `3daa8ca0db8a632050e3f965c705df2d271d07a450cb7110a9780537d623274d` | Verified |

### 3.2 Authoritative 4-Fold Baseline R Metrics (Fixed-Budget Milestone)
- **Macro Matched Binary AP:** **`0.279207 ± 0.026467`** (Student's $t$ 95% CI: `[0.194990, 0.363424]`)
- **Micro Pooled Binary AP:** **`0.272580`** ($N=39,576$ tokens, 4,213 error tokens)
- **Macro Acoustic Reliance Gap:** **`+0.013064 ± 0.003997`** (Student's $t$ 95% CI: `[+0.000344, +0.025784]`, strictly positive)
- **Macro CTC PER:** **`23.90% ± 0.31%`**
- **Default Threshold Error Rates ($\tau = 0.50$):** $\text{FNR} = \mathbf{83.53\%}$, $\text{FPR} = \mathbf{2.93\%}$
- **Operating Threshold Error Rates ($\tau = 0.20$):** $\text{FNR} = \mathbf{51.86\%}$, $\text{FPR} = \mathbf{17.94\%}$

### 3.3 Strict Per-Fold Test Isolation
Across all 4 folds:
- $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$, $S_{\text{val}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$, $S_{\text{train}}^{(k)} \cap S_{\text{val}}^{(k)} = \emptyset$.
- Zero test split files ($S_{\text{test}}^{(1..4)}$, 3,599 utterances across 24 speakers) are accessed, unpacked, staged, or evaluated.

---

## 4. Scientific Prioritization of Arm 2A & Arm 2B

### 4.1 Rationale for Prioritizing Loss Interventions Over Architecture
1. **Severe Imbalance is the Proven Primary Bottleneck:** In Baseline R, error prevalence is only 10.6430% (pooled 10.6453%). The 89.35% correct tokens overwhelmingly dominate the standard unweighted BCE loss, forcing the model toward conservative predictions where 83.5% of errors are missed at $\tau=0.50$.
2. **One-Variable-at-a-Time Cleanliness:** Altering the cross-attention architecture (Arm 1) or adding a multi-task head (Arm 3) introduces confounding variables (new parameters, gradient interference). In contrast, loss re-weighting (Arm 2) preserves the neural architecture, parameter count, and initialization bit-for-bit identical to Baseline R.
3. **Complementary Imbalance Strategies:**
   - **Arm 2A (Focal BCE):** Non-linear margin-based downweighting of easy negatives, focusing gradients dynamically on ambiguous and hard mispronunciations.
   - **Arm 2B (Class-Weighted BCE):** Linear static scaling of the positive error gradient by the inverse prevalence ratio.

### 4.2 Mathematical Formulations

#### Arm 2A: Masked Focal BCE Loss
$$\mathcal{L}_{\text{focal}}(z, y) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \left[ y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right]$$
where:
$$p = \sigma(z), \quad p_t = y \cdot p + (1 - y) \cdot (1 - p), \quad \alpha_t = y \cdot \alpha + (1 - y) \cdot (1 - \alpha)$$
- Pinned Hyperparameters: $\gamma = 2.0$ (focusing parameter), $\alpha = 0.75$ (positive error class weight).
- At $\sigma(z) \to 0$ for $y=0$ (easy correct negative), $(1 - p_t)^\gamma \to 0$, suppressing irrelevant background gradients.

#### Arm 2B: Masked Class-Weighted BCE Loss
$$\mathcal{L}_{\text{weighted}}(z, y) = -\left[ w_{\text{pos}} \cdot y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right]$$
- Pinned Hyperparameter: $w_{\text{pos}} = \mathbf{8.40}$
- Derivation: Based on the authoritative macro error prevalence $\overline{\text{Prev}} = 10.6430\%$:
  $$w_{\text{pos}} = \frac{1 - \overline{\text{Prev}}}{\overline{\text{Prev}}} = \frac{1 - 0.10643}{0.10643} \approx 8.396 \implies \mathbf{8.40}$$

---

## 5. One-Variable-at-a-Time (OVAT) Experimental Matrix

All arms maintain strict bit-level hyperparameter parity with Baseline R, varying exclusively the binary loss criterion:

| Parameter / Contract | Baseline R (Reference) | Arm 2A (Focal BCE) | Arm 2B (Class-Weighted BCE) |
|---|:---:|:---:|:---:|
| **Backbone** | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` |
| **Backbone State** | Frozen (`requires_grad = False`) | Frozen (`requires_grad = False`) | Frozen (`requires_grad = False`) |
| **Trainable Heads** | CTC Head + Cross-Attention + Binary MLP | CTC Head + Cross-Attention + Binary MLP | CTC Head + Cross-Attention + Binary MLP |
| **Random Seed** | `1337` | `1337` | `1337` |
| **Batch / Accumulation** | Micro-batch 2, Accumulation 2 (Eff. 4) | Micro-batch 2, Accumulation 2 (Eff. 4) | Micro-batch 2, Accumulation 2 (Eff. 4) |
| **Learning Rate / Schedule** | $1 \times 10^{-3}$, AdamW, Cosine 25 Ep | $1 \times 10^{-3}$, AdamW, Cosine 25 Ep | $1 \times 10^{-3}$, AdamW, Cosine 25 Ep |
| **Warmup Updates** | 100 fixed updates | 100 fixed updates | 100 fixed updates |
| **Execution Cap** | 10 completed epochs (max 6,000 updates)| 10 completed epochs (max 6,000 updates)| 10 completed epochs (max 6,000 updates)|
| **Early Stopping** | Patience 5 on Val AP (`blank < 0.99`) | Patience 5 on Val AP (`blank < 0.99`) | Patience 5 on Val AP (`blank < 0.99`) |
| **Precision** | CUDA FP16 AMP (init scale 4096) | CUDA FP16 AMP (init scale 4096) | CUDA FP16 AMP (init scale 4096) |
| **Binary Loss Criterion** | `MaskedBCEWithLogitsLoss(pos_weight=None)` | `MaskedFocalBCEWithLogitsLoss(gamma=2.0, alpha=0.75)` | `MaskedBCEWithLogitsLoss(pos_weight=8.40)` |
| **CTC Loss Weight** | 1.0 | 1.0 | 1.0 |
| **Binary Loss Weight** | 1.0 | 1.0 | 1.0 |

### 5.1 Pinned Configuration Artifacts
- Pilot configs: `configs/rq1_trn_03b_pilot_arm2a_focal.yaml`, `configs/rq1_trn_03b_pilot_arm2b_weighted.yaml`
- Full 4-fold Arm 2A configs: `configs/rq1_trn_03b_arm2a_fold{1,2,3,4}.yaml`
- Full 4-fold Arm 2B configs: `configs/rq1_trn_03b_arm2b_fold{1,2,3,4}.yaml`

---

## 6. Endpoints, Metrics & Pedagogical Operating Points

### 6.1 Dual Checkpoint Accounting
To prevent confounding training duration with treatment effect:
1. **Primary Milestone Endpoint:** Strictly **Epoch 6 / 3,600 optimizer updates** (0 AMP skips). Provides identical compute exposure for paired hypothesis testing.
2. **Secondary Exploratory Endpoint:** Validation-selected **`best.pt`** (peak validation AP within patience 5). Accounts for potential shifts in convergence rate under focal/weighted loss.

### 6.2 Evaluation Protocols
- **Protocol O:** Matched audio AP, deterministic within-speaker cyclic shift-1 mismatched AP, acoustic gap $\Delta_{\text{gap}} = \text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}}$.
- **Protocol Z:** Zero-oracle G2P inference without gold TextGrids, DP Levenshtein alignment for scoring.
- **CTC PER Guardrail:** Greedy decoding edit distance $\le \mathbf{26.0\%}$ and validation blank fraction $\le \mathbf{0.85}$.

### 6.3 Pedagogical Operating Points (FNR at Fixed FPR)
Because class-weighting alters the logit distribution (shifting raw sigmoid probabilities upward), static threshold comparisons ($\tau = 0.50$) are methodologically invalid across differently weighted losses. Models must be evaluated at **fixed False Alarm Rates (FPR)**:
1. **Low False Alarm Mode ($\text{FPR} \le 10\%$):** Strict feedback for independent learners; minimizes student frustration.
2. **Standard Pedagogical Operating Point ($\text{FPR} \le 15\%$):** Balanced classroom/tutoring mode; target $\text{FNR} < 40\%$.
3. **High Sensitivity Mode ($\text{FPR} \le 20\%$):** Permissive mode for diagnostic screening.

### 6.4 Statistical Uncertainty Reporting
- Paired fold differences: $\Delta \text{AP}^{(k)} = \text{AP}_{\text{Arm}}^{(k)} - \text{AP}_{\text{Baseline}}^{(k)}$
- Student's $t$ paired 95% CI with $df=3$ ($t_{0.025, 3} = 3.182$).
- Mandatory disclosure of small-sample cluster bias ($G=8$ validation speakers).

---

## 7. Assessment of Validation Reuse & Model-Selection Bias

A critical methodological boundary must be formally acknowledged:
1. **Validation Reuse:** The 8 validation speakers ($S_{\text{val}}^{(1..4)}$, 1,199 utterances, 39,576 tokens) were already utilized during Phase RQ1-TRN-03A to benchmark Baseline R and evaluate the acoustic gap.
2. **Selection Bias:** Utilizing this same validation set to compare Arm 2A and Arm 2B introduces model-selection bias (optimization reuse).
3. **No Independent Statistical Confirmation:** Out-of-fold validation comparisons in Phase RQ1-TRN-03B guide engineering decisions and hypothesis pruning, but **cannot and will not be claimed as independent confirmatory proof of generalization**.
4. **Confirmatory Testing Reserved for Tier 3:** Authoritative, publication-grade statistical confirmation is reserved exclusively for the 24 held-out test speakers ($S_{\text{test}}^{(1..4)}$, 3,599 utterances, 119,151 tokens) evaluated strictly once under frozen thresholds in Phase **RQ1-EVAL-01**.

---

## 8. Two-Stage Execution Strategy & Authorization Gates

To preserve compute quota and enforce research discipline:

```text
RQ1-TRN-03B Phased Execution Flow
│
├── [CURRENT STEP] Step 15: Experimental Design & Readiness (GATE A PASSED)
│   ├── Audit & close Step 14 residual issues
│   ├── Implement MaskedFocalBCEWithLogitsLoss & pos_weight buffer
│   ├── Author pinned immutable configs for Pilot and 4-Fold runs
│   └── Verify 310 regression tests (100% green)
│
├── [OPERATOR AUTHORIZATION REQUIRED] Gate B: Remote-GPU Pilot Launch
│   ├── Target: Train Arm 2A and Arm 2B on Fold 1 (3,600 updates / 6 epochs each)
│   ├── Platform: Kaggle Tesla T4 × 2 (Private Execution)
│   ├── Budget: 2 runs × ~50 min = ~1.7 GPU hours (< 6% of weekly quota)
│   └── Verification: Numerical stability, zero AMP skips, CTC PER ≤ 26.0%, PR curve check
│
├── [OPERATOR DECISION REQUIRED] Gate C: Pilot Review & Selection
│   ├── Review Pilot Fold 1 AP, acoustic gap, and PR curve elevation
│   └── Select winning arm (or authorize dual deployment if margin is inconclusive)
│
└── [OPERATOR AUTHORIZATION REQUIRED] Gate D: Full 4-Fold Cross-Validation Launch
    ├── Target: Train selected arm across remaining Folds 2, 3, 4
    ├── Budget: 3 runs × ~50 min = ~2.5 GPU hours (< 9% of weekly quota)
    └── Synthesis: 4-fold paired validation AP, acoustic gap, and FNR@15%FPR comparison
```

### 8.1 Total Estimated GPU Workload
- **Pilot Stage (Fold 1):** 2 runs $\times$ ~50 min $\approx$ **1.7 GPU hours**
- **Full 4-Fold Stage (Folds 2–4 for winning arm):** 3 runs $\times$ ~50 min $\approx$ **2.5 GPU hours**
- **Total RQ1-TRN-03B GPU Workload:** **~4.2 GPU hours** (< 14% of weekly 30.00h Kaggle quota).

---

## 9. Scientific Acceptance Criteria

For an intervention to be declared successful and advanced to final test evaluation in RQ1-EVAL-01:
1. **Primary Metric (AP Elevation):** Paired mean $\overline{\Delta \text{AP}} = \overline{\text{AP}}_{\text{Arm}} - \overline{\text{AP}}_{\text{Baseline}} > \mathbf{+0.020}$ across the 4 folds at the 3,600-update primary milestone.
2. **Acoustic Reliance Integrity:** Macro Acoustic Gap $\overline{\Delta_{\text{gap}}} \ge \mathbf{+0.010}$ with 95% Student's $t$ CI lower bound $\ge 0.000$ (no acoustic decoupling).
3. **Pedagogical Error Rate:** $\text{FNR}_{@15\%\text{FPR}} \le \mathbf{40.0\%}$ (detecting $\ge 60\%$ of pronunciation errors at false alarm rate $\le 15\%$).
4. **Acoustic Model Stability:** Greedy CTC PER $\le \mathbf{26.0\%}$ and blank emission fraction $\le \mathbf{0.85}$ across all 4 folds.

---

## 10. Next Authorized Execution Step & Stop Statement

In strict adherence to `AGENTS.md` ("Never start training without explicit user authorization. STOP before execution"):

- **Status:** Gate A PASSED. Design and regression tests complete.
- **Execution Boundary:** No remote GPU runs launched. No local training attempted. No test sets accessed.
- **Next Step Requiring Operator Authorization:** **Gate B** — Staging and launching the Remote-GPU Pilot on Fold 1 for Arm 2A and Arm 2B on Kaggle Tesla T4 $\times$ 2.

**STOP. AWAITING OPERATOR REVIEW AND AUTHORIZATION FOR GATE B.**
