# LANGFENS — RQ1-TRN-02B / 16-Final Three-Model Scientific Assessment and Research Gate Closure

**Recorded:** 2026-10-29T04:00:00+07:00 (Asia/Ho_Chi_Minh)<br>
**Source examined / pinned:** `b08842567ec92d0de000207a91b516f13c46ab13` on `review/rq1-trn-02b-kaggle-official-runs`<br>
**Process state:** `COMPLETED_FINAL_ASSESSMENT; RESEARCH_GATE_CLOSED`<br>
**Ablation Hypothesis Status:** **`NOT_SUPPORTED`** on Fold 1 under pre-registered matched exposure<br>
**Recommended Production Baseline:** **Reference R (`best.pt`, SHA-256 `a2445c8d...`)** retained as primary production anchor<br>
**Secondary Research Baseline:** **Control C0 (`fixed_budget_checkpoint.pt`, SHA-256 `0dc1d4fb...`)** noted for significant CTC PER improvement (22.02% vs 23.68%)<br>
**TEST Partition Isolation:** Strictly sealed (0 test utterances accessed, staged, or inspected)<br>

---

## 1. Executive Summary

This checkpoint establishes the final scientific consolidation of Phase RQ1-TRN-02B, evaluating and comparing all three Fold 1 models:
1. **Reference R (Historical Production Anchor):** Checkpoint `a2445c8d...`, trained in Phase 02A to 3,600 updates.
2. **Control Arm C0 ($p=0.00$):** Fine-tuning control, cold-initialized from Baseline R, trained on Kaggle Tesla T4 $\times$ 2 for strictly 6 epochs / 3,600 updates (0 skips).
3. **Treatment Arm C1 ($p=0.25$ Whole-Phone Prompt Dropout):** Ablation treatment, cold-initialized from the SAME Baseline R, trained on Kaggle Tesla T4 $\times$ 2 for strictly 6 epochs / 3,600 updates (0 skips).

The final assessment synthesizes primary Protocol O evaluation, secondary Protocol Z zero-oracle evaluation, disaggregated per-speaker analysis, 1,000-resample speaker-stratified cluster bootstrapping, pairwise decision agreement, and exploratory `best.pt` analysis.

### Primary Scientific Conclusions:
1. **Hypothesis Verdict:** The hypothesis that whole-phone binary prompt embedding dropout ($p=0.25$) improves acoustic conditioning is **NOT_SUPPORTED on Fold 1**.
2. **Primary Causal Contrast ($\Delta_{\text{dropout}} = \text{C1} - \text{C0}$):** $\mathbf{-0.007918}$ (Control C0 = **0.310304** vs. Treatment C1 = **0.302386**).
3. **Statistical Uncertainty (1,000 Bootstrap Resamples):** 95% CI is **`[-0.018328, +0.000385]`** (mean: $-0.008329$). In 96.3% of resamples, C1 is inferior to C0. Because the upper confidence limit touches zero, the evidence indicates lack of positive effect rather than an uncontested negative effect.
4. **Acoustic Reliance Gap:** Control C0 = **0.017543** (95% CI `[+0.0012, +0.0342]`) vs Treatment C1 = **0.011423** (95% CI `[-0.0030, +0.0252]`). Prompt dropout degraded reliance on acoustic context.
5. **Canonical Error Prior:** Treatment C1 (**0.302386**) underperforms the static train-only canonical phone error prior (**0.303063**).
6. **Zero-Oracle Protocol Z:** Aligned-only AP: Control C0 = **0.307582** vs Treatment C1 = **0.300992** ($\Delta_Z = \mathbf{-0.006590}$, coverage 99.63%).
7. **Production Recommendation:** Reference R is retained as the authoritative production baseline. Control C0 provides an important reference point for future joint multi-task optimization.

---

## 2. Technical Audits & Provenance Reconciliation

### 2.1 Resolution of NJS/RRBI Token-Count Discrepancy
- **Audit Findings:** An independent audit of `L2ArcticMDDDataset` and raw prediction records across all 300 validation utterances confirmed that the true non-padding token counts are:
  - Speaker `NJS` (150 utterances): Exactly **5,011 tokens** (641 error tokens, 4,370 correct tokens; prevalence = 12.7919%).
  - Speaker `RRBI` (150 utterances): Exactly **4,945 tokens** (516 error tokens, 4,429 correct tokens; prevalence = 10.4348%).
  - Total Validation Tokens: Exactly **9,956 tokens** (1,157 error tokens, 8,799 correct tokens; prevalence = 11.6211%).
- **Discrepancy Source:** The draft values `5,162` and `4,794` in Step 15 Markdown prose were an inadvertent manual transcription error (which shifted 151 tokens from RRBI to NJS). All JSON evaluation reports (`PAIRED_EVALUATION_REPORT.json` and `PAIRED_PREDICTION_RECORDS.json`) and historical baseline document `CORRECTED_PER_SPEAKER_CONTROLS.md` consistently record 5,011 for NJS and 4,945 for RRBI.
- **Resolution:** Corrected in `PAIRED_SCIENTIFIC_EVALUATION.md` and `15-cpu-paired-scientific-evaluation.md`. Documented authoritatively in `FINAL_THREE_MODEL_ASSESSMENT.md`.

### 2.2 Verification of Training Commit Parity
- **Audit of C0 vs C1 Training Commits:** Pinned commits `ce05e38` (C0) and `032b0b3` (C1) were diffed. The diff contains solely the added Kaggle deployment wrapper files `deploy/kaggle_c1/kaggle_notebook.ipynb` and `deploy/kaggle_c1/kernel-metadata.json`.
- **Invariance Proof:** Zero differences in model architecture, training loop, loss functions, optimizer schedule, dataset processing, or seed configuration. The intervention variable $p$ (0.00 vs 0.25) was the sole experimental distinction.

### 2.3 Credential Hygiene Review
- All persistent synchronization drivers (`audit_scripts/kaggle_checkpoint_sync.py`) resolve credentials dynamically via environment variables or local user OAuth token caches (`~/.cache/huggingface/token`).
- No plaintext tokens, API keys, or private identifiers were committed to git. All token patterns are protected by `.gitignore`.

---

## 3. Consolidated Three-Model Evaluation Suite

### 3.1 Primary Fixed-Budget Suite (6 Epochs / 3,600 Updates)

| Metric / Dimension | Reference R | Control C0 ($p=0.00$) | Treatment C1 ($p=0.25$) | Causal Delta ($\text{C1} - \text{C0}$) | Bootstrap 95% CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Matched Binary AP** | **0.317021** | 0.310304 | 0.302386 | **-0.007918** | **`[-0.018328, +0.000385]`** |
| **Mismatched Binary AP** | 0.294722 | 0.292761 | 0.290963 | -0.001798 | N/A |
| **Acoustic Reliance Gap** | **0.022299** | 0.017543 | 0.011423 | **-0.006120** | **`[-0.019675, +0.005596]`** |
| **Train Canonical Prior AP** | 0.303063 | 0.303063 | 0.303063 | 0.000000 | Baseline Anchor |
| **Margin over Prior** | +0.013958 | +0.007241 | **-0.000677** | -0.007918 | Negative for C1 |
| **Greedy CTC PER** | 23.68% | 22.02% | **22.01%** | -0.01% | Both pass guardrail |
| **Operating Point $\tau = 0.50$ Precision** | **49.14%** | 47.01% | 43.50% | -3.51% | False positives rise |
| **Operating Point $\tau = 0.50$ Recall** | **17.37%** | 9.51% | 13.31% | +3.80% | C0 is conservative |
| **Operating Point $\tau = 0.50$ Specificity** | 97.64% | **98.59%** | 97.73% | -0.86% | High true negatives |
| **Operating Point $\tau = 0.50$ $F_1$ Score** | **0.2567** | 0.1582 | 0.2038 | +0.0456 | Tradeoff shift |
| **Operating Point $\tau = 0.20$ Precision** | 29.18% | **33.53%** | 32.38% | -1.15% | C0 most precise |
| **Operating Point $\tau = 0.20$ Recall** | **53.85%** | 45.20% | 44.86% | -0.34% | R has highest recall |
| **Operating Point $\tau = 0.20$ $F_1$ Score** | 0.3785 | **0.3850** | 0.3761 | -0.0089 | C0 achieves peak F1 |
| **Protocol Z Aligned-Only AP** | **0.313530** | 0.307582 | 0.300992 | **-0.006590** | Consistent drop |
| **Protocol Z Token Coverage** | 99.63% | 99.63% | 99.63% | 0.00% | 9,919 / 9,956 tokens |

### 3.2 Disaggregated Per-Speaker Results

| Speaker | Dimension / Metric | Reference R | Control C0 | Treatment C1 | Causal Delta ($\text{C1} - \text{C0}$) |
|---|---|:---:|:---:|:---:|:---:|
| **Speaker NJS** | Matched AP | **0.394927** | 0.388309 | 0.390072 | +0.001763 |
| (5,011 tokens) | Acoustic Gap | **0.016183** | 0.006828 | 0.013370 | +0.006542 |
| (Prevalence 12.79%) | CTC PER | 23.41% | **21.52%** | 21.54% | +0.02% |
| **Speaker RRBI** | Matched AP | **0.251519** | 0.246000 | 0.236243 | **-0.009757** |
| (4,945 tokens) | Acoustic Gap | **0.016540** | 0.015444 | 0.004555 | **-0.010889** |
| (Prevalence 10.43%) | CTC PER | 23.95% | 22.53% | **22.49%** | -0.04% |

---

## 4. Primary vs. Exploratory Checkpoints (`best.pt`)

- **Fixed-Budget Contract:** The pre-registered primary evaluation point (3,600 updates / 6 epochs) strictly controls for training exposure. At this endpoint, Treatment C1 is inferior to Control C0 ($\Delta = -0.007918$).
- **Exploratory `best.pt`:** C1 reached its validation peak at Epoch 1 (600 updates; Matched AP 0.321258), followed by monotonic decline across epochs 2–6 (0.3204 $\to$ 0.3109 $\to$ 0.3021 $\to$ 0.3012 $\to$ 0.3026 $\to$ 0.3023). In contrast, C0 reached its peak at Epoch 4 (2,400 updates; Matched AP 0.314170).
- **Confounder Prohibition:** Comparing C1 at 600 updates to C0 at 2,400 updates introduces an 1,800-update exposure confounder. The exploratory `best.pt` results must not supersede the primary matched-budget evaluation.

---

## 5. Model Comparative Summary & Baseline Selection

1. **Reference R:** Best overall detector (AP 0.317021, Acoustic Gap 0.022299, Protocol Z AP 0.313530). Retained as the authoritative production baseline.
2. **Control C0:** Best speech recognition performance (CTC PER 22.02%, p < 10^-5 vs R), high specificity (98.59%), with minor AP drift (-0.0067 vs R). Retained as a secondary reference model for multi-task research.
3. **Treatment C1:** Falsified ablation candidate. Degraded AP (0.302386), collapsed acoustic gap on speaker RRBI (0.004555), and increased false positives. Rejected.

---

## 6. Output Artifacts & Final Gate

- Detailed Final Assessment: `manifests/RQ1-TRN-02B/FINAL_THREE_MODEL_ASSESSMENT.md`
- Step Checkpoint 16: `manifests/RQ1-TRN-02B/checkpoints/16-final-three-model-assessment-and-gate-closure.md`
- Phase Overview & Index: `manifests/RQ1-TRN-02B/PROCESS_CHECKPOINT.md` and `docs/process-checkpoints/INDEX.md`
- Evaluation Evidence: `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json`
- Prediction Records: `manifests/RQ1-TRN-02B/evaluation/PAIRED_PREDICTION_RECORDS.json`

**RESEARCH GATE CLOSED:** No further training, hyperparameter tuning, or evaluation will be executed under RQ1-TRN-02B. Subsequent research phases are deferred pending explicit operator authorization.
