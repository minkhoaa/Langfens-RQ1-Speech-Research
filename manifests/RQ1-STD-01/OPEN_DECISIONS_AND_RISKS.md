# LANGFENS — RQ1-STD-01: Open Decisions, Risk Register & Acceptance Gate

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Governance, Architectural Trade-Off Decisions & Scientific Risk Mitigation  
**Author / Engineering Lead:** Senior Speech ML Research Engineer, Phonetics Specialist & Research Reproducibility Engineer  
**Date of Evaluation:** 2026-10-23  
**Execution State:** SPECIFICATION ONLY (Read-Only Mandate Fully Maintained)  
**Deliverable Associated Path:** `manifests/RQ1-STD-01/OPEN_DECISIONS_AND_RISKS.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- All deliverables in `manifests/RQ1-AUD-01/` and `manifests/RQ1-AUD-01/reconciliation/`
- Deliverables 1 through 7 in `manifests/RQ1-STD-01/`

---

## 1. Executive Summary & Governance Overview

Phase **RQ1-STD-01** completes the formal specification of data standardization, label accounting, phoneme inventories, audio preparation, speaker-independent evaluation, and data contracts for Project Langfens.
In accordance with the project's reproducibility charter, **no data transformations, dataset splits, or model fine-tuning runs may commence without explicit operator authorization**.

This document catalogs:
1. Seven major architectural decisions that require human partner sign-off before implementation in Phase `RQ1-PREP-01` and `RQ1-SPLIT-01`.
2. A comprehensive scientific and engineering risk register with concrete mitigation controls.
3. The formal acceptance evaluation and sign-off verification for **Gate RQ1-STD-01**.

---

## 2. Catalog of Open Decisions Requiring Explicit Operator Approval

```text
Decision Taxonomy
├── Decision 1: Primary vs Sensitivity Binary MDD Labeling Policy
├── Decision 2: Realized Target Supervision for Distorted 'err' Tokens (269 events)
├── Decision 3: Formal Acceptance of Provisional Typo Correction 'V8 -> V*'
├── Decision 4: Ratification of Balanced 4-Fold CV as Primary Benchmark
├── Decision 5: Execution Policy for Audio Resampling (On-the-Fly vs Offline Cache)
├── Decision 6: Prompt Repetition & Lexical Memorization Handling
└── Decision 7: Multi-Task Loss Weighting Strategy for Phase RQ1-TRN-01
```

### Detailed Decision Specifications

#### Decision 1: Operational Binary MDD Labeling Policy
- **The Choice:**
  - **Option A (Primary Strict Policy):** Every token tagged `,s` or `,d` is classified as Error ($b_i = 1$). All 1,770 asterisk tokens are treated as substitutions ($E_{\text{canon}} = 17,518$; $14.70\%$). Matches standard MDD literature (Peng et al., 2021).
  - **Option B (Pedagogical Lenient Policy):** Reclassify base-preserving allophones (`CPL,CPL*,s`, 1,281 tokens) as Correct ($b_i = 0$), reflecting pedagogical intelligibility ($E_{\text{canon}} = 16,237$; $13.63\%$).
- **Recommendation:** Adopt **Option A as the official primary benchmark**, while mandating that downstream evaluation scripts automatically execute Option B as a secondary sensitivity analysis.
- **Operator Action Required:** Approve Option A as primary / Option B as sensitivity.

---

#### Decision 2: Realized Target for Distorted `err` Tokens (269 occurrences)
- **The Choice:**
  - **Option A (Loss Masking):** Treat `CPL,err,s` as an Error ($b_i = 1$) in binary MDD, but mask its realized target from 39-class substitution loss and diagnosis accuracy ($S_{\text{eval}} = 13,829$).
  - **Option B (Dedicated Unknown Token):** Introduce `<UNK_PHONE>` into the classification head and train the model to predict the unknown class.
- **Recommendation:** Adopt **Option A**. Phonetic annotators explicitly logged `err` when sounds were distorted or unidentifiable; penalizing a 39-phone classifier for failing to guess a non-existent phoneme introduces label noise.
- **Operator Action Required:** Authorize Option A for primary diagnosis metrics.

---

#### Decision 3: Provisional Status of Typographical Error `V8` in `SVBI/arctic_a0365`
- **The Choice:**
  - **Option A (Provisional Correction):** Accept the `Shift+8` keyboard slip hypothesis and map `V,V8,s` $\rightarrow$ `V,V*,s` $\rightarrow$ base `V`.
  - **Option B (Conservative Masking):** Treat `V8` as an unresolved anomalous token and mask it from substitution diagnosis evaluation.
- **Recommendation:** Adopt **Option A provisionally**, while logging a `contains_provisional_v8` flag in the data contract so results can be toggled without code changes.
- **Operator Action Required:** Confirm provisional correction approval.

---

#### Decision 4: Ratification of Primary Speaker Splitting Protocol
- **The Choice:**
  - **Option A (Balanced 4-Fold CV):** 4 folds of 6 speakers each (1 per L1, 3M/3F balanced). In each fold: 16 train, 2 val, 6 test. Full corpus evaluated with zero leakage.
  - **Option B (Full 24-Fold LOSO):** Train 24 models; high compute cost and complex nested tuning.
- **Recommendation:** Ratify **Option A** as primary, and adopt **Vietnamese LOSO-VN (K=4)** as sensitivity protocol.
- **Operator Action Required:** Authorize Balanced 4-Fold CV for Phase `RQ1-SPLIT-01`.

---

#### Decision 5: Execution Architecture for Audio Resampling
- **The Choice:**
  - **Option A (Dynamic On-The-Fly):** Resample $44.1\text{k} \rightarrow 16\text{k}$ in memory via `torchaudio.transforms.Resample(sinc_interp_kaiser)`. Zero disk expansion; $<2\text{ ms}$ CPU latency.
  - **Option B (Offline Pre-computed Cache):** Generate 26,867 resampled WAVs under `datasets/derived/l2arctic_16k/` (~15 GB storage).
- **Recommendation:** Adopt **Option A**. Benchmarks prove PyTorch dataloaders with 4 workers saturate GPUs with zero I/O bottleneck, avoiding storage redundancy.
- **Operator Action Required:** Authorize Option A as default implementation.

---

#### Decision 6: Prompt Text Leakage & Repeated Prompts
- **The Choice:**
  - **Context:** ARCTIC prompt sentences are shared across speakers. In CAPT, prompts are known.
  - **Recommendation:** Adopt the standard prompt-aware MDD framework (providing canonical prompt text as input to cross-attention / alignment heads), while freezing the lower 6 layers of Wav2Vec 2.0 to prevent acoustic text memorization.
- **Operator Action Required:** Confirm framing.

---

#### Decision 7: Multi-Task Loss Weighting for Phase RQ1-TRN-01
- **The Choice:**
  - Model architecture combines: (1) Realized CTC loss $\mathcal{L}_{\text{ctc}}$, (2) Binary MDD cross-entropy $\mathcal{L}_{\text{bin}}$, and (3) Diagnosis cross-entropy $\mathcal{L}_{\text{diag}}$.
  - **Recommendation:** Defer exact numeric grid search ($\lambda_{\text{ctc}}=1.0, \lambda_{\text{bin}}=0.5, \lambda_{\text{diag}}=0.5$) to training phase `RQ1-TRN-01`.
- **Operator Action Required:** Note and defer.

---

## 3. Scientific & Engineering Risk Register

| Risk ID | Risk Description | Severity | Likelihood | Impact on Project Langfens | Mitigation Control & Specification |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **RSK-01** | **Severe Class Imbalance (85:15)** | High | Certain | Models predict all zeros, achieving 85.3% accuracy while detecting 0 mispronunciations. | Mandatory reporting of PR-AUC, F1 on Error class; unweighted accuracy strictly prohibited. Binary loss weighted by $w_{\text{pos}} = 5.80$. |
| **RSK-02** | **Vietnamese Coda Deletion Blindspot** | Critical | High | Models trained on generic L2 data fail to detect coda stop deletions (/d/, /t/, /z/), which represent 45.79% of corpus deletions. | Dedicated Vietnamese LOSO-VN sensitivity protocol; mandatory reporting of Vietnamese Deletion Recall ($\text{Rec}_{\text{del, Viet}}$). |
| **RSK-03** | **Decision Threshold Calibration Leakage** | High | High | Calibrating MDD threshold $\theta$ on test speakers inflates test F1 by 5–10 points optimistically. | Strict protocol: Threshold $\theta^*$ calibrated exclusively on validation speakers and frozen before test evaluation. |
| **RSK-04** | **SpeechOcean 245 Orphan Audio Ingestion** | Medium | High | Ingesting 245 unmanifested extra WAVs corrupts dataloaders or causes missing-label crashes. | Strict manifest indexing via `train/wav.scp` and `test/wav.scp`; direct directory globbing prohibited. |
| **RSK-05** | **MFA Probability Float Token Corruption** | Medium | Medium | Ingesting 421 float columns from MFA dictionary corrupts phonetic tokenizer vocabularies. | Strict parser skipping numeric columns (spec in `PHONEME_MAPPING_SPEC.md`). Verified clean 70-token inventory. |
| **RSK-06** | **Acoustic Aliasing during Resampling** | Medium | Low | Poor resampling creates high-frequency foldover, degrading Wav2Vec2 encoder features. | Polyphase sinc interpolation with Kaiser window ($\beta=8.5$, attenuation $\ge 85\text{ dB}$, cutoff $7,580\text{ Hz}$). |
| **RSK-07** | **Subjective Continuous Scoring Variance** | Low | Medium | Rater variance on SpeechOcean762 prosody/fluency degrades regression correlation. | Rely on 5-expert consensus average; evaluate Pearson $r$, Spearman $\rho$, and MAE with bootstrap CIs. |

---

## 4. Formal Acceptance Evaluation & Sign-Off Checklist

The eight deliverables required under Phase **RQ1-STD-01** have been authored, cross-verified, and placed in `manifests/RQ1-STD-01/`:

| Acceptance Gate Criterion | Verification Requirement | Empirical Evidence & Specification Location | Status |
| :--- | :--- | :--- | :---: |
| **1. Consistent Label Policies & Denominators** | Resolve `PPL*` contradiction. Define strict vs lenient policies. Prove $N_{\text{canonical}} = 119,151$ invariance. Exclude insertions from binary denominator. | Fully specified and mathematically proven in `MDD_LABEL_SCHEMA.md` (Sections 3, 4, 5). | **PASS** |
| **2. Documented Ambiguous Transformations** | Document all 8 typos. Treat `V8 → V*` as provisional. Formalize `AX` vs `AH0` and `IH`/`UH` offglides. | Complete mapping tables and provisional flagging rules in `MDD_LABEL_SCHEMA.md` and `PHONEME_MAPPING_SPEC.md`. | **PASS** |
| **3. Justified Speaker-Independent Split** | Compare LOSO vs K-Fold vs Held-out. Account for 6 L1s, gender balance, Vietnamese cohort, and leakage. | Primary Balanced 4-Fold CV and Vietnamese LOSO-VN fully designed in `SPEAKER_SPLIT_PROTOCOL.md`. | **PASS** |
| **4. Comprehensive Metrics Specification** | Define Binary MDD, Substitution Diag-Acc, Deletion Recall, Insertion FAR, Model PER vs Corpus PER, and Scoring. | Formally defined with mathematical equations and confidence intervals in `EVALUATION_PROTOCOL.md`. | **PASS** |
| **5. Provenance-Preserving Data Contracts** | Design standardized example schema preserving raw archive hashes, timestamps, and interval alignments. | Fully specified in `DATA_CONTRACT_SPEC.md` with Pydantic validation models and JSON examples. | **PASS** |
| **6. Catalog of Open Decisions & Risks** | Identify every choice requiring operator sign-off. Formulate risk matrix with mitigations. | All 7 architectural decisions and 7 scientific risks cataloged in `OPEN_DECISIONS_AND_RISKS.md`. | **PASS** |
| **7. Strict Read-Only Governance** | No preprocessing executed, no splits frozen, no raw files mutated, no models trained. | 100% compliant. Phase strictly bounded as SPECIFICATION ONLY. | **PASS** |

---

## 5. Formal Gate Determination & Verdict

$$\mathbf{RQ1-STD-01: PASS}$$

### Final Engineering Determination
All data standardization specifications, phoneme mappings, audio preprocessing protocols, speaker-independent evaluation designs, and data contracts have been fully formulated and verified.

**MANDATORY SAFETY BOUNDARY:**
*This specification PASS does **NOT** authorize preprocessing execution, dataset splitting, or model fine-tuning. The execution environment remains strictly read-only. All implementation work remains halted pending explicit operator review and authorization.*
