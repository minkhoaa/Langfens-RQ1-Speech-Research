# LANGFENS — RQ1-STD-01: Data Standardization & Evaluation Policy

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Phase Status:** SPECIFICATION ONLY (Read-Only Mandate; No Preprocessing, Splitting, or Training Authorized)  
**Author / Engineering Lead:** Senior Speech ML Research Engineer, Phonetics Specialist & Research Reproducibility Engineer  
**Date of Ratification:** 2026-10-23  
**Target Benchmarks:** L2-ARCTIC Release v5.0 (3,599 Gold Praat TextGrids) & SpeechOcean762 (OpenSLR SLR101, 5,000 Scored Utterances)  
**Deliverable Directory:** `manifests/RQ1-STD-01/` (Synced to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`)  

---

## 1. Executive Summary & Standardization Mandate

Project Langfens focuses on phoneme-level Mispronunciation Detection and Diagnosis (MDD), Goodness of Pronunciation (GOP) scoring, and pedagogical feedback generation for second-language (L2) English learners, with particular emphasis on Vietnamese native speakers.

Following the forensic integrity audits (**RQ1-AUD-01**) and mathematical reconciliation (**RQ1-AUD-01R**), this policy document establishes the foundational principles, architectural boundaries, and operational guidelines governing all downstream data preparation (`RQ1-PREP-01`), speaker partitioning (`RQ1-SPLIT-01`), and acoustic model fine-tuning (`RQ1-TRN-01`).

### Core Findings Governing This Policy
1. **Corpus Role Specialization:**
   - **L2-ARCTIC v5.0** is the **sole gold-standard benchmark for explicit phoneme error diagnosis** (14,098 substitutions, 3,420 deletions, 1,092 insertions across 24 speakers and 6 L1 backgrounds).
   - **SpeechOcean762** is the **primary benchmark for multi-aspect continuous pronunciation scoring** (94,445 phones, 31,816 words, 5,000 sentences scored by a 5-expert consensus panel). It does **not** provide aligned diagnostic error targets and must not be used for substitution classification.
2. **Mathematical Denominator Separation:**
   - Token-level binary MDD operates strictly on canonical non-silence prompt tokens ($N_{\text{canonical}} = 119,151$).
   - Acoustic insertion events ($I = 1,092$) occur over silent intervals without canonical reference tokens; they must never be summed into binary classification denominators.
   - Phone Error Rate ($\text{PER} = 15.62\%$) is a sequence-level Levenshtein edit distance metric, strictly separated from the Canonical Error Rate ($14.70\%$).
3. **Vietnamese L1 Cohort Priority:**
   - The Vietnamese cohort (`HQTV`, `PNV`, `THV`, `TLV`) exhibits the highest error rate corpus-wide ($24.43\%$) and accounts for $45.79\%$ of all corpus deletions ($1,566 / 3,420$). Dedicated sensitivity protocols are mandatory.

---

## 2. The Five Pillars of Langfens Data Engineering

All data transformations, feature extraction pipelines, and evaluation harnesses must strictly comply with five invariant principles:

```text
The Five Invariant Principles of Project Langfens
├── 1. Immutable Raw Source of Truth (Zero Mutation of Gold Archives)
├── 2. Multi-Layer Non-Destructive Phoneme Representation
├── 3. Reversibility & Bit-Level Provenance Traceability
├── 4. Separation of Continuous Quality Scoring vs Categorical Diagnosis
└── 5. Leakage-Free Speaker-Independent Evaluation Architecture
```

### Principle 1: Immutable Raw Source of Truth
- The raw archives (`l2arctic_release_v5.0.zip` and `speechocean762.tar.gz`), reference lexicons (`cmudict-0.7b`, `english_us_arpa.dict`), and original Praat TextGrids remain completely untouched in read-only storage.
- No in-place text normalization, header manipulation, or file deletion is permitted.
- The 245 unmanifested extra WAV files in SpeechOcean762 (`SPEAKER9600`–`SPEAKER9649`) remain in raw archives to preserve archive integrity, but are excluded from model dataloaders via explicit manifest indexing.

### Principle 2: Multi-Layer Non-Destructive Phoneme Representation
Phonetic annotations exist in distinct abstraction layers to prevent information loss:
- **Layer 0 (Raw Ground Truth):** Preserves raw TextGrid strings, stress marks, annotator deviation tags (`*`), unknown tags (`err`), and Praat timing boundaries.
- **Layer 1 (Canonical 39-Phone ARPAbet):** Normalized, stress-free base phonemes used for canonical ASR, GOP forced alignment, and prompt sequence encoding.
- **Layer 2 (Lexical Stress Space — 69 Phones):** Full stress-bearing vowels (`0`, `1`, `2`) and consonants used for lexical stress evaluation on SpeechOcean762.
- **Layer 3 (Diagnostic CTC Space — 43 Classes):** 39 base phonemes + extended schwa (`AX`) + `<UNK_PHONE>` + `<SIL>` + `<BLANK>`, designed for non-native realized phoneme sequence decoding.

### Principle 3: Reversibility & Bit-Level Provenance Traceability
- Every processed training example, aligned frame, or evaluation token must maintain a bidirectional pointer back to its raw archive, speaker ID, utterance ID, tier name, interval index, and raw millisecond timestamp.
- Any label normalization (e.g., stripping trailing whitespace, normalizing cataloged typos) must be rule-based, deterministic, and logged in data manifests.

### Principle 4: Separation of Scoring vs Categorical Diagnosis
- **Continuous Quality Scoring (SpeechOcean762):** Modeled via regression or posterior expectation (GOP, GOPT Transformer). Evaluated using Pearson Correlation Coefficient (PCC), Spearman Rank Correlation ($\rho$), MAE, and MSE against expert consensus targets ($0.0 - 2.0$).
- **Categorical Error Diagnosis (L2-ARCTIC):** Modeled via multi-class classification or sequence transduction. Evaluated using Precision, Recall, F1, PR-AUC, and Confusion Matrices on explicit substitutions, deletions, and additions.

### Principle 5: Leakage-Free Speaker-Independent Architecture
- Acoustic models must never evaluate on speech from speakers present in their training partition.
- Cross-validation partitions must balance L1 backgrounds and biological sex.
- Decision threshold calibration (e.g. for GOP or binary posterior heads) must be executed strictly on validation speakers, never on test speakers.
- Repeated prompt texts across speakers must be explicitly monitored to prevent acoustic memorization.

---

## 3. Specification Deliverables Map

The complete standardization policy is partitioned into eight focused specifications:

| Deliverable File | Focus Area | Mandatory Contents & Scope |
| :--- | :--- | :--- |
| `STANDARDIZATION_POLICY.md` | Policy & Governance | Executive summary, core principles, document map, execution boundaries, and audit trail. |
| `MDD_LABEL_SCHEMA.md` | Label Semantics & Accounting | Complete schema taxonomy, resolution of `PPL*` contradiction, primary (strict) vs sensitivity (lenient) policies, denominators, insertion treatment, and typo normalization rules. |
| `PHONEME_MAPPING_SPEC.md` | Phonetic Inventories & Mappings | 39-phone core taxonomy, Layer 0–3 representations, `AX` vs `AH0` harmonization, IPA conversion tables, MFA v3.0 dictionary handling, and CTC blank/silence disambiguation. |
| `AUDIO_PREPARATION_SPEC.md` | Acoustic Preprocessing Standards | 44.1 kHz to 16 kHz resampling methods, polyphase filter parameters, timestamp invariance proofs, frame quantization math, WAV integrity constraints, and audio-annotation pairing checks. |
| `SPEAKER_SPLIT_PROTOCOL.md` | Partitioning Architecture | Evaluation of 24 L2-ARCTIC speakers, comparison of LOSO vs Balanced K-Fold vs Held-Out, primary 4-fold recommendation, Vietnamese LOSO sensitivity protocol, and leakage mitigation. |
| `EVALUATION_PROTOCOL.md` | Metrics & Benchmark Framework | Mathematical definitions for Binary MDD, Substitution Diagnosis, Deletion Detection, Insertion FAR, Model PER vs Corpus PER, Scoring Correlations, Macro/Micro aggregation, and 95% bootstrap CIs. |
| `DATA_CONTRACT_SPEC.md` | Dataloader Schema & Interfaces | Concrete JSON/Parquet schema for standardized examples, typing definitions, uncertainty flags, alignment intervals, provenance fields, and Pydantic validation contracts. |
| `OPEN_DECISIONS_AND_RISKS.md` | Risk Management & Sign-Off | Catalog of 7 architectural decisions requiring explicit human authorization, scientific risk matrix, mitigation controls, and formal gate sign-off block. |

---

## 4. Execution Governance & Gate Boundaries

```text
Current Project Milestone: RQ1-STD-01
├── State: SPECIFICATION RATIFIED & VERIFIED
├── Permitted Actions:
│   ├── Create and inspect specification markdown documents
│   ├── Run non-destructive read-only audit scripts and integrity verifications
│   └── Synchronize specification files to Google Drive (kltn_gdrive:)
└── Strictly Prohibited Actions (Require Explicit Operator Authorization):
    ├── Executing audio resampling or writing converted WAVs to disk
    ├── Modifying gold Praat TextGrids or raw archive files
    ├── Generating, freezing, or executing speaker train/test splits
    ├── Running Montreal Forced Aligner (MFA) alignment pipelines
    └── Instantiating PyTorch models, fine-tuning Wav2Vec2, or executing evaluations
```

### Transition to Subsequent Phase
Upon review and formal acceptance of the eight deliverables in `manifests/RQ1-STD-01/` by the lead operator, Project Langfens will transition to:
1. **Phase RQ1-SPLIT-01:** Deterministic generation and freezing of speaker-independent cross-validation manifests.
2. **Phase RQ1-PREP-01:** Implementation of verified PyTorch dataset loaders, dynamic 16 kHz audio resampling, and non-destructive interval parsers adhering strictly to these specifications.
