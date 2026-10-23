# LANGFENS — RQ1-AUD-01: MDD Research Suitability & Feasibility Assessment

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase F — Research Dataset Suitability Assessment  
**Audit Timestamp (UTC):** 2026-10-22T20:40:00Z  
**Research Scope:** Phoneme-level Mispronunciation Detection & Diagnosis (MDD), Goodness of Pronunciation (GOP), Wav2Vec 2.0 Fine-Tuning, and Educational Pronunciation Feedback.  
**Execution Lead:** Senior Speech ML Research Engineer & Phonetics Researcher  

---

## 1. Executive Summary & Assessment Methodology

This assessment evaluates the empirical suitability of our audited datasets (**L2-ARCTIC v5.0** and **SpeechOcean762**) across the four planned research directions of Project Langfens.
Every feasibility rating is backed strictly by observed data evidence gathered during Phases B through E:
- No speculative assumptions are made regarding unavailable labels.
- Evaluation respects the separation between continuous pronunciation quality scoring and discrete phonological error diagnosis.
- Crucial scientific risks regarding class imbalance, speaker leakage, and L1 dialect bias are rigorously identified.

---

## 2. Research Suitability Matrix

| Research Direction | Target Task | Dataset | Verified Available Evidence | Uncovered Gaps & Constraints | Feasibility Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **A. GOP Scoring** | Acoustic posterior scoring & thresholding | **SpeechOcean762** | 5,000 utterances; 94,445 phones; 5-expert continuous scores ($0.0-2.0$); 250 balanced speakers; official Kaldi & GOPT baselines. | Phone time boundaries absent from `scores.json` (requires automated MFA alignment prior to feature extraction). | **FEASIBLE** |
| **A. GOP Scoring** | Likelihood ratio scoring against discrete errors | **L2-ARCTIC** | 3,599 gold TextGrids with acoustic boundaries; 26,867 forced alignments; 44.1 kHz audio. | No continuous human Likert scores; evaluation must rely on ROC-AUC / EER against binary error ground truth. | **CONDITIONAL** |
| **B. Wav2Vec2 CTC** | Acoustic representation fine-tuning for non-native phonemes | **L2-ARCTIC** | 26,867 utterances (~27 hours); 24 speakers across 6 L1s; canonical prompt transcripts; 3,599 gold realized sequences. | Pretrained Wav2Vec2 requires 16 kHz audio (requires non-destructive on-the-fly resampling from 44.1 kHz); automated alignments in non-gold subset (~23k utts). | **FEASIBLE** |
| **B. Wav2Vec2 CTC** | Non-native phoneme recognition fine-tuning | **SpeechOcean762** | 5,000 utterances at 16 kHz; 250 speakers; canonical phone sequences for all words. | Only Mandarin L1 speakers (no cross-lingual accent transfer); no realized phone sequence supervision. | **FEASIBLE** (Canonical CTC) / **NOT_SUPPORTED** (Error CTC) |
| **C. Error Diagnosis** | Phoneme substitution detection ($CPL \rightarrow PPL$) | **L2-ARCTIC** | 14,098 explicit substitution annotations; full phoneme inventory; 388 $DH \rightarrow D$, 257 $ER \rightarrow AH$ patterns. | Heavy class imbalance towards top 20 error pairs; 269 `err` tags require special token handling. | **FEASIBLE** |
| **C. Error Diagnosis** | Phoneme deletion detection ($CPL \rightarrow \emptyset$) | **L2-ARCTIC** | 3,420 explicit deletion events with exact silent interval timestamps. | 45.79% of all deletions are concentrated in Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`). | **FEASIBLE** |
| **C. Error Diagnosis** | Phoneme insertion / addition detection | **L2-ARCTIC** | 1,092 explicit insertion events over adjusted intervals. | Rare class ($<1\%$ of total tokens); high risk of false alarms. | **FEASIBLE** |
| **C. Error Diagnosis** | Explicit error diagnosis ($CPL \rightarrow PPL$) | **SpeechOcean762** | 5-expert bracket notes in `scores-detail.json`. | No standardized aligned diagnosis ground truth in `scores.json`; `mispronunciations` field is empty/absent. | **NOT_SUPPORTED** |
| **D. Educational CAPT** | Multi-aspect student report card feedback | **SpeechOcean762** | Utterance accuracy, completeness, fluency, prosody, and word stress scores. | Cannot diagnose specific phonetic articulatory mechanics (e.g. tongue height or voicing errors). | **FEASIBLE** |
| **D. Educational CAPT** | Actionable articulatory error feedback | **L2-ARCTIC** | Explicit substitution pairs mapped to phonological features (place, manner, voicing). | Lacks sentence-level fluency, prosody, and pedagogical rubric scores. | **CONDITIONAL** |

---

## 3. Deep Dive by Research Direction

### Direction A: GOP-Based Pronunciation Scoring
- **SpeechOcean762 is the ideal benchmark for Direction A.**
  - The dataset provides 5-expert consensus scores across multiple granularities.
  - The phone-level accuracy score ($0.0 - 2.0$) provides the direct target for Pearson Correlation Coefficient (PCC) and Mean Squared Error (MSE) regression benchmarks.
  - **Prerequisite for Execution:** Acoustic time intervals must be extracted using MFA (`english_us_arpa` v3.0.0) on the 16 kHz WAV files before computing acoustic frame posterior probabilities.
- **L2-ARCTIC provides conditional support for Direction A.**
  - Because human labels are categorical (`correct`, `sub`, `del`), GOP scores cannot be benchmarked via PCC correlation against human grades.
  - Instead, GOP scores must be evaluated as an anomaly detector (computing ROC-AUC, Precision-Recall AUC, and Equal Error Rate to classify correct vs mispronounced phones).

---

### Direction B: Wav2Vec 2.0 + Phoneme CTC Fine-Tuning
- **L2-ARCTIC is the primary training and evaluation corpus for Direction B.**
  - Peng et al. (2021) and Shahin et al. (2023) prove that fine-tuning `wav2vec2-base` or `wav2vec2-large` on L2-ARCTIC prevents catastrophic forgetting when using Connectionist Temporal Classification (CTC) over ARPAbet tokens.
  - **Acoustic Nuance:** L2-ARCTIC raw audio is sampled at 44.1 kHz. Pretrained Wav2Vec 2.0 models (e.g. `facebook/wav2vec2-base-960h`) strictly require single-channel 16 kHz input. Resampling must be integrated dynamically in dataloaders during future phases.
  - **Supervision Target Strategy:**
    1. *Stage 1 (Acoustic Adaptation):* Train on canonical phone sequences across the full ~27 hours (including forced-aligned sentences) to adapt encoder representations to accented acoustic features.
    2. *Stage 2 (MDD Fine-Tuning):* Fine-tune specifically on the 3,599 gold manual transcriptions using realized phone sequences (including substitutions and insertions).

---

### Direction C: Phoneme Error Diagnosis (Substitutions, Deletions, Insertions)
- **L2-ARCTIC is the only verified corpus supporting explicit phonetic diagnosis.**
  - Substitution diagnosis requires predicting the exact perceived phone $\hat{y}$ given canonical target $y$. With 14,098 verified substitution events, L2-ARCTIC offers rich empirical evidence across diverse phonetic environments.
  - Deletion diagnosis is fully supported because annotators explicitly tagged the silence intervals where omitted phones occurred (`CPL,sil,d`).
  - **SpeechOcean762 cannot be used for Direction C.** Our Phase D audit confirmed that OpenSLR SLR101 does not provide realized phone diagnosis ground truth in `scores.json`. Training a diagnostic classifier to output "DH substituted by D" using SpeechOcean762 is scientifically unsupported.

---

### Direction D: Educational Pronunciation Feedback
- **A Hybrid Paradigm is required for educational CAPT applications:**
  1. **Holistic Assessment Layer (from SpeechOcean762):** Predicts sentence-level fluency (0–10), prosody (0–10), completeness (0–10), and word stress (5/10) to give students an overall proficiency rubric.
  2. **Fine-Grained Diagnostic Layer (from L2-ARCTIC):** Detects specific phoneme errors (e.g. "You substituted /d/ for /ð/ in 'the'") and generates articulatory explanations (e.g. "Place the tip of your tongue between your teeth instead of against the alveolar ridge").

---

## 4. Methodological Risks & Splitting Protocol Directives

### 1. The Class Imbalance Risk
- **Observed Evidence:**
  - L2-ARCTIC: 85.30% of all tokens are Correct; only 11.83% are Substitutions, 2.87% Deletions, and 0.92% Additions.
  - SpeechOcean762: 80.68% of all phone scores are 2.0 (perfect); only 2.42% are 0.0.
- **Scientific Impact:** A naive model predicting "Correct" on every token achieves >85% accuracy while possessing zero diagnostic utility.
- **Evaluation Rule:** Future benchmark evaluation must report:
  - Phone Error Detection: Precision, Recall, and F1-score on the **Error (minority) class**.
  - Diagnosis: Hierarchical Error Rate (HER) and Substitution F1.
  - ROC-AUC and PR-AUC curves.

### 2. Speaker & L1 Leakage Risk
- **Observed Evidence:** L2-ARCTIC contains only 24 speakers (4 per L1). If utterances from the same speaker appear in both train and test splits, the model memorizes speaker-specific vocal tract acoustics rather than learning generalized phonological error representations.
- **Strict Protocol Directives for Future Splitting:**
  - **Speaker-Independent Partitioning:** No speaker may ever appear in more than one partition.
  - **L1-Balanced Cross-Validation:** Standard literature baselines (Peng et al., 2021; Tu et al., 2025) employ a 6-fold or 8-fold cross-validation protocol where test splits contain unseen speakers from each L1.
  - For Vietnamese-specific evaluation, a Leave-One-Speaker-Out (LOSO) scheme across `HQTV`, `PNV`, `THV`, and `TLV` must be utilized.

---

## 5. Acceptance Evaluation for Gate G5

| Criterion | Requirement | Finding | Status |
| :--- | :--- | :--- | :---: |
| **GOP Feasibility Evaluated** | Assessed against both corpora | Feasible on SO762; Conditional on L2 | **PASS** |
| **Wav2Vec2 Feasibility Evaluated** | Assessed for CTC & SSL adaptation | Feasible on L2 (16kHz requirement documented) | **PASS** |
| **Diagnosis Feasibility Evaluated** | Sub/Del/Add capabilities analyzed | Supported on L2; Not supported on SO762 | **PASS** |
| **Educational Feasibility Evaluated** | Rubric vs articulatory feedback | Hybrid recommendation formulated | **PASS** |
| **Risk Factors Audited** | Imbalance and leakage risks specified | Concrete evaluation metrics & protocols defined | **PASS** |

**Overall Gate G5 (MDD_FEASIBILITY_ASSESSED): PASS**
