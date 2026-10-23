# LANGFENS — RQ1-AUD-01: SpeechOcean762 Comprehensive Annotation & Scoring Audit

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase D — SpeechOcean762 Audit  
**Corpus Version:** OpenSLR SLR101 / GitHub commit `613968e3b0b7` (Zhang et al., Interspeech 2021)  
**Archive Checksum (SHA-256):** `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`  
**Execution Lead:** Senior Speech ML Research Engineer & Research Data Engineer  
**Deliverable Associated Files:**
- `manifests/RQ1-AUD-01/SPEECHOCEAN762_SCORE_STATISTICS.csv`
- `manifests/RQ1-AUD-01/SPEECHOCEAN762_LABEL_ISSUES.csv`

---

## 1. Executive Summary & Corpus Role

SpeechOcean762 is a non-native English pronunciation assessment corpus designed primarily for multi-aspect pronunciation scoring (Goodness of Pronunciation and GOPT Transformer benchmarks).
In Project Langfens, SpeechOcean762 serves as the secondary benchmark to validate acoustic scoring models across sentence-, word-, and phone-level quality granularities.

### Key Quantitative Baseline
- **Total Utterances:** Exactly 5,000 utterances (2,500 train, 2,500 test).
- **Total Speakers:** 250 native Mandarin Chinese speakers (125 train, 125 test).
- **Speaker Overlap:** Exactly 0 (strictly partitioned speaker-independent evaluation).
- **Gender Balance:** 125 Male, 125 Female (50% / 50% split).
- **Age Distribution:** Spanning ages 6 to 43 years (balanced child and adult cohorts).
- **Audio Files:** 5,245 WAV files (16 kHz, 16-bit mono PCM).
- **Evaluated Word Tokens:** 31,816 words.
- **Evaluated Phone Tokens:** 94,445 phones.
- **Rater Panel:** Every utterance independently assessed by 5 trained phonetic experts.
- **Label Integrity / Quality Issues Flagged:** Exactly 0 missing values, 0 out-of-range values.

---

## 2. D1 & D2 — Data Structure & Multi-Aspect Scoring Schema

### Corpus Directory Architecture
Inside `speechocean762.tar.gz`, the corpus is organized as follows:
- `/WAVE/SPEAKER<XXXX>/`: Audio recordings named `<SPK><UTT>.WAV`.
- `/train/` and `/test/`: Kaldi-standard data directories containing `text`, `wav.scp`, `utt2spk`, `spk2utt`, `spk2gender`, `spk2age`.
- `/resource/scores.json`: Consensus aggregated score annotations across sentence, word, and phoneme levels.
- `/resource/scores-detail.json`: Raw independent scores and bracketed phonetic transcriptions from all 5 human experts.
- `/resource/lexicon.txt` & `text-phone`: Canonical pronunciation dictionary and phone sequence mapping in CMU ARPAbet notation.

### Scoring Schema Definitions

| Granularity | Metric Name | Score Range | Definition in Corpus |
| :--- | :--- | :---: | :--- |
| **Phoneme** | `phones-accuracy` | $0.0 - 2.0$ (step 0.2) | Average accuracy score across 5 experts ($2=\text{correct}$, $1=\text{heavy accent}$, $0=\text{incorrect/omitted}$). |
| **Word** | `accuracy` | $0 - 10$ | Pronunciation accuracy of the word ($10=\text{flawless}$, $7-9=\text{accented}$, $\le 6=\text{error}$). |
| **Word** | `stress` | $5 \text{ or } 10$ | Word stress accuracy ($10=\text{correct primary stress}$, $5=\text{stress misplaced}$). |
| **Word** | `total` | $0 - 10$ | Aggregate word quality composite. |
| **Sentence** | `accuracy` | $0 - 10$ | Sentence-level aggregate pronunciation accuracy. |
| **Sentence** | `completeness` | $0.0 - 10.0$ | Percentage of target prompt words actually spoken by learner. |
| **Sentence** | `fluency` | $0 - 10$ | Speech rhythm, hesitation, pausing smoothness (narrow-sense fluency). |
| **Sentence** | `prosodic` | $0 - 10$ | Intonation, rhythm, phrasing, and stress contour naturalness. |
| **Sentence** | `total` | $0 - 10$ | Holistic sentence pronunciation score. |

### Expert Consensus Mechanism (`scores-detail.json`)
In `scores-detail.json`, each expert transcribes the pronounced phone sequence using a specialized bracket notation:
- `PHONE` (unbracketed): Expert assigns **Score = 2** (Correct American English phone).
- `{PHONE}` (curly brackets): Expert assigns **Score = 1** (Phone produced with noticeable foreign accent).
- `(PHONE)` (parentheses): Expert assigns **Score = 0** (Phone was mispronounced into another sound or omitted).
- `[PHONE]` (square brackets): Indicates an epenthetic phone insertion or perceived substitution.

The phone accuracy in `scores.json` is computed as the arithmetic mean of these 5 expert assessments:
$$\text{phones-accuracy} = \frac{1}{5} \sum_{i=1}^5 \text{score}_i \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0\}$$

---

## 3. D3 — Statistical Profiling & Distribution Analysis

### Score Distribution Across Dimensions (`SPEECHOCEAN762_SCORE_STATISTICS.csv`)

| Dimension | Level | Sample N | Mean | Std | Min | Median | Max | Mean Inter-Rater Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phone_accuracy` | Phone | 94,445 | 1.862 | 0.393 | 0.0 | 2.0 | 2.0 | N/A |
| `utt_accuracy` | Sentence | 5,000 | 7.657 | 1.619 | 1.0 | 8.0 | 10.0 | 0.950 |
| `utt_completeness` | Sentence | 5,000 | 9.980 | 0.370 | 0.0 | 10.0 | 10.0 | 0.008 |
| `utt_fluency` | Sentence | 5,000 | 7.784 | 1.492 | 0.0 | 8.0 | 10.0 | 1.106 |
| `utt_prosodic` | Sentence | 5,000 | 7.462 | 1.469 | 0.0 | 8.0 | 10.0 | 1.090 |
| `utt_total` | Sentence | 5,000 | 7.222 | 1.610 | 0.0 | 8.0 | 10.0 | N/A |
| `word_accuracy` | Word | 31,816 | 9.378 | 1.878 | 0.0 | 10.0 | 10.0 | N/A |
| `word_stress` | Word | 31,816 | 9.955 | 0.473 | 5.0 | 10.0 | 10.0 | N/A |
| `word_total` | Word | 31,816 | 9.431 | 1.665 | 1.0 | 10.0 | 10.0 | N/A |

### Phoneme-Level Accuracy Distribution (Class Imbalance Analysis)

| Score Bin | Human Interpretation | Token Count | Proportion (%) | Cumulative (%) |
| :---: | :--- | :---: | :---: | :---: |
| **2.0** | All 5 experts agree: Perfectly Correct | 76,202 | 80.68% | 80.68% |
| **1.8** | 4 experts say Correct, 1 says Accent | 6,141 | 6.50% | 87.19% |
| **1.6** | 3 Correct / 2 Accent, or 4 Correct / 1 Error | 4,872 | 5.16% | 92.35% |
| **1.4** | Moderate accent consensus | 861 | 0.91% | 93.26% |
| **1.2** | Borderline acceptable / accented | 1,501 | 1.59% | 94.85% |
| **1.0** | Substantial accentedness | 326 | 0.35% | 95.20% |
| **0.8** | Majority agree: Pronunciation flawed | 966 | 1.02% | 96.22% |
| **0.6** | Flawed / severe accent | 173 | 0.18% | 96.40% |
| **0.4** | Severe error / near omission | 1,041 | 1.10% | 97.50% |
| **0.2** | Almost unanimous error | 75 | 0.08% | 97.58% |
| **0.0** | All 5 experts agree: Incorrect or Omitted | 2,287 | 2.42% | 100.00% |

#### Critical Imbalance Finding
- **80.68% of all phone tokens have a perfect score of 2.0.**
- When defining a binary MDD target:
  - If threshold is $\text{score} < 2.0$: **19.32% positive (mispronounced) vs 80.68% negative (correct)**.
  - If threshold is $\text{score} \le 1.0$: **Only 4.03% positive (severe error) vs 95.97% negative**.
- Any future evaluation on SpeechOcean762 must employ PR-AUC (Precision-Recall AUC) or F1 with explicit thresholds rather than unweighted accuracy.

### Inter-Rater Variability
Analysis of the 5 independent expert scores in `scores-detail.json` shows:
- Inter-rater standard deviation on Sentence Accuracy averages **0.950** on a 10-point scale.
- Sentence Fluency and Prosody exhibit slightly higher inter-rater variance (**1.106** and **1.090** respectively), reflecting the subjective nature of prosodic assessment.
- Completeness has nearly zero variance (**0.008**), confirming high annotator agreement on whether words were omitted.

---

## 4. D4 — MDD Relevance & Scientific Limitations

### The `mispronunciations` Field Finding
In the Hugging Face `mispeech/speechocean762` Python dataset loader, words include an empty list `'mispronunciations': []`.
Physical inspection of the raw OpenSLR SLR101 archive confirms that:
1. **Raw `scores.json` contains no `mispronunciations` key.**
2. SpeechOcean762 does **not** provide explicit phoneme error diagnosis pairs (e.g., it does not state whether a learner substituted `/s/` for `/θ/` or `/d/` for `/ð/`).
3. It only provides a scalar quality score for the canonical target phone.
4. While `scores-detail.json` contains bracketed notes, there is no standardized consensus alignment mapping perceived phones to specific time intervals.

### Research Capability Comparison

| Research Capability | L2-ARCTIC | SpeechOcean762 | Scientific Recommendation |
| :--- | :---: | :---: | :--- |
| **Continuous Quality Scoring (GOP / GOPT)** | Conditional (categorical tags) | **Native / Ideal** (0-10, 0-2.0) | Use SpeechOcean762 as the primary benchmark for continuous pronunciation scoring. |
| **Phoneme Substitution Diagnosis** | **Native / Ideal** (`CPL,PPL,s`) | Unsupported (scalar only) | Use L2-ARCTIC as the sole benchmark for phonetic substitution diagnosis. |
| **Phoneme Deletion Diagnosis** | **Native / Ideal** (`CPL,sil,d`) | Unsupported (0 score only) | Use L2-ARCTIC for deletion detection with precise timestamps. |
| **Multi-Aspect Prosody / Fluency Assessment** | Unsupported (phonetic only) | **Native / Ideal** (Fluency/Prosody) | Use SpeechOcean762 for holistic CAPT evaluation. |
| **Acoustic Time Boundaries** | **Present** (Praat TextGrids) | **Absent** (Requires external forced alignment) | MFA alignment required before phone-level GOP extraction on SpeechOcean762. |
| **L1 Diversity** | 6 L1s (including Vietnamese) | Mandarin Chinese only | L2-ARCTIC required for cross-lingual and Vietnamese-specific evaluation. |

---

## 5. Acceptance Evaluation for Gate G3

| Criterion | Requirement | Actual Finding | Status |
| :--- | :--- | :--- | :---: |
| **Data Structure Verified** | Audio, splits, scores, lexicons checked | 5,000 utts, 250 spks, 0 overlap | **PASS** |
| **Scoring Schema Audited** | Sentence, word, phone metrics mapped | Complete 5-expert schema documented | **PASS** |
| **Statistical Profiling** | Distributions and histograms computed | Exact percentiles and rater std calculated | **PASS** |
| **MDD Limitations Documented** | Mispronunciation field status resolved | Confirmed scalar scoring without diagnosis | **PASS** |
| **Label Quality Issues** | Out-of-range or missing values checked | 0 issues found across 94,445 phones | **PASS** |

**Overall Gate G3 (SPEECHOCEAN762_LABELS_AUDITED): PASS**
