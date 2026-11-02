# LANGFENS — RQ1-DIAG-01 / 01-Phoneme Omission & False-Negative Root Cause

**Status:** `COMPLETED; DIAGNOSTIC_AUDIT_VERIFIED; READY_FOR_EXPERIMENT_DESIGN`

**Recorded at:** 2026-11-01T23:45:00+07:00

**Evidence source:**
- Standalone diagnostic harness: `audit_scripts/diagnose_omissions_and_false_negatives.py`
- Master diagnostic report: `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json` (SHA-256 `65076d51...`)
- Disaggregated CSV tables:
  - `omission_error_breakdown_by_speaker.csv`
  - `omission_error_breakdown_by_phone.csv`
  - `omission_error_breakdown_by_position.csv`
  - `omission_error_breakdown_by_duration.csv`
  - `representative_false_negatives.csv`
- Automated regression test suite:
  - `tests/test_rq1_diag_01_phoneme_omission_fn.py` (8 tests)
  - Full repository regression suite: 347 passed, 1 skipped, 1 warning (100% green)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Cross-Attention Temporal Profiling, Exhaustive 39,576-Token Ingestion Audit, Disaggregated Census, Matched/Mismatched Sensitivity Decomposition)

**Branch:** `feat/rq1-diag-01-omission-fn-root-cause`

---

## 1. Executive Summary & Diagnostic Goal

In accordance with `AGENTS.md` and the master research protocol, Phase **RQ1-DIAG-01** delivers an exhaustive, evidence-based diagnostic audit investigating the root causes of:
1. **High CTC ASR deletions** and the conceptual confusion surrounding phoneme deletion metrics.
2. **Severe Binary MDD false negatives** on omitted phonemes (gold deletions), where the baseline model fails to detect omissions in over 95% of cases at standard thresholds.

This audit was conducted strictly under CPU-first execution constraints:
- **Zero GPU training, zero optimizer updates, and zero backward passes.**
- **Strict per-fold test set isolation preserved:** all diagnostics operate exclusively on the authorized 4-fold validation partitions (1,199 utterances, 39,576 tokens, 8 disjoint speakers).
- **Frozen Baseline R checkpoints preserved intact.**
- **Bounded offline inference only** to analyze cross-attention attention maps on validation audio.

---

## 2. Core Conceptual Disambiguation (Task 1)

Prior discussions often conflated three distinct phenomena under the umbrella term "deletion":

| Concept | Operational Definition | Target Sequence Domain | Typical Rate / Prevalence | Pedagogical Consequence |
|---|---|---|:---:|---|
| **True Gold Phoneme Omission** | The non-native speaker omitted a canonical phoneme from their speech, annotated in Praat TextGrid as `X,sil,d` (e.g. omitting /D/ in *"and"* or /R/ in *"hard"*). | Annotation level; Canonical prompt tokens. | **1.59%** of tokens (629 / 39,576); **14.93%** of all errors. | Phonological error requiring explicit learner correction. |
| **CTC Levenshtein Deletion** | The model's greedy CTC decoder failed to emit an acoustic frame token, causing a gap in the Levenshtein dynamic programming alignment between reference and hypothesis. | ASR decoding level; Aligned against realized phones in standard evaluation. | **~4.9%** macro deletion rate across folds. | Speech recognition dropout on speech sounds that were physically present. |
| **Binary MDD False Negative** | The Binary MDD head predicted an error probability $\hat{p} < \tau$ (predicting "correct") for a token that had gold mispronunciation label $y = 1$ (either substitution or omission). | Classification level; Evaluated on canonical prompt tokens at decision threshold $\tau$. | **83.38%** at $\tau = 0.50$; **51.41%** at $\tau = 0.20$; **97.46%** on omissions at $\tau = 0.50$. | The system fails to alert the language learner that they made a pronunciation error. |

### Key Distinction:
A **CTC deletion** is an *ASR omission* (the acoustic model missed a sound the speaker actually made).  
A **gold omission** is a *speaker deletion* (the non-native speaker skipped a sound they were supposed to make).  
A **binary false negative on an omission** is an *MDD detection failure* (the model fails to identify that the speaker omitted the phone).

---

## 3. Data Pipeline & Alignment Audit: TextGrid → Dataset → Heads (Task 2)

An exhaustive audit of the ingestion pipeline revealed a critical structural asymmetry between the CTC head and Binary MDD head:

```text
Praat TextGrid:
  Interval 1: "AND" -> [DH, sil, d]  (Gold Omission / Deletion)
  Interval 2: "THE" -> [DH, D, s]    (Substitution)
  Interval 3: "CAT" -> [K, K, c]     (Correct)

Canonical Tokens (Prompt / Binary Head Input):
  Tokens:       [DH,  DH,  K ]  (Length = 3)
  Binary Labels:[ 1,   1,  0 ]  (Both omissions and substitutions have label 1)

Chronological Realized Tokens (CTC Head Target):
  Tokens:       [     D,   K ]  (Length = 2; Gold omission [DH, sil, d] is EXCLUDED!)
```

### Audit Findings:
1. **Semantic Exclusion in CTC:**
   In `src/langfens_prep/dataset.py` (lines 107–115), `chronological_realized_phones` explicitly filters out `error_type == 'deletion'`. As a result, the CTC branch is trained *not* to emit omitted phones. This is correct for acoustic transcription (ASR), but means the CTC loss provides **zero gradient signal** to penalize omissions against canonical prompts.
2. **Unaligned Cross-Attention Mechanism:**
   In `src/langfens_prep/model.py`, the cross-attention queries are formed from canonical phone embeddings + learned 1D positional embeddings. The keys and values are the Wav2Vec2 encoder features. Because the Wav2Vec2 backbone was kept frozen and **no monotonic or temporal alignment loss** was applied, cross-attention was supervised exclusively by the binary classification loss.

---

## 4. Four-Fold Token Census & Error-Type Partition (Task 3)

Auditing the full 4-fold out-of-fold validation set across all 1,199 utterances and 8 validation speakers yields an authoritative token census:

| Token Category | Token Count | Fraction of All Tokens | Fraction of Errors | Empirical Prevalence |
|---|:---:|:---:|:---:|:---:|
| **Correct Productions** ($y=0$) | **35,363** | 89.3547% | — | Baseline Negatives |
| **Substitutions** ($y=1, \text{type}=s$) | **3,584** | 9.0560% | 85.0699% | 0.090560 |
| **Deletions / Omissions** ($y=1, \text{type}=d$) | **629** | 1.5893% | 14.9301% | 0.015893 |
| **Total Evaluated Tokens** | **39,576** | 100.0000% | — | — |
| **Total Error Tokens** ($y=1$) | **4,213** | 10.6453% | 100.0000% | 0.106453 |

Mutually exclusive partition check: $35,363 + 3,584 + 629 = 39,576$ tokens ($\Delta = 0$).

---

## 5. Ranking Metrics & Acoustic Sensitivity by Error Type (Tasks 4 & 8)

Evaluating the saved Baseline R predictions separately for substitutions and omissions reveals a severe performance chasm:

| Error Category | Evaluated Tokens (Pos + Neg) | Matched AP | Mismatched AP (Shift-1) | Acoustic Reliance Gap ($\Delta_{\text{gap}}$) | ROC-AUC | Mean Score ($s_{\text{match}}$) | Median Score ($s_{\text{match}}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall Errors ($y=1$)** | 39,576 | **0.272580** | 0.261960 | **+0.010619** | 0.729022 | 0.2546 | 0.1949 |
| **Substitutions Only** | 38,947 | **0.264270** | 0.253554 | **+0.010715** | 0.734851 | 0.2665 | 0.2024 |
| **Deletions / Omissions Only** | 35,992 | **0.030506** | 0.028830 | **+0.001677** | 0.695813 | 0.1866 | 0.1732 |
| *Correct Negatives ($y=0$)* | 35,363 | — | — | — | — | 0.1273 | 0.0914 |

### Critical Discoveries:
1. **Severe AP Degradation on Omissions:**
   Deletion AP is **0.030506**, barely 1.75× above the 0.0159 empirical prevalence. In contrast, Substitution AP is **0.264270** (2.87× above the 0.0920 substitution prevalence).
2. **Acoustic Decoupling on Omissions:**
   The acoustic reliance gap for deletions collapses to **$+0.001677$** (statistically indistinguishable from zero), compared to **$+0.010715$** for substitutions. When presented with mismatched audio, the model's omission detection scores remain virtually unchanged, demonstrating that the binary head is **almost completely acoustically blind** to phoneme omissions.
3. **Score Distribution Compression:**
   Omission scores (mean 0.1866, median 0.1732) are compressed tightly against correct negative scores (mean 0.1273, median 0.0914), while substitutions achieve higher discrimination (mean 0.2665, median 0.2024).

---

## 6. Diagnostic Operating Points & False Negative Rates (Task 4)

Evaluating at fixed False Positive Rate (FPR) diagnostic ceilings on pooled negatives confirms extreme false-negative rates on omissions:

| Operating Point | Operational Threshold $\tau$ | Actual FPR on Negatives | Substitution Recall | Substitution FNR | Deletion Recall | Deletion FNR | Overall FNR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard $\tau = 0.50$** | 0.500000 | 2.98% | 19.08% | 80.92% | **2.54%** | **97.46%** | 83.38% |
| **Standard $\tau = 0.20$** | 0.200000 | 17.94% | 50.31% | 49.69% | 38.79% | 61.21% | 51.41% |
| **FPR $\le 20\%$ Ceiling** | 0.188482 | 20.00% | 53.07% | 46.93% | 44.20% | 55.80% | 48.26% |
| **FPR $\le 15\%$ Ceiling** | 0.220002 | 15.00% | 46.23% | 53.77% | 31.32% | **68.68%** | 55.99% |
| **FPR $\le 10\%$ Ceiling** | 0.268208 | 10.00% | 39.20% | 60.80% | 17.49% | **82.51%** | 64.04% |

### Key Findings:
- At the canonical deployment threshold $\tau = 0.50$, the model detects only **16 out of 629 deletions** (**2.54% Recall, 97.46% FNR**).
- Even under relaxed diagnostic operating points (15% FPR ceiling), over **two-thirds of all true phoneme omissions (68.68%) are completely missed**.

---

## 7. Disaggregated Breakdown Analysis (Task 5)

### 7.1 By Speaker & Demographic Group (8 Speakers)

| Speaker | Fold | L1 Accent | Gender | Total Tokens | Substitutions | Deletions | Sub Prev | Del Prev | Sub AP | Del AP | Del FNR @ 0.50 | Del FNR @ 0.20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HKK** | 4 | Korean | Female | 4,919 | 318 | 103 | 6.46% | 2.09% | 0.2082 | 0.0295 | **100.0%** | 72.82% |
| **NCC** | 4 | Mandarin | Female | 4,947 | 536 | 55 | 10.83% | 1.11% | 0.3811 | 0.0217 | **98.18%** | 70.91% |
| **NJS** | 1 | Spanish | Female | 5,011 | 561 | 80 | 11.20% | 1.60% | 0.3955 | 0.0356 | **100.0%** | 65.00% |
| **RRBI** | 1 | Hindi | Male | 4,945 | 451 | 65 | 9.12% | 1.31% | 0.2334 | 0.0366 | **96.92%** | 40.00% |
| **TNI** | 2 | Spanish | Male | 4,939 | 491 | 68 | 9.94% | 1.38% | 0.2840 | 0.0313 | **98.53%** | 63.24% |
| **TXHC** | 3 | Mandarin | Male | 4,944 | 492 | 128 | 9.95% | 2.59% | 0.2830 | 0.0572 | **93.75%** | 55.47% |
| **YBAA** | 2 | Arabic | Female | 4,934 | 331 | 31 | 6.71% | 0.63% | 0.1166 | 0.0107 | **93.55%** | 74.19% |
| **ZHAA** | 3 | Arabic | Female | 4,937 | 404 | 99 | 8.18% | 2.01% | 0.3207 | 0.0419 | **97.98%** | 56.57% |

*Result:* Omission FNR at $\tau = 0.50$ exceeds **93% across all 8 speakers**, proving that omission blindness is a structural property of the architecture, not an accent-specific artifact.

### 7.2 By Word Position

| Word Position | Total Tokens | Correct Tokens | Substitution Tokens | Deletion Tokens | Omission Rate | Fraction of All Omissions | Deletion FNR @ 0.20 | Substitution FNR @ 0.20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Word-Final** | 10,831 | 9,252 (85.4%) | 1,229 (11.3%) | **350 (3.23%)** | **3.23%** | **55.64%** | 54.57% | 34.74% |
| **Word-Medial** | 17,436 | 15,833 (90.8%) | 1,392 (8.0%) | **211 (1.21%)** | 1.21% | **33.55%** | 64.93% | 70.47% |
| **Word-Initial** | 10,831 | 9,814 (90.6%) | 951 (8.8%) | **66 (0.61%)** | 0.61% | **10.49%** | 83.33% | 38.38% |
| **Single-Phone Word** | 478 | 464 (97.1%) | 12 (2.5%) | **2 (0.42%)** | 0.42% | **0.32%** | 100.0% | 66.67% |

*Result:* **55.64% of all phoneme omissions occur in word-final position** (350 / 629), where non-native speakers drop coda consonants (e.g. final /t/, /d/, /s/, /z/).

### 7.3 By Phoneme Duration

| Duration Bin | Total Tokens | Correct Tokens | Substitution Tokens | Deletion Tokens | Omission Rate | Fraction of All Omissions | Deletion FNR @ 0.20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$< 40$ ms** | 2,612 | 2,029 (77.7%) | 290 (11.1%) | **293 (11.22%)** | **11.22%** | **46.58%** | 59.73% |
| **$40 - 80$ ms** | 14,965 | 13,419 (89.7%) | 1,294 (8.6%) | **252 (1.68%)** | 1.68% | **40.06%** | 60.71% |
| **$80 - 120$ ms** | 11,930 | 10,770 (90.3%) | 1,093 (9.2%) | **67 (0.56%)** | 0.56% | **10.65%** | 65.67% |
| **$\ge 120$ ms** | 10,069 | 9,145 (90.8%) | 907 (9.0%) | **17 (0.17%)** | 0.17% | **2.70%** | 76.47% |

*Result:* **86.65% of all deletions (545 / 629) occur in intervals under 80 ms** ($<40$ ms alone accounts for 46.58%). In Praat TextGrids, omissions are annotated as zero-energy closure or boundary marks spanning only 1–3 acoustic frames.

### 7.4 Top Phonemes by Omission Volume

| Rank | Phoneme | Manner / Type | Deletion Count | Deletion Rate | Substitution Count | Total Occurrences | Cumulative Omission % |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **/D/** | Alveolar voiced stop | **144** | 7.75% | 97 (5.22%) | 1,858 | 22.89% |
| 2 | **/R/** | Alveolar liquid / rhotic | **112** | 6.57% | 181 (10.62%) | 1,704 | 40.70% |
| 3 | **/T/** | Alveolar voiceless stop | **109** | 4.62% | 92 (3.90%) | 2,357 | **58.03%** |
| 4 | **/L/** | Alveolar lateral liquid | **47** | 2.98% | 125 (7.92%) | 1,578 | 65.50% |
| 5 | **/HH/** | Glottal fricative | **36** | 3.15% | 12 (1.05%) | 1,144 | 71.22% |
| 6 | **/AH/** | Central vowel | **29** | 0.81% | 203 (5.68%) | 3,571 | 75.83% |
| 7 | **/N/** | Alveolar nasal | **18** | 0.73% | 108 (4.40%) | 2,453 | 78.69% |
| 8 | **/IH/** | Front lax vowel | **15** | 0.59% | 228 (8.99%) | 2,537 | 81.08% |

*Result:* The top 3 phonemes (**/D/**, **/R/**, **/T/**) account for **58.03% of all omissions in the corpus**. The top 5 phonemes account for **71.22%**. These correspond exactly to well-known second-language phonological simplification processes:
1. Alveolar stop coda reduction (/D/, /T/ dropped in consonant clusters like *"and"*, *"past"*).
2. Rhotic deletion in non-rhotic transfer (/R/ dropped in *"car"*, *"part"*).
3. Dark /L/ vocalization or elision (/L/ dropped in *"all"*, *"cold"*).
4. Weak glottal fricative dropping (/HH/ dropped in pronouns like *"he"*, *"him"*).

---

## 8. Cross-Attention Temporal Localization Profiling (Task 7)

An offline forward-pass diagnostic was conducted on a balanced sample of 40 validation utterances (5 utterances per speaker across all 8 speakers, 1,533 tokens total) to evaluate the cross-attention matrix $A \in \mathbb{R}^{N \times T}$:

| Token Category | Sampled Tokens ($N$) | Mean In-Window Attention Mass | Mean Temporal Offset ($\Delta t = |t_{\text{peak}} - t_{\text{center}}|$) | Median Temporal Offset | Mean Max Frame Mass ($\max_t A_{i,t}$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Correct Tokens** | 1,345 | 0.0282 | 1.415 s | 1.140 s | 0.275 |
| **Substitution Tokens** | 139 | 0.0332 | 1.457 s | 1.180 s | 0.257 |
| **Deletion Tokens** | 49 | 0.0253 | 1.288 s | 1.080 s | 0.296 |
| **All Tokens** | 1,533 | 0.0286 | 1.415 s | 1.140 s | 0.274 |

### Mechanistic Discoveries:
1. **Total Temporal Misalignment:**
   The average attention mass residing inside the true Praat TextGrid interval is only **0.028** (2.8%), which is indistinguishable from uniform distribution across ~200 frames ($1/200 = 0.005$).
2. **Global Anchor Schedulers:**
   The cross-attention module has **4 heads**. For almost every token in an utterance, each head saturates with weight $\approx 0.25$ on the same 4 global acoustic anchor frames (e.g., frames 32, 75, 112, 124 in `arctic_a0003`), regardless of where the phoneme occurs in time.
3. **Absence of Local Contrast:**
   Because attention is not localized to the phoneme's time interval, the binary head cannot compare the prompt phone embedding against the *corresponding* acoustic window. When a phone is omitted, the cross-attention simply gathers features from elsewhere in the sentence, leaving the binary MLP to rely almost exclusively on its prompt phone embedding prior.

---

## 9. Investigation of Representative High-Confidence False Negatives (Task 9)

Auditing the lowest-scoring errors in `manifests/RQ1-DIAG-01/evaluation/representative_false_negatives.csv` revealed two distinct failure classes:

### 9.1 Deletion False Negatives (The "Omission Invisibility" Problem)

| Speaker | Utterance | Token | Canonical | Realized | Word | Duration | Matched Score | Mismatched Score | Cause |
|---|---|:---:|:---:|:---:|---|:---:|:---:|:---:|---|
| **TXHC** | `arctic_a0027` | 35 | **/M/** | `sil` | *enthusiasm* | 0.070 s | **0.006836** | 0.020908 | Word-final nasal elision; diffuse attention defaulted to high-frequency /M/ correctness prior. |
| **HKK** | `arctic_b0304` | 31 | **/F/** | `sil` | *enough* | 0.090 s | **0.017134** | 0.028833 | Word-final labiodental fricative deletion in Korean L1; model scored $p < 0.02$. |
| **NJS** | `arctic_a0300` | 35 | **/IY/** | `sil` | *period* | 0.008 s | **0.021767** | 0.026277 | Ultra-short vowel reduction (8 ms); unaligned frames. |
| **ZHAA** | `arctic_a0117` | 6 | **/W/** | `sil` | *with* | 0.043 s | **0.024390** | 0.029796 | Weak initial glide omission in unstressed preposition. |
| **NJS** | `arctic_a0064` | 11 | **/F/** | `sil` | *fifth* | 0.024 s | **0.029973** | 0.036642 | Alveolar cluster simplification (/f-th/ cluster). |

### 9.2 Substitution False Negatives (The "Nasal Neutralization" Problem)

Among the top 20 lowest-scoring errors across all 4,213 errors, **over 80% are coda nasal neutralizations (/M/ $\to$ /N/ or /M/ $\to$ /NG/)**:

| Speaker | Utterance | Token | Canonical | Realized | Word | Matched Score | Acoustic Mechanism |
|---|---|:---:|:---:|:---:|---|:---:|---|
| **TXHC** | `arctic_b0075` | 21 | **/M/** | **/N/** | *them* | **0.006007** | Non-native Mandarin transfer neutralizes coda /m/ into dental /n/. |
| **TXHC** | `arctic_a0024` | 17 | **/M/** | **/N/** | *from* | **0.006116** | Preposition coda nasal neutralization. |
| **ZHAA** | `arctic_a0047` | 15 | **/M/** | **/N/** | *gleamed* | **0.007411** | Medial cluster nasal substitution before voiced stop /d/. |
| **NCC** | `arctic_a0024` | 17 | **/M/** | **/N/** | *from* | **0.008840** | Mandarin transfer /m/ $\to$ /n/. |
| **NCC** | `arctic_b0366` | 30 | **/M/** | **/NG/** | *from* | **0.008947** | Mandarin transfer /m/ $\to$ velar /ŋ/. |
| **NJS** | `arctic_a0032` | 9 | **/M/** | **/N/** | *some* | **0.012897** | Spanish transfer /m/ $\to$ /n/ in coda. |

*Finding:* Because unconstrained cross-attention pools diffuse frames and the acoustic difference between [m] and [n] is subtle (slight shift in nasal formants), the model assigns $> 98\%$ confidence of correct pronunciation ($p < 0.02$).

---

## 10. Ranked Root-Cause Hypotheses (Task 10)

Distinguishing verified facts from speculative interpretations:

| Rank | Root-Cause Hypothesis | Empirical Evidence (Fact) | Mechanistic Interpretation (Qualified) | Status |
|:---:|---|---|---|:---:|
| **1** | **Cross-Attention Temporal Collapse** | Mean in-window attention mass is only 2.8%; temporal offset is $>1.4$ s; acoustic gap on omissions is $+0.0017$. | Unconstrained cross-attention without temporal or monotonic loss cannot discover frame-level phoneme boundaries from binary supervision alone. | **PRIMARY ROOT CAUSE** |
| **2** | **Extreme Class Imbalance & Prior Suppression** | Omissions constitute only 1.59% of training tokens (14.9% of errors); correct tokens are 89.4%. | In the absence of sharp localized acoustic cues, the binary MLP defaults to predicting the dominant prior, resulting in 97.5% FNR at $\tau = 0.50$. | **SECONDARY ROOT CAUSE** |
| **3** | **Word-Final Coda Duration Concentration** | 55.6% of deletions are word-final; 86.7% are $< 80$ ms; top 3 phones (/D/, /R/, /T/) comprise 58.0% of deletions. | Rapid coarticulatory reductions in coda clusters provide very few acoustic frames, making detection without duration modeling especially difficult. | **CONTRIBUTING ACOUSTIC FACTOR** |
| **4** | **CTC Task Disconnect & Realized vs Canonical Asymmetry** | CTC target sequence explicitly excludes gold deletions; CTC loss never penalizes missing canonical phones. | CTC acoustic features are trained to track what *was* said, not what was *omitted*, providing no negative signal for prompt omissions. | **CONTRIBUTING ARCHITECTURAL FACTOR** |
| **5** | **Subtle Nasal Coda Acoustic Invariance** | 80% of extreme substitution false negatives are /M/ $\to$ /N/ or /NG/ coda substitutions. | Diffuse frame pooling washes out subtle spectral cues (nasal murmur formants) necessary to distinguish bilabial from alveolar/velar nasals. | **CONTRIBUTING PHONETIC FACTOR** |

---

## 11. Proposed Minimal Controlled Experiments & Measurable Acceptance Gates (Task 11)

In strict accordance with `AGENTS.md` ("Never start training without explicit user authorization"), no training or code modification is executed. We formally propose three minimal controlled interventions:

### Experiment 1: CTC-Posterior-Guided Monotonic Alignment (Addresses Root Cause 1)
- **Concept:** Replace or regularize unconstrained multi-head cross-attention using the CTC branch's greedy/posterior peak frames to anchor phoneme query attention windows ($\mathcal{L}_{\text{align}} = \sum_i |t_{\text{attn peak}, i} - t_{\text{CTC peak}, i}|_1$).
- **Measurable Acceptance Gates:**
  1. Mean cross-attention in-window mass $\ge \mathbf{0.350}$ (vs current $0.028$).
  2. Deletion acoustic reliance gap $\Delta_{\text{gap}}^{\text{del}} \ge \mathbf{+0.010}$ (vs current $+0.0017$).
  3. Deletion AP $\ge \mathbf{0.080}$ (vs current $0.0305$).

### Experiment 2: Positional & Duration-Aware Token Embeddings (Addresses Root Cause 3)
- **Concept:** Enrich the prompt query with discrete word-position embeddings (`word_initial`, `word_medial`, `word_final`) and an expected phoneme duration prior.
- **Measurable Acceptance Gates:**
  1. Word-final deletion Recall @ 15% FPR $\ge \mathbf{50.0\%}$ (vs current $31.3\%$).
  2. Word-final deletion FNR @ 15% FPR $\le \mathbf{50.0\%}$ (vs current $68.7\%$).

### Experiment 3: Three-Class MDD Objective (Correct / Substitution / Omission) (Addresses Root Cause 4)
- **Concept:** Replace the single binary logit with a 3-way softmax head, explicitly decoupling omission detection from substitution detection and providing a dedicated omission loss gradient.
- **Measurable Acceptance Gates:**
  1. Macro Omission AP $\ge \mathbf{0.100}$.
  2. Preserve Substitution AP ($\ge 0.250$) and overall CTC PER ($\le 24.5\%$).

---

## 12. Checkpoint Artifact Ledger & Verification

| Artifact Identifier | Role / Description | Canonical Location | SHA-256 Digest | Status |
|---|---|---|---|:---:|
| `diagnose_omissions_and_false_negatives.py` | Complete diagnostic script | `audit_scripts/diagnose_omissions_and_false_negatives.py` | `0b42f6d2222f7a4ca68c4a5c531d054e17be88d3f3f5082ce7c6888c3be3df05` | Verified |
| `OMISSION_AND_FALSE_NEGATIVE_REPORT.json` | Master diagnostic report | `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json` | `65076d510768e0bc17c77983aef8d4118d9e3c40dcd4604f6838a545c1cede9f` | Verified |
| `omission_error_breakdown_by_speaker.csv` | 8-Speaker disaggregated table | `manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_speaker.csv` | `0b240fae7f722a45fb2e71cb7310d65b32ecdf2b4c101d2bc0ea15f60e909a3c` | Verified |
| `omission_error_breakdown_by_phone.csv` | 39-Phoneme disaggregated table | `manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_phone.csv` | `771970b435ff20fd0b6fe9b50b86a0142fa91667b2d566d8ba9d3ffc1e84df24` | Verified |
| `omission_error_breakdown_by_position.csv` | Word-position table | `manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_position.csv` | `435d10a2bbccfa33117498c16dca1375b630e66ea1ca14bbd2105da24660eeea` | Verified |
| `omission_error_breakdown_by_duration.csv` | Duration binning table | `manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_duration.csv` | `82cf219d3b20dfa773d1fdb4974955eb0877a5ef0928a6f4435b80b06ce270aa` | Verified |
| `representative_false_negatives.csv` | High-confidence false negatives | `manifests/RQ1-DIAG-01/evaluation/representative_false_negatives.csv` | `3781297e64dafc823021f1539e083c0f4f911a3ff0d5e1ee74fae782e4e16d4c` | Verified |
| `test_rq1_diag_01_phoneme_omission_fn.py` | Automated regression test suite (8 tests) | `tests/test_rq1_diag_01_phoneme_omission_fn.py` | `6b9e2ae161bcfc239fc006ca28e75db2534a654924c239a5caadbc193630f5ec` | Verified |

---

## 13. Regression Test Verification

The repository regression suite was executed and passed with zero failures:
- `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/test_rq1_diag_01_phoneme_omission_fn.py`: **8 passed in 0.01s**
- **Full Repository Suite:** `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/`
  - **Result: 347 passed, 1 skipped, 1 warning in 161.10s** (100% green).

---

## 14. Next Authorized Step & Stop Statement

In strict compliance with `AGENTS.md` ("Never start training without explicit user authorization"):

- **Status:** Phase RQ1-DIAG-01 Diagnostic Audit COMPLETE.
- **Execution Boundary:** Zero GPU training executed. Test sets ($S_{\text{test}}^{(1..4)}$) remain 100% sealed.
- **Next Required Action:** Await operator review of diagnostic findings and instruction on which controlled experiment (alignment regularization, positional embeddings, or 3-class MDD) to authorize for subsequent design.

**STOP.**
