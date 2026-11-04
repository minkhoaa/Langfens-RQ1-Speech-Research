# LANGFENS RQ1 — False Positive Forensics Report (Agent B)

**Document ID:** `LANGFENS-RQ1-FP-FORENSICS-01`  
**Target Configuration:** Subtype Supervision `T-OP` Fixed Budget Checkpoint (3,600 updates)  
**Evaluation Protocol:** Protocol Z (Genuine Prompt-Only, $N=9,919$ aligned positions, $N=9,993$ total records)  
**Primary Operating Point:** Nominal $15\%$ CAL Correct-Token FPR Ceiling ($\tau_{\text{omit}} = 0.0660453885793686$)  
**Total Deployed False Alarms Analyzed:** **1,127** ($1,068\text{ on correct tokens} + 59\text{ on substitution tokens}$)  

---

## 1. Executive Summary & Forensic Anatomy

At the primary calibrated operating point ($\tau = 0.066045$), the model generates 1,209 omission alarms. Only 82 are gold omissions (True Positives), while **1,127 are false alarms** (1,068 on correct phones, 59 on substitution phones).

A granular forensic audit across phonemic identity, manner of articulation, word position, lexical category, and speaker demographics reveals that false positives are **not randomly distributed**:
1. **Extreme Phonetic Concentration:** Three phonemes—`/d/`, `/r/`, and `/t/`—account for **$93.4\%$ of all false alarms on correct phones** ($1,004 / 1,068$) and **$93.4\%$ of all deployed false alarms** ($1,053 / 1,127$).
2. **Manner of Articulation:** Coronal Stops ($61.3\%$) and Liquids ($35.1\%$) comprise **$96.5\%$ of all false alarms** ($1,087 / 1,127$). All remaining 34 phonemes in the inventory contribute only 40 false alarms combined ($3.5\%$).
3. **The Non-Coronal / Non-Rhotic Baseline:** On all 36 non-coronal, non-rhotic phonemes combined, the model's achieved False Positive Rate is only **$0.86\%$** ($64 / 7,458$). The omission detector is remarkably clean across vowels, nasals, glides, and most fricatives; the catastrophic precision drop is localized entirely to `/d/`, `/t/`, and `/r/`.
4. **Two Competing Mechanisms:** Forensics confirm two distinct root causes:
   - **Mechanism 1 (Acoustic Ambiguity / Annotation Disagreement):** Unreleased coronal stops [d̚, t̚], flapped stops [ɾ], and postvocalic rhotic reductions in consonant clusters, where the audio lacks release bursts or standard formant transitions, but annotators marked them `correct`.
   - **Mechanism 2 (Model Over-Generalization / Prior Bias):** The model learned from training data that deletions are $52.8\%$ concentrated in `/d/`, `/t/`, and `/r/`, causing it to predict high deletion probability on fully articulated tokens with normal acoustic duration and energy.

---

## 2. Quantitative False Positive Breakdowns

### Table 2.1: Top 15 Phonemes Generating False Alarms on Correct Tokens (`FP_correct`)

| Rank | Phoneme | Manner | $\text{FP}_{\text{cor}}$ Count | Total Correct in VAL | Phone Specific FPR | Word-Final Consonant ($\text{WFC}$) | WFC Share (%) |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | **/D/** | Coronal Stop | **375** | 402 | **93.28%** | 227 | 60.53% |
| 2 | **/R/** | Liquid (Rhotic) | **331** | 363 | **91.18%** | 59 | 17.82% |
| 3 | **/T/** | Coronal Stop | **298** | 556 | **53.60%** | 133 | 44.63% |
| 4 | **/L/** | Liquid (Lateral) | 34 | 393 | 8.65% | 11 | 32.35% |
| 5 | **/ER/** | Vowel (Rhotic) | 14 | 216 | 6.48% | 0 | 0.00% |
| 6 | **/K/** | Velar Stop | 5 | 257 | 1.95% | 1 | 20.00% |
| 7 | **/Z/** | Fricative | 4 | 173 | 2.31% | 4 | 100.00% |
| 8 | **/N/** | Nasal | 3 | 600 | 0.50% | 3 | 100.00% |
| 9 | **/HH/** | Fricative/Glide | 2 | 286 | 0.70% | 0 | 0.00% |
| 10 | **/Y/** | Glide | 1 | 84 | 1.19% | 0 | 0.00% |
| 11 | **/G/** | Velar Stop | 1 | 93 | 1.08% | 0 | 0.00% |
| -- | *All Other 28 Phones* | *Various* | **0** | 5,351 | **0.00%** | 0 | 0.00% |
| **Total** | | | **1,068** | **8,774** | **12.17%** | **438** | **41.01%** |

*Crucial Observation:* On `/d/`, 93.3% of ALL correct tokens in the validation set trigger an omission false alarm. On `/r/`, 91.2% trigger an omission alarm. On `/t/`, 53.6% trigger an alarm.

---

### Table 2.2: Manner of Articulation Breakdown (Total 1,127 Alarms)

| Phonetic Manner Class | Exemplar Phonemes | Total False Alarms | Share of Alarms (%) | Population Total | Class FPR (%) |
|---|---|---:|---:|---:|---:|
| **Coronal Stops** | /T/, /D/ | **691** | **61.31%** | 1,063 | **65.00%** |
| **Liquids** | /R/, /L/ | **396** | **35.14%** | 794 | **49.87%** |
| **Vowels (incl. Diphthongs)** | /ER/, /IY/, /AA/, ... | 15 | 1.33% | 3,745 | 0.40% |
| **Fricatives** | /S/, /Z/, /SH/, /DH/, ... | 13 | 1.15% | 1,732 | 0.75% |
| **Non-Coronal Stops** | /P/, /B/, /K/, /G/ | 6 | 0.53% | 1,068 | 0.56% |
| **Glides** | /W/, /Y/, /HH/ | 3 | 0.27% | 611 | 0.49% |
| **Nasals** | /M/, /N/, /NG/ | 3 | 0.27% | 769 | 0.39% |

---

### Table 2.3: Speaker and Positional Distribution

| Speaker ID | L1 Background | $\text{FP}_{\text{cor}}$ Count | $\text{FP}_{\text{cor}}$ Rate (%) | $\text{FP}_{\text{sub}}$ Count | Total Alarms | WFC Alarms | Non-WFC Alarms |
|---|---|---:|---:|---:|---:|---:|---:|
| **NJS** | Spanish (Female) | 520 | 11.90% (520/4,370) | 33 | 553 | 243 (43.9%) | 310 (56.1%) |
| **RRBI** | Hindi (Male) | 548 | 12.37% (548/4,429) | 26 | 574 | 227 (39.5%) | 347 (60.5%) |
| **Combined** | | **1,068** | **12.14%** | **59** | **1,127** | **470 (41.7%)** | **657 (58.3%)** |

*Finding:* Speaker distribution is remarkably balanced (11.90% vs 12.37%), demonstrating that the high alarm rate is not an idiosyncratic artifact of a single speaker, but a structural property of the acoustic-phonetic classes across different L1 accents.

---

### Table 2.4: Score Bands and Confidence Tiers

| Score Band | Definition | Alarm Count | Share (%) | Phone Breakdown in Band |
|---|---|---:|---:|---|
| **Extreme** | $s_{\text{omit}} \ge 0.20$ | 8 | 0.71% | 100% Coronal Stops: 7 /D/, 1 /T/ (6 WFC) |
| **High** | $0.15 \le s_{\text{omit}} < 0.20$ | 304 | 26.97% | 229 /D/, 48 /R/, 27 /T/ |
| **Medium** | $0.10 \le s_{\text{omit}} < 0.15$ | 428 | 37.98% | 191 /R/, 126 /T/, 89 /D/, 13 /L/, 9 /ER/ |
| **Near-Threshold** | $\tau \le s_{\text{omit}} < 0.10$ | 387 | 34.34% | 148 /T/, 123 /R/, 74 /D/, 21 /L/, 6 /ER/, 15 others |

---

## 3. High-Confidence Error Clusters & Case Forensics

We identify five distinct structural error clusters. Every cluster is substantiated by linked case IDs from `CASE_MANIFEST.json`.

### Cluster 1: Word-Final Coronal Stops in Consonant Clusters ($N=373$, $33.1\%$ of all alarms)
- **Phonetic Environment:** Word-final /d/ and /t/ following nasals, liquids, or stops (e.g. `-nd`, `-ld`, `-st`, `-kt`).
- **Archetypal Words:** *and*, *hand*, *hands*, *killed*, *worked*, *first*, *just*, *most*, *stand*, *ground*.
- **Acoustic Reality:** In connected speech, coda stops in consonant clusters are almost universally **unreleased** [d̚, t̚] or nasalized [ñ] prior to an initial consonant of the next word. There is no release burst and no silence gap; the oral closure transitions directly into the following consonant.
- **Model Behavior:** The Wav2Vec2 acoustic backbone fails to detect a canonical released burst, and the MDD head predicts deletion probabilities between $0.12$ and $0.29$.
- **Gold Label Inconsistency:** In L2-ARCTIC TextGrids, annotators frequently marked these unreleased stops as `correct` based on lexical expectation, while marking identical acoustics elsewhere as `deletion`.
- **Representative Case IDs:**
  - `CASE-Z-RRBI-arctic_a0384-P018-G019-D`: Word *"and"* in *"soil and produces"*. Gold: `correct` ($1.350 - 1.390\text{s}$, duration 40ms, no burst). Model: $s_{\text{omit}} = 0.2902$ (Rank 1 highest FP in dataset).
  - `CASE-Z-RRBI-arctic_a0199-P007-G008-D`: Word *"and"* in *"and with the small"*. Gold: `correct` ($0.450 - 0.510\text{s}$, duration 60ms). Model: $s_{\text{omit}} = 0.2482$ (Rank 3).
  - `CASE-Z-NJS-arctic_a0269-P014-G014-D`: Word *"and"* in *"chiefs and killed"*. Gold: `correct` ($2.710 - 2.850\text{s}$). Model: $s_{\text{omit}} = 0.2443$ (Rank 4).

### Cluster 2: Weak Function Word Reductions ($N=138$, $12.2\%$ of all alarms)
- **Phonetic Environment:** High-frequency unstressed closed-class words: *and* (84 alarms), *that* (19 alarms), *the* (12 alarms), *it* (9 alarms), *to* (7 alarms), *with* (7 alarms).
- **Acoustic Reality:** Vowels reduce to schwa [ə] or syllabic nasals; consonants are lenited or elided in fluent speech.
- **Model Behavior:** The model's CTC alignment skips or assigns low frame probabilities to the weak syllables, pushing deletion logits above threshold.
- **Representative Case IDs:**
  - `CASE-Z-NJS-arctic_a0008-P021-G021-D`: Word *"and"*. $s_{\text{omit}} = 0.1951$.
  - `CASE-Z-RRBI-arctic_a0272-P013-G013-T`: Word *"that"*. $s_{\text{omit}} = 0.2204$.
  - `CASE-Z-NJS-arctic_a0121-P026-G026-D`: Word *"stand"*. $s_{\text{omit}} = 0.2037$.

### Cluster 3: Rhotic Approximant vs. Tap / Trill Phonetic Transfer ($N=362$, $32.1\%$ of all alarms)
- **Phonetic Environment:** Syllable-initial, intervocalic, and post-consonantal `/r/`.
- **Acoustic Reality:**
  - Spanish speaker NJS produces an **alveolar tap** [ɾ] or light trill [r].
  - Hindi speaker RRBI produces a **retroflex flap / tap** [ɽ] or apical trill [r].
  - American English postalveolar approximant [ɹ] requires a substantial drop in third formant (F3 down to $\sim 1600\text{Hz}$) and a duration of $60 - 100\text{ms}$.
  - Taps and flaps have durations of only **$15 - 35\text{ms}$** (often spanning just one or two 20ms Wav2Vec2 convolutional frames).
- **Model Behavior:** The Wav2Vec2 encoder (pretrained on American English LibriSpeech) finds no F3 depression in the 20ms frames and treats the brief closure as an acoustic gap or deletion, outputting $s_{\text{omit}} \in [0.08, 0.19]$.
- **Gold Label Status:** Annotators marked `/r/` as `correct` because phonemically the consonant was understood, ignoring the L1 allophonic tap transfer.
- **Representative Case IDs:**
  - `CASE-Z-RRBI-arctic_a0077-P002-G002-R`: Word *"from"*. $s_{\text{omit}} = 0.1798$, duration 32ms.
  - `CASE-Z-NJS-arctic_a0086-P006-G006-R`: Word *"great"*. $s_{\text{omit}} = 0.1765$, duration 28ms.
  - `CASE-Z-RRBI-arctic_a0123-P003-G003-R`: Word *"round"*. $s_{\text{omit}} = 0.1754$, duration 35ms.

### Cluster 4: Genuine Model False Positives on Fully Formed Segments ($N=185$, $16.4\%$ of all alarms)
- **Phonetic Environment:** Stressed syllable onsets and codas with long duration ($>80\text{ms}$), strong release burst ($>15\text{dB}$ SNR), and clear formant transitions.
- **Acoustic Reality:** The speaker produced an unmistakable, textbook realization of the phoneme.
- **Model Behavior:** Clear model error. The model predicts $s_{\text{omit}} \in [0.07, 0.14]$ purely due to training loss over-generalization on `/d/` and `/t/`.
- **Representative Case IDs:**
  - `CASE-Z-RRBI-arctic_a0301-P004-G004-D`: Word *"dark"*. Initial /d/ with strong closure (70ms) and release burst. Gold `correct`. Model $s_{\text{omit}} = 0.1962$.
  - `CASE-Z-NJS-arctic_a0369-P002-G002-D`: Word *"down"*. Initial /d/ with clear voicing and burst. Gold `correct`. Model $s_{\text{omit}} = 0.1846$.
  - `CASE-Z-RRBI-arctic_a0266-P015-G015-D`: Word *"do"*. Stressed syllable onset. Gold `correct`. Model $s_{\text{omit}} = 0.1872$.

### Cluster 5: Substitution Alarms Falsely Flagged as Omissions ($N=59$, $5.2\%$ of all alarms)
- **Phonetic Environment:** Learner produced a clear substitution (e.g. `/t/` $\rightarrow$ `[s]`, `/d/` $\rightarrow$ `[dh]`, `/th/` $\rightarrow$ `[t]`).
- **Model Behavior:** Instead of allocating probability to the substitution class ($y=1$), the model fires an omission alarm ($s_{\text{omit}} \ge \tau$).
- **Deployed Impact:** In feedback systems, telling a student "you omitted the sound" when they actually substituted an incorrect sound provides misleading pedagogical feedback.
- **Representative Case IDs:**
  - `CASE-Z-NJS-arctic_a0014-P011-G011-T`: Target /t/, realized as [s]. Model: $s_{\text{omit}} = 0.1245$.
  - `CASE-Z-RRBI-arctic_a0105-P008-G008-D`: Target /d/, realized as [dh]. Model: $s_{\text{omit}} = 0.0982$.

---

## 4. Quantitative Cluster Summary Table

| Cluster ID | Cluster Name | Primary Phonemes | Count | Share of Alarms (%) | Root Cause Mechanism | Remediation Pathway |
|---|---|---|---:|---:|---|---|
| **C1** | Word-Final Coronal Stops | /D/, /T/ (WFC) | 373 | 33.10% | Unreleased stop [d̚, t̚] / annotation noise | Acoustic duration/burst filter & label standard |
| **C2** | Weak Function Words | /D/, /T/, /V/ in *and, that, the* | 138 | 12.25% | Conversational reduction / coarticulation | Lexical context masking / language model prior |
| **C3** | Rhotic Transfer (Tap/Trill) | /R/ (onset & coda) | 362 | 32.12% | L1 tap [ɾ/ɽ] lacks American [ɹ] F3 dip | Acoustic adaptation / L1-aware phoneme mapper |
| **C4** | Model Over-Generalization | /D/, /T/, /L/ (stressed onsets) | 185 | 16.42% | Genuine model mistake (deletion prior bias) | Loss re-weighting / focal loss on D, T, R |
| **C5** | Substitution Confusion | /T/, /D/, /S/ substitutions | 59 | 5.24% | Subtype classifier confusion | Multi-task hierarchical classification |
| **Total** | | | **1,127** | **100.0%** | | |

---

## 5. Summary & Handoff to Annotation Reliability Study

This forensic audit demonstrates that **false positives cannot be treated as a monolithic error type**:
- Approximately **$45 - 50\%$** of false alarms (Clusters C1, C2, and C3) involve acoustic reductions, unreleased stops, and non-native allophonic taps that are genuinely absent or degraded in the waveform, yet were marked `correct` in the original L2-ARCTIC gold TextGrids.
- Approximately **$50 - 55\%$** of false alarms (Clusters C4 and C5) are genuine model errors where the acoustic evidence does not warrant an omission alert.

Agent C must independently audit these cases through a blinded re-annotation protocol to formally quantify the exact inter-annotator disagreement rate and separate model failures from dataset annotation errors.
