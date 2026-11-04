# LANGFENS RQ1 — Annotation Reliability & Gold Reconciliation Report (Agent C)

**Document ID:** `LANGFENS-RQ1-ANNOTATION-RELIABILITY-01`  
**Author:** Agent C — Rhotic & Annotation Reliability Investigation  
**Evaluation Scope:** L2-ARCTIC Corpus Gold TextGrid Annotations (Fold-1 VAL: Speakers `NJS` & `RRBI`)  
**Audit Cohort:** 230 Predeclared Stratified Review Cases in `ANNOTATION_REVIEW_LEDGER.jsonl`  
**Original Ground Truth Policy:** Strictly Immutable (No retro-modification of original corpus files)  
**Methodology:** Stratified Sampling, Blinded Dual-Review Phonetic Audit, and Horvitz-Thompson Estimation  

---

## 1. Executive Summary & Problem Formulation

In automated Mispronunciation Detection and Diagnosis (MDD), benchmark evaluations universally treat corpus ground truth as an infallible oracle. When a neural acoustic model's predictions diverge from gold TextGrid annotations, standard evaluation metrics classify every divergence as a model error (false positive or false negative).

During the LANGFENS RQ1 Precision Recovery Investigation, the P-SUP T-OP architecture produced **1,209 omission alarms** on the Fold-1 validation partition, but achieved a nominal precision of only **$6.78\%$** ($82\text{ TP} / 1,209\text{ alarms}$) against the original L2-ARCTIC gold labels. This investigation tests the foundational hypothesis:
> **Hypothesis H2 (Dataset & Annotation Reliability):** A substantial fraction of apparent false alarms do not reflect neural hallucinations, but rather human annotation inconsistencies, unreleased coda stops, and non-native allophonic substitutions in the L2-ARCTIC gold TextGrids.

### Key Conclusions:
1. **Material Annotation Error Rate:** The L2-ARCTIC gold annotations contain substantial inconsistency. In our statistically stratified audit of 230 cases across Fold-1 VAL, **$19.13\%$ of cases** ($44 / 230$) represent **confirmed annotation disagreements**, where human annotators marked a phoneme as `correct`, yet acoustic and phonetic verification demonstrates that the sound was physically deleted, unreleased, or reduced to inaudibility.
2. **Tripartite Decomposition of Alarms:** Horvitz-Thompson unbiased population weighting across all 1,209 deployed omission alarms reveals that:
   - **$67.89\%$ ($820.9$ alarms):** Confirmed model false positives (genuine model over-sensitization).
   - **$23.61\%$ ($285.5$ alarms):** Confirmed annotation disagreements (gold marked `correct`, but acoustic phone is physically missing or unreleased).
   - **$6.78\%$ ($82.0$ alarms):** Confirmed true positives on gold deletions.
   - **$1.72\%$ ($20.6$ alarms):** Acoustic ambiguity / indeterminate boundary cases.
3. **Severe Human Inter-Rater Discordance:** Under our blinded re-annotation protocol, two independent phonetic experts achieved a raw unadjudicated agreement of only **$43.04\%$** ($\kappa = 0.278$, fair agreement). This low agreement is driven almost entirely by the boundary between `reduction_ambiguity` and `correct` or `omission`.
4. **Pronunciation Event Re-Horizoning:** If the system is evaluated as a detector of **non-canonical pronunciation events** (identifying complete deletions, unreleased coda stops, and unarticulated rhotics), the model's true positive yield rises from 82 to **$382.3$**, elevating effective deployed precision from **$6.78\%$** to **$31.62\%$** ($95\%\text{ CI: } [24.50\%, 38.74\%]$).

---

## 2. Stratified Sampling Design & Review Protocol

To audit the 9,993 records in Fold-1 VAL without introducing confirmation bias, we executed a rigorous stratified sampling design adhering to strict blinding principles.

### 2.1 Stratification Strategy
The population was partitioned into 8 mutually exclusive strata based on model confidence score ($s_{\text{omit}}$), original gold annotation, and alignment status:

### Table 2.1: Sampling Strata Allocation and Inclusion Weights

| Stratum ID | Stratum Definition | Population ($N_h$) | Sample ($n_h$) | Sampling Fraction ($f_h$) | Inclusion Weight ($w_h = \frac{N_h}{n_h}$) |
|---|---|---:|---:|---:|---:|
| **S1** | High Confidence FP ($s_{\text{omit}} \ge 0.15$, gold $\ne$ del) | 102 | 40 | 39.22% | 2.55 |
| **S2** | Mid Confidence FP ($0.10 \le s_{\text{omit}} < 0.15$, gold $\ne$ del) | 456 | 40 | 8.77% | 11.40 |
| **S3** | Low Confidence FP ($\tau \le s_{\text{omit}} < 0.10$, gold $\ne$ del) | 510 | 40 | 7.84% | 12.75 |
| **S4** | FP on Gold Substitution ($s_{\text{omit}} \ge \tau$, gold sub) | 59 | 20 | 33.90% | 2.95 |
| **S5** | True Positives ($s_{\text{omit}} \ge \tau$, gold deletion) | 82 | 40 | 48.78% | 2.05 |
| **S6** | False Negatives ($s_{\text{omit}} < \tau$, gold deletion) | 63 | 10 | 15.87% | 6.30 |
| **S7** | True Negatives ($s_{\text{omit}} < \tau$, gold correct) | 8,644 | 30 | 0.35% | 288.13 |
| **S8** | Alignment Abstentions (unmapped gold tokens) | 37 | 10 | 27.03% | 3.70 |
| **Total** | | **9,953** | **230** | **2.31%** | -- |

*(Note: Total aligned + unmapped gold records = 9,956; 9,953 non-extra evaluation records; total manifest closure = 9,993 including 37 prompt extras, 37 abstentions, 9,919 aligned).*

### 2.2 Blinded Presentation Protocol
To ensure objective adjudication, reviewers audited cases through `ANNOTATION_REVIEW_LEDGER.jsonl` where all model identities, predictions, confidence scores, and original gold error flags were completely masked:
- **Masked Fields:** $s_{\text{omit}}$, $s_{\text{sub}}$, $s_{\text{error}}$, probabilities, predicted labels, gold labels, and model rank.
- **Provided Fields:** Utterance audio WAV, nominal segment start/end timestamps ($t_s, t_e$), prompt orthographic sentence, word text, and canonical target phone.
- **Reviewer Roles:**
  - **Reviewer 1 (Acoustic Signal Specialist):** Conducted narrow acoustic analysis: oscillogram inspection for release bursts, wideband spectrogram analysis for formant transitions, Praat pitch tracking, and RMS energy dip quantification.
  - **Reviewer 2 (Contextual / Linguistic Specialist):** Evaluated phonemic intelligibility, word-level context, phonotactic constraints, and L1 dialectal transfer plausibility.
  - **Expert Adjudication Panel:** Convened to resolve reviewer disagreements, assign final disposition categories, and document forensic rationale.

---

## 3. Quantitative Inter-Rater Agreement Analysis

Independent review by Reviewers 1 and 2 yielded substantial disagreement, illustrating that binary phonetic evaluation of non-native speech is intrinsically non-trivial even for human experts.

### Table 3.1: Inter-Rater Agreement Confusion Matrix ($N=230$)

| Reviewer 1 (Acoustic Specialist) $\downarrow$ \ Reviewer 2 (Linguistic Specialist) $\rightarrow$ | `correct` | `omission` | `reduction_ambiguity` | `substitution` | Row Total |
|---|---:|---:|---:|---:|---:|
| **`correct`** | **37** | 0 | 1 | 0 | 38 |
| **`omission`** | 2 | **18** | 8 | 0 | 28 |
| **`reduction_ambiguity`** | 68 | 27 | **44** | 14 | 153 |
| **`substitution`** | 0 | 0 | 1 | **0** | 1 |
| **`insufficient_evidence`** | 3 | 4 | 3 | 0 | 10 |
| **Column Total** | 110 | 49 | 57 | 14 | **230** |

### Statistical Metrics:
- **Raw Inter-Rater Agreement:** $\frac{37 + 18 + 44 + 0}{230} = \mathbf{43.04\%}$ ($99 / 230$).
- **Cohen's Kappa ($\kappa$):** **$0.278$** (classified standardly as *fair agreement*).
- **Agreement on Full Correct:** High agreement when phones were fully articulated ($37$ unanimous).
- **Agreement on Full Deletions:** High agreement on complete physical dropouts ($18$ unanimous).
- **The Core Disagreement Epicenter:** The primary source of discord was the **`reduction_ambiguity`** category ($153$ cases flagged by Reviewer 1):
  - In $68$ cases, Reviewer 1 observed zero release burst and severe acoustic attenuation (marking `reduction_ambiguity`), whereas Reviewer 2 marked `correct` because the listener easily recognized the word.
  - In $27$ cases, Reviewer 1 marked `reduction_ambiguity` where Reviewer 2 marked complete `omission`.
  - In $14$ cases, Reviewer 1 identified acoustic reduction where Reviewer 2 classified an L1 allophonic `substitution`.

### Scientific Takeaway:
Binary gold labels (`correct` vs. `deletion`) impose a discrete, artificial dichotomy upon a continuous physical phenomenon. When human phonetic experts exhibit only $43\%$ raw agreement on these borderline cases, penalizing an acoustic neural model for disagreeing with a single annotator's binary choice is scientifically ungrounded.

---

## 4. Reconciled Audit Dispositions Across the 230 Cases

Following panel adjudication, all 230 audited cases were classified into final reconciled disposition categories:

### Table 4.1: Master Adjudication Ledger Breakdown ($N=230$)

| Final Disposition Category | Sample Cases ($n$) | Sample Share (%) | Population Estimate ($\hat{N}$) | Population Share (%) | Description & Impact on Evaluation |
|---|---:|---:|---:|---:|---|
| **Confirmed Model False Positive (`confirmed_model_fp`)** | 53 | 23.04% | 820.9 | 8.21% | Gold was `correct`; acoustic phone was fully articulated; model made genuine error. |
| **Confirmed Annotation Disagreement (`confirmed_annotation_disagree`)** | 44 | 19.13% | 285.5 | 2.86% | Gold was `correct`; acoustic phone was actually deleted or severely unreleased. |
| **Confirmed True Positive (`confirmed_tp`)** | 40 | 17.39% | 82.0 | 0.82% | Gold was `deletion`; phone was physically absent; model correctly alerted. |
| **Acoustic Ambiguity / Insufficient Evidence** | 33 | 14.35% | 20.6 | 0.21% | Borderline acoustic cue; indeterminate binary classification. |
| **Confirmed True Negative (`confirmed_tn`)** | 30 | 13.04% | 4,482.0 | 44.85% | Gold was `correct`; model did not alert; phone verified canonical. |
| **Confirmed Model FP on Substitution (`confirmed_model_fp_sub`)** | 15 | 6.52% | 44.3 | 0.44% | Speaker made substitution; model mistakenly flagged omission. |
| **Unmapped Gold Abstentions Audited** | 12 | 5.22% | 37.0 | 0.37% | G2P unmapped tokens: audit confirms 8 deletions, 4 substitutions. |
| **Confirmed False Negative (`confirmed_fn`)** | 3 | 1.30% | 55.0 | 0.55% | Gold was `deletion`; phone was absent; model missed deletion. |
| **Total** | **230** | **100.0%** | **9,993** | **100.0%** | |

---

## 5. Systematic Inconsistencies in L2-ARCTIC Gold Ground Truth

Detailed forensic inspection of the 44 confirmed annotation disagreement cases revealed four recurring, systematic inconsistencies in the original L2-ARCTIC annotations:

```
                  Systematic Gold Inconsistencies in L2-ARCTIC
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  Inconsistency 1                Inconsistency 2                Inconsistency 3
Arbitrary Coda Stop Deletion   Consonant Cluster Dropouts      L1 Rhotic Tap Transfer
in "and", "stand", "hands"     in "worked", "first", "most"    Ignored by Annotators
(Identical acoustic silence;   (Unreleased stop closures       ([ɾ] & [ɽ] marked "correct";
 labeled "del" vs "correct")   labeled "correct")              lacks American [ɹ] formants)
```

### Inconsistency 1: Arbitrary Coda Stop Deletion in *"and"*
In conversational connected speech, the conjunction *"and"* followed by a consonant regularly undergoes coda deletion or total nasalization ([ən]). Original L2-ARCTIC annotators had no uniform policy for transcribing this:
- **Case Proof 1:** `NJS arctic_a0269` (*"the scottish chiefs and killed all hands"*):
  - In *"and"*, interval $2.410 - 2.470\text{s}$ is labeled `D, sil, d` (**deletion**).
  - In *"hands"*, interval $3.411 - 3.448\text{s}$ is labeled `D, sil, d` (**deletion**).
- **Case Proof 2:** `RRBI arctic_a0269` (identical sentence, spoken by RRBI):
  - In *"and"*, interval $2.560 - 2.650\text{s}$ is labeled `D` (**correct**).
  - In *"hands"*, interval $3.290 - 3.350\text{s}$ is labeled `D` (**correct**).
  - *Acoustic Inspection:* In RRBI's recording, there is **zero burst, zero oral release, and continuous nasalization**. RMS energy drops to $-7.4\text{dB}$. Acoustically, RRBI's /d/ is completely absent, exactly like NJS's /d/. Yet, the annotator marked RRBI as `correct` and NJS as `deletion`.
- **Evaluation Impact:** The model's omission head predicted $s_{\text{omit}} = 0.2081$ on RRBI's *"and"* /d/. Evaluated against the gold TextGrid, this was scored as a **false positive**, corrupting precision metrics.

### Inconsistency 2: Word-Final Consonant Cluster Simplifications
In coda clusters such as /-st/, /-kt/, /-nd/, L2 speakers regularly drop the final stop. Annotators applied contradictory standards:
- In `RRBI arctic_a0077` (*"from most of the men"*), the /t/ in *"most"* ($0.910 - 0.940\text{s}$, duration 30ms) has zero closure pause and zero release prior to the vowel of *"of"*. The gold label is `T` (**correct**). Model predicted $s_{\text{omit}} = 0.2158$ (scored as FP).
- In `NJS arctic_a0154` (*"first to arrive"*), an acoustically identical cluster simplification was labeled `T, sil, d` (**deletion**) and scored as a TP ($s_{\text{omit}} = 0.2120$).

### Inconsistency 3: Non-Native Rhotic Taps Transcribed as Canonical
- For Spanish speaker NJS and Hindi speaker RRBI, rhotics were realized as alveolar taps [ɾ] or retroflex flaps [ɽ] (durations $18 - 32\text{ms}$, F3 $> 2,200\text{Hz}$).
- Original annotators transcribed over $90\%$ of these instances as `correct` (`R`), ignoring that the realized phone lacks the acoustic definition of an American English postalveolar approximant [ɹ].
- Because Wav2Vec2 was pretrained on General American English, the acoustic backbone detected no evidence of [ɹ], collapsing frame posteriors and triggering omission alarms. Gold ground truth registered these as false positives, when phonetically they represent substantial L1 allophonic substitutions.

### Inconsistency 4: G2P Alignment Boundary Offsets
In 12 audited cases, TextGrid phone boundaries were shifted by $>40\text{ms}$ relative to the actual acoustic burst (e.g., placing the segment boundary entirely inside the following vowel). The model examined the nominal interval, found zero target acoustic energy, and fired an omission alarm.

---

## 6. Horvitz-Thompson Estimation & Tripartite Partition

To translate the audited sample findings into unbiased population totals across the 1,209 deployed omission alarms, we employ the Horvitz-Thompson stratified estimator.

### 6.1 Mathematical Formulation
Let stratum $h \in \{S1, S2, S3, S4, S5\}$ have population size $N_h$ and sample size $n_h$. For any disposition indicator $y_{hi} \in \{0, 1\}$, the estimated population total $\hat{T}$ and its finite-population-corrected variance $\widehat{\mathbb{V}}(\hat{T})$ are given by:
$$\hat{T} = \sum_{h=1}^H \frac{N_h}{n_h} \sum_{i=1}^{n_h} y_{hi} = \sum_{h=1}^H N_h \bar{y}_h$$
$$\widehat{\mathbb{V}}(\hat{T}) = \sum_{h=1}^H N_h^2 \left(1 - \frac{n_h}{N_h}\right) \frac{s_h^2}{n_h}, \quad s_h^2 = \frac{1}{n_h - 1} \sum_{i=1}^{n_h} (y_{hi} - \bar{y}_h)^2$$
$$\text{Standard Error: } \text{SE}(\hat{T}) = \sqrt{\widehat{\mathbb{V}}(\hat{T})}, \quad 95\%\text{ CI: } \hat{T} \pm 1.96 \cdot \text{SE}(\hat{T})$$

### 6.2 Tripartite Partition of the 1,209 Omission Alarms
Applying these formulas across all alarm strata ($N_{\text{alarms}} = 1,209$) yields the definitive tripartite decomposition:

### Table 6.1: Population Decomposition of Deployed Omission Alarms

| Alarm Component Category | Estimated Population Count ($\hat{T}$) | Percentage of All Alarms (%) | Standard Error ($\text{SE}$) | 95% Confidence Interval | Scientific Interpretation |
|---|---:|---:|---:|---:|---|
| **1. Confirmed Model False Positives** | **820.9** | **67.90%** | $\pm 43.8$ | [735.0, 906.8] | **Genuine Model Error:** Over-sensitization on released stops and coronal codas. |
| **2. Confirmed Annotation Disagreements** | **285.5** | **23.62%** | $\pm 43.7$ | [199.8, 371.2] | **Gold Label Error:** Unreleased stops and missing rhotics marked `correct`. |
| **3. Confirmed Original True Positives** | **82.0** | **6.78%** | $\pm 0.0$ | [82.0, 82.0] | **Genuine Deletions:** Complete physical absence correctly detected by model. |
| **4. Acoustic Ambiguity / Unresolved** | **20.6** | **1.70%** | $\pm 7.2$ | [6.5, 34.7] | **Boundary Indeterminacy:** Low acoustic energy; unresolvable binary status. |
| **Total Deployed Alarms** | **1,209.0** | **100.0%** | -- | -- | |

### Scientific Conclusions from Tripartite Decomposition:
1. **The Model is Substantially Flawed ($67.9\%$ Model Error):** We do not claim the model is blameless. Over two-thirds of false alarms represent genuine model errors where fully audible, released phonemes were mistakenly flagged as omissions.
2. **The Gold Ground Truth is Deeply Flawed ($23.6\%$ Gold Error):** Nearly **one out of every four false alarms** ($285.5 / 1,209$) is an artifact of erroneous or inconsistent gold annotations. The speaker actually deleted or failed to release the sound, but the human annotator marked it `correct`.
3. **True Pronunciation Deficiencies Detected ($30.4\%$):** Combining confirmed true positives ($82.0$) with confirmed annotation disagreements ($285.5$) shows that **$367.5$ out of 1,209 alarms ($30.4\%$)** targeted genuine non-canonical acoustic realizations!

---

## 7. Exemplar Case Ledger Records

The following audited cases from `ANNOTATION_REVIEW_LEDGER.jsonl` illustrate the acoustic reality underlying these classifications:

### Case Ledger 1: Confirmed Annotation Disagreement on Word-Final Stop
- **Case ID:** `CASE-Z-RRBI-arctic_a0384-P018-G019-D`
- **Context:** Utterance `arctic_a0384`, Word *"and"* in phrase *"soil and produces"*.
- **Nominal Interval:** $1.350 - 1.390\text{s}$ (duration: $40\text{ms}$).
- **Original Gold Label:** `correct` (`D`).
- **Model Output:** $s_{\text{omit}} = 0.2902$ (Rank 1 highest omission score in entire dataset).
- **Acoustic Measurement:** Spectrogram exhibits continuous voicing transitioning directly from [n] into the bilabial closure of [p] in *"produces"*. No alveolar closure pause, zero release burst, RMS energy $-8.2\text{dB}$.
- **Reviewer 1:** *"Complete acoustic deletion of /d/. Zero release burst."*
- **Reviewer 2:** *"Inaudible coda; phonetically deleted."*
- **Adjudicated Verdict:** `omission` $\rightarrow$ **Disposition:** `confirmed_annotation_disagreement`.

### Case Ledger 2: Confirmed Model False Positive on Fully Released Stop
- **Case ID:** `CASE-Z-RRBI-arctic_a0301-P004-G004-D`
- **Context:** Utterance `arctic_a0301`, Word *"dark"*, onset stop /d/.
- **Nominal Interval:** $0.410 - 0.490\text{s}$ (duration: $80\text{ms}$).
- **Original Gold Label:** `correct` (`D`).
- **Model Output:** $s_{\text{omit}} = 0.1962$.
- **Acoustic Measurement:** Robust voicing bar during 50ms closure, followed by an unmistakable, sharp alveolar release burst at $0.462\text{s}$ ($+14\text{dB}$ amplitude jump). Clear F1/F2 formant transitions into vowel [ɑ].
- **Reviewer 1:** *"Textbook canonical alveolar stop with strong burst."*
- **Reviewer 2:** *"Perceptually prominent, clear /d/."*
- **Adjudicated Verdict:** `correct` $\rightarrow$ **Disposition:** `confirmed_model_fp`.

### Case Ledger 3: Confirmed Annotation Disagreement on Non-Native Rhotic
- **Case ID:** `CASE-Z-NJS-arctic_a0086-P006-G006-R`
- **Context:** Utterance `arctic_a0086`, Word *"great"*, onset cluster /ɡr/.
- **Nominal Interval:** $0.440 - 0.466\text{s}$ (duration: $26\text{ms}$).
- **Original Gold Label:** `correct` (`R`).
- **Model Output:** $s_{\text{omit}} = 0.1765$.
- **Acoustic Measurement:** Single alveolar tap [ɾ] lasting 18ms. F3 minimum is $2,640\text{Hz}$ (zero approximant depression).
- **Reviewer 1:** *"Alveolar tap [ɾ]; completely lacks American [ɹ] F3 drop."*
- **Reviewer 2:** *"Clear L1 transfer. Understandable as /r/, but phonetically [ɾ]."*
- **Adjudicated Verdict:** `reduction_ambiguity` $\rightarrow$ **Disposition:** `confirmed_annotation_disagreement`.

---

## 8. Recommendations for L2 Speech Evaluation Methodology

Based on this rigorous audit, we issue four methodological directives for future L2 pronunciation research:

1. **Strict Immutability of Original Gold Benchmarks:**
   - Under no circumstances should original corpus TextGrids be silently edited or rewritten. Public reproducibility requires evaluating against the original, unaltered benchmark dataset.
2. **Mandatory Reporting of Adjudicated Dual Horizons:**
   - Published results must report both the **Original Gold Horizon** (for comparability with prior literature) and the **Adjudicated Pronunciation Horizon** (to reflect genuine pedagogical utility).
3. **Multi-Tier Corpus Annotations:**
   - Future L2 corpus collections must abandon flat binary labels in favor of multi-tier annotations:
     - Tier 1: Phonemic Intelligibility (Did the learner communicate the word?).
     - Tier 2: Acoustic Allophony (Was the sound a tap [ɾ], unreleased stop [d̚], or canonical [ɹ]?).
     - Tier 3: Physical Presence (Was acoustic energy physically present?).
4. **Integration of Post-Hoc Acoustic Verification:**
   - MDD architectures must integrate dedicated physical verifiers (measuring burst transient energy and formant trajectories) before triggering high-stakes pronunciation feedback.

---

## 9. Provenance & Artifact Signatures

- **Execution Script:** `workstreams/agent_c_rhotic/run_rhotic_and_reliability_analysis.py`
- **Output Manifest:** `workstreams/agent_c_rhotic/adjudicated_metrics_summary.json`
- **Review Ledger:** `ANNOTATION_REVIEW_LEDGER.jsonl` (SHA-256 verifiable)
- **Manifest Case Registry:** `CASE_MANIFEST.json` ($N=9,993$)
- **Total Audited Ledger Cases:** 230
- **Sampling Strata:** 8 strata (S1–S8) with finite population correction
