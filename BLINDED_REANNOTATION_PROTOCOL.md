# LANGFENS RQ1 — Blinded Re-Annotation Protocol (Agent C)

**Document ID:** `LANGFENS-RQ1-REANNOTATION-PROTOCOL-01`  
**Purpose:** Independent Gold Annotation Reliability Study for L2-ARCTIC Phoneme Annotations  
**Authority:** `AGENTS.md` & LANGFENS RQ1 Precision Investigation Mission  
**Target Sample:** Predeclared Stratified Cohort ($N=230$ tokens from Fold-1 VAL)  
**Ledger Output:** `ANNOTATION_REVIEW_LEDGER.jsonl`  

---

## 1. Objectives & Non-Negotiable Safeguards

The mission of this protocol is to determine whether apparent false positive omission predictions by Subtype Supervision (`T-OP`) reflect genuine model classification errors, inconsistencies/ambiguities in the original L2-ARCTIC gold labels, or borderline acoustic phenomena.

### Mandatory Blindness Invariants
1. **Zero Model Exposure:** Human reviewers and phonetic auditors must **never receive model prediction scores** (`s_omit`, `s_error`, `s_sub`), probability vectors, or model-generated ranks.
2. **Zero Outcome Exposure:** Reviewers must **never see the evaluation outcome** (e.g. `TP`, `FP`, `TN`, `FN`) or the sampling stratum.
3. **Zero Gold Outcome Exposure:** Reviewers must **never see the original TextGrid error classification** (`correct`, `deletion`, `substitution`) during the primary annotation phase.
4. **Original Gold Immutability:** Original L2-ARCTIC TextGrids and frozen split manifests must remain byte-for-byte immutable. All adjudicated findings must be written strictly to a versioned external ledger (`ANNOTATION_REVIEW_LEDGER.jsonl`).
5. **Human Adjudication Requirement:** Automated ASR transcripts, forced alignment timestamps, and acoustic thresholding must **never be treated as authoritative replacement ground truth**. If a human expert panel is unavailable, the review package must be prepared and preserved, but formal certification must remain marked as `UNRESOLVED` (prohibiting unverified automated relabeling claims).

---

## 2. Predeclared Stratified Sampling Design

To maximize statistical power across critical decision boundaries while bounding review cost, 230 cases were sampled from Fold-1 VAL using fixed seed `1337` in `CASE_MANIFEST.json`:

| Stratum ID | Stratum Definition | Population Size ($N_h$) | Sample Size ($n_h$) | Sampling Probability ($p_h$) | Selection Weight ($w_h = 1/p_h$) |
|---|---|---:|---:|---:|---:|
| **S1** | Extreme FP ($s_{\text{omit}} \ge 0.20$, gold correct) | 8 | 8 | 1.0000 | 1.00 |
| **S2** | High FP ($0.12 \le s_{\text{omit}} < 0.20$, gold correct) | 293 | 50 | 0.1706 | 5.86 |
| **S3** | Near-Threshold FP ($\tau \le s_{\text{omit}} < 0.12$, gold correct) | 767 | 40 | 0.0522 | 19.18 |
| **S4** | Substitution Alarms ($s_{\text{omit}} \ge \tau$, gold substitution) | 59 | 20 | 0.3390 | 2.95 |
| **S5** | True Positives ($s_{\text{omit}} \ge \tau$, gold omission) | 82 | 40 | 0.4878 | 2.05 |
| **S6** | False Negatives ($s_{\text{omit}} < \tau$, gold omission) | 55 | 30 | 0.5455 | 1.83 |
| **S7** | True Negatives Control ($s_{\text{omit}} < 0.01$, gold correct) | 4,482 | 30 | 0.0067 | 149.40 |
| **S8** | Unmapped Gold Abstentions (unaligned gold positions) | 37 | 12 | 0.3243 | 3.08 |
| **Total** | | **9,993** | **230** | — | — |

*Mathematical Justification:* Oversampling rare high-confidence false alarms (100% of S1) and true positives/negatives ensures dense qualitative analysis, while Horvitz-Thompson weights $w_h$ allow mathematically unbiased population-level estimation with closed-form standard errors.

---

## 3. Blinded Presentation Interface & Reviewer Payload

For every case, the reviewer receives only the blinded presentation dictionary (`blinded_presentation`):

```json
{
  "case_id": "CASE-Z-RRBI-arctic_a0384-P018-G019-D",
  "speaker_id": "RRBI",
  "utterance_id": "arctic_a0384",
  "prompt_sentence": "the farmer works the soil and produces grain",
  "word_text": "and",
  "canonical_phone": "D",
  "target_xmin_sec": 1.350,
  "target_xmax_sec": 1.390,
  "target_duration_ms": 40.0,
  "audio_window_start_sec": 1.250,
  "audio_window_end_sec": 1.490
}
```

### Presentation Rules:
- The reviewer is presented with the prompt sentence, highlighting the target word and intended phoneme.
- The reviewer listens to:
  1. The complete utterance waveform.
  2. The target word waveform.
  3. The local padded audio window ($\pm 100\text{ms}$ around the nominal segment).
- The reviewer inspects the narrowband and wideband spectrograms (formant tracks, closure bar, release burst, voicing bar).
- No model prediction score, threshold, or original gold label is displayed.

---

## 4. Phonetic Review Taxonomy & Decision Rubrics

Reviewers must select exactly one primary judgment category from the following six mutually exclusive options:

### 1. `omission` (Phone Absent / Deleted)
- **Criteria:** Complete physical absence of the canonical acoustic cues.
- For stops (/t, d, p, b, k, g/): Zero oral closure silence, zero release burst, and continuous formant transitions between adjacent vowels/nasals without interruption.
- For fricatives/nasals/liquids: Zero characteristic aperiodic friction, zero nasal murmur, or zero rhotic F3 dip.
- **Rule:** If an expected coda stop in a cluster has no audible burst, but a distinct closure interval ($>40\text{ms}$) is present, it must NOT be labeled `omission`; it must be evaluated under `reduction_ambiguity`.

### 2. `correct` (Canonical Realization Present)
- **Criteria:** The phoneme is clearly articulated in accordance with standard acoustic-phonetic characteristics.
- For stops: Visible closure interval followed by an audible release burst and voice onset time (VOT) appropriate for the phoneme.
- For vowels/liquids: Steady-state formants spanning $>40\text{ms}$.

### 3. `substitution` (Different Phoneme Realized)
- **Criteria:** The speaker produced an identifiable alternative phoneme rather than the prompt phoneme.
- Examples: Target `/th/` realized as `[t]` or `[s]`; target `/dh/` realized as `[d]`; target `/z/` realized as `[s]`.
- Reviewers must record the specific realized phone symbol in their notes.

### 4. `reduction_ambiguity` (Partial Cue / Unreleased / Coarticulated / Tapped)
- **Criteria:** The phone is not physically absent, but fails canonical textbook criteria due to conversational reduction, L1 transfer, or clustering.
- Sub-categories:
  - *Unreleased stop:* Tongue closure occurred, but no release burst occurred prior to the next consonant (e.g. *and killed* where /d/ merges into /k/ closure).
  - *Alveolar/retroflex tap:* Rhotic /r/ or coronal stop realized as an ultra-brief tap [ɾ/ɽ] ($15 - 30\text{ms}$).
  - *Vowel centralization / lenition:* Syllable nucleus reduced to weak murmur.
- **Pedagogical Meaning:** The sound is recognizable to human listeners through context, but acoustically substandard for strict ASR phonetic matching.

### 5. `mapping_issue` (Alignment / Boundary Misalignment)
- **Criteria:** The phoneme was pronounced, but the TextGrid boundary was severely shifted ($>50\text{ms}$) into an adjacent phone, or a word boundary alignment error occurred.

### 6. `insufficient_evidence` (Low SNR / Acoustic Artifact)
- **Criteria:** Audio clipping, low signal-to-noise ratio, background breath/pop, or recording artifact makes acoustic adjudication scientifically indeterminate.

---

## 5. Multi-Reviewer Adjudication Workflow

To ensure scientific rigor, each sampled case undergoes a three-stage adjudication:

```
[Sampled Case] ───> [Reviewer 1: Acoustic Perspective] ──────┐
               ───> [Reviewer 2: Contextual Perspective] ─────┼──> [Consensus Check]
                                                               │         │
                                                       (Agreement)  (Disagreement)
                                                               │         │
                                                               ▼         ▼
                                                       [Adjudicated] [Expert Panel]
                                                                         │
                                                                         ▼
                                                                   [Final Verdict]
```

1. **Independent Evaluation:** Reviewer 1 (focusing on strict physical acoustic evidence) and Reviewer 2 (focusing on L1 phonetic transfer and conversational phonology) independently score the case.
2. **Consensus Rule:** If Reviewer 1 and Reviewer 2 assign the identical category, the verdict is accepted without change (`inter_rater_agreement = true`).
3. **Adjudication Rule:** If reviewers disagree:
   - If one reviewer selects `reduction_ambiguity` and the other selects `omission` or `correct`, the case is adjudicated as **`reduction_ambiguity`** (acknowledging the acoustic indeterminacy).
   - If one selects `omission` and the other selects `correct`, the expert panel examines the waveform spectrogram to determine if closure energy is present. If energy exists, it is adjudicated as `reduction_ambiguity`; if total silence ($<-40\text{dB}$) spans the entire window, it is adjudicated as `omission`.
   - Other disagreements are reconciled by consensus notes recorded in the ledger.

---

## 6. Disposition Categories & Downstream Reconciliation

Each adjudicated record is mapped to one of the following research categories:

1. **`confirmed_model_fp`:** The original gold was `correct`, and adjudication confirmed `correct`. The model's omission alarm is an unambiguous model error.
2. **`confirmed_annotation_disagreement`:** The original gold was `correct`, but adjudication determined the phone was actually `omission` or `reduction_ambiguity` (the model was right to detect non-standard speech; the gold annotation was deficient).
3. **`confirmed_model_fp_sub`:** The speaker produced a substitution, but the model erroneously fired an omission alarm.
4. **`confirmed_tp`:** Both gold and adjudication agree the phone was omitted (True Positive).
5. **`confirmed_fn`:** The phone was truly omitted, but the model failed to detect it (False Negative).
6. **`insufficient_evidence`:** Acoustic noise or alignment boundary drift prevents definitive classification.

---

## 7. Versioning & Archival Schema

The resulting adjudication records are stored in `ANNOTATION_REVIEW_LEDGER.jsonl`. Each line represents one self-contained, cryptographically verifiable JSON record.

Original gold files (`authorized_RQ1-TRN-02A_fold1_train_val_only.zip`) remain untouched. All downstream analyses must read from `ANNOTATION_REVIEW_LEDGER.jsonl` rather than modifying raw benchmark files.
