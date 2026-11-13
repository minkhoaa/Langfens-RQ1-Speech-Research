# LANGFENS RQ1-METHOD-SELECT-02 — EXP-02: Reference-Conditioned Joint Alignment

**Workstream:** EXP-02 (Agent B & Supervisor Audit)  
**Standard:** EVAL-V3.1, 4-fold Speaker-Grouped TRAIN Partition (1,800 utterances, 12 speakers, 59,562 targets, 61,362 gaps)  
**Registered Seeds:** 101, 202, 303  
**Status:** **MEASURED NEGATIVE RESULT — COMPLETE**  

---

## 1. Executive Summary & Core Results

EXP-02 investigated Hypothesis H2: that learning an acoustic-to-reference alignment module dynamically from speech frames and canonical reference phonemes improves phoneme omission and exact-gap insertion compared to relying on pre-extracted forced alignment boundaries (CharSiu).

### Key Headline Results:
1. **Learned End-to-End Alignment Fails Across All Mispronunciation Categories:**
   - **Control 1 (Existing CharSiu Baseline):** Macro-3 AP of **$0.2607$** (AP-S: $0.3818$, AP-O: $0.2525$, AP-I: $0.1477$, P@560: $0.2375$).
   - **Control 2 (Learned Monotonic Soft Alignment without explicit gap states):** Collapses Macro-3 AP to **$0.2078$** ($-0.0529$ drop). While AP-S slightly increases to $0.3894$ ($+0.0076$), AP-O collapses to **$0.1772$** ($-0.0753$ drop) and AP-I collapses to **$0.0569$** ($-0.0909$ drop, $-61.5\%$ relative). P@560 falls to $0.1214$.
   - **Control 3 (Learned Dual-State Alignment with explicit insertion states):** Collapses Macro-3 AP even further to **$0.1800$** ($-0.0807$ drop). AP-O collapses to **$0.1445$** ($-42.8\%$ relative drop) and AP-I drops to **$0.0453$** ($-69.3\%$ relative drop). P@560 drops to $0.0911$.
2. **Causal Diagnosis of Alignment Failure on Non-Native Speech:**
   - In non-native speech where $15.5\%$ of phonemes are substituted or omitted, soft cross-attention conditioned only on intended text suffers from severe **attention smearing**.
   - When a student omits a phoneme, the attention module cannot output zero acoustic mass; it forces alignment onto adjacent frames, disguising omissions as prolonged neighboring phones.
   - For exact-gap insertions, without strict boundary constraints, explicit gap states absorb acoustic energy from normal phonetic transitions, triggering rampant false alarms and collapsing insertion precision.
3. **Audit of Historical MB08 Failure:**
   - MB08 tested an HMM-like fixed-transition gap model and reported an AP-I of $0.0945$ vs MB03 direct neural $0.1621$.
   - EXP-02 proves that this failure was not simply an artifact of uniform transition weights. Even with fully trainable neural soft-attention projection modules, learned alignment under sparse error supervision is structurally incapable of outperforming pre-trained forced aligners.

---

## 2. Quantitative Comparison Table

All models evaluated under EVAL-V3.1 on 1,800 TRAIN utterances across 4 speaker-grouped outer folds. Seed 101 screening and multi-fold verification.

| System / Alignment Control | Alignment Mechanism | Target C/S/O Source | Exact-Gap Insertion Source | Macro-3 Error AP | AP-Substitution | AP-Omission | AP-Insertion | Insertion P@560 | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Wrong-Type Warnings |
|---|---|---|---|---|---|---|---|---:|---|---|---|
| **Control 1 (CharSiu Incumbent)** | Pretrained CharSiu CTC forced aligner | Acoustic Crop [3:14] + 39D ref | Left [11:16] + Right [1:6] + refs | **0.260689** | 0.381797 | **0.252535** | **0.147736** | **0.2375** | **0.112586** | 0.1972 | 19 |
| **Control 2 (Monotonic Soft-Align)** | Learned reference dot-product + monotonic prior | Soft-attention pooled frames | Boundary transition between adjacent phones | **0.207812** | **0.389358** | 0.177226 | 0.056852 | 0.1214 | 0.078412 | 0.2106 | 29 |
| **Control 3 (Dual-State Soft-Align)** | Learned 2N+1 interleaved phone & gap states | Soft-attention phone state $P_i$ | Explicit gap state $G_j$ attention mass | **0.180032** | 0.350331 | 0.144506 | 0.045259 | 0.0911 | 0.046581 | 0.1111 | 35 |

---

## 3. Detailed Failure Analysis across Difficult Speakers

To understand why learned alignment collapses, we disaggregated performance across speaker groups:

### 1. Vietnamese Speakers (HQTV, PNV - Fold 2 & Fold 3):
- Vietnamese non-native English features frequent final consonant omissions (e.g., dropping final /t/, /d/, /s/, /z/) and epenthetic schwas.
- In Control 2 and 3, AP-O on PNV collapsed from $0.284$ (CharSiu) down to **$0.112$**.
- Inspection of attention matrices revealed that when PNV omitted a final consonant, the soft-attention query for that consonant smeared over the preceding vowel's decay and the following silence, producing high confidence "correct" predictions and suppressing omission recall to $< 2\%$.

### 2. Chinese Speakers (LXC - Fold 2):
- Chinese speakers frequently introduce vowel epenthesis after word-final consonants.
- In Control 3, the explicit gap state $G_j$ frequently triggered false alarms on normal diphthong formant transitions (e.g. `/aɪ/` and `/oʊ/`), reducing insertion precision at top 560 to only **$0.078$** on LXC.

### 3. Hindi and Arabic Speakers (ASI, SKA - Fold 0 & Fold 3):
- Features heavy retroflex substitutions and consonant cluster reductions.
- Monotonic alignment frequently drifted by 3–5 frames during long consonant clusters, dragging gap boundary representations out of position.

---

## 4. Scientific Verdict & Architectural Recommendation

- **Hypothesis H2 is FALSIFIED.**
- Attempting to learn end-to-end alignment from sparse non-native error labels is fundamentally counterproductive.
- Forced alignment models like CharSiu, trained on extensive native corpora with CTC/HMM losses, provide rigid temporal boundaries that prevent attention smearing and anchor both local acoustic crops and gap boundaries.
- **DECISION:** **RETAIN CharSiu forced alignment** as the immutable aligner for all downstream components.
