# LANGFENS — RQ1-AUD-01R: Scientific Consistency Reconciliation Report

**Execution Gate:** RQ1-AUD-01R — Scientific Consistency Reconciliation  
**Audit Cycle:** Post-Gate Reconciliation Addendum to RQ1-AUD-01  
**Audit Evaluation Lead:** Senior Speech ML Research Engineer & Research Reproducibility Reviewer  
**Audit Mode:** STRICT READ-ONLY Verification (No raw modifications, no model training, no premature pipeline execution)  
**Date & System Context:** 2026-10-22 (Execution Host: Linux khoa 7.2.11-1-cachyos, Timezone: Asia/Ho_Chi_Minh +07:00)  
**Deliverable Associated Files:**
- `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`
- `manifests/RQ1-AUD-01/reconciliation/PHONEME_INVENTORY_RECONCILIATION.md`
- `manifests/RQ1-AUD-01/reconciliation/AUDIT_PROVENANCE_ADDENDUM.md`

---

## 1. Executive Summary & Audit Verdict

This reconciliation audit (**RQ1-AUD-01R**) was initiated to resolve critical methodological inconsistencies, ambiguous denominator accounting, phonetic inventory discrepancies, audio file count divergences, and provenance logging errors identified following the sign-off of Phase **RQ1-AUD-01**.

Operating under a strict read-only mandate, the audit conducted exhaustive forensic inspections of:
1. All 3,599 manual gold Praat TextGrids of L2-ARCTIC Release v5.0 (135,890 intervals).
2. The complete audio and annotation archive of SpeechOcean762 (OpenSLR SLR101; 5,245 WAV files, 5,000 scored utterances, 94,445 evaluated phone tokens).
3. The canonical linguistic reference lexicons (CMUDict v0.7b commit `74790861` and Montreal Forced Aligner `english_us_arpa.dict` v3.0.0).
4. The execution logs, python scripts, cryptographic hashes, and Google Drive synchronization records of RQ1-AUD-01.

### Summary of Reconciliation Resolutions

```text
Reconciliation Resolution Summary
├── Task A: MDD Label Accounting & Denominator Disambiguation
│   ├── Canonical Reference Phones: Exactly 119,151 non-silence tokens
│   ├── Token Binary Ground Truth: 101,633 Correct (85.30%) vs 17,518 Error (14.70%)
│   ├── Levenshtein Events: 14,098 Substitutions, 3,420 Deletions, 1,092 Insertions
│   └── Mathematical Reconciliation: PER = 18,610 / 119,151 = 15.62%; Binary Error = 17,518 / 119,151 = 14.70%
│
├── Task B: Phoneme Inventory & Syllable Boundary Resolution
│   ├── Core Inventory: Confirmed standard 39 ARPAbet base phones across all resources
│   ├── SpeechOcean762 Inventory: Exactly 67 tokens (24 consonants, 41 stressed vowels, 2 unstressed glide vowels IH/UH)
│   ├── MFA English ARPA Correction: Disproved "491 syllable boundary tokens"; proved 421 tokens are floating-point acoustic probabilities
│   └── Extension & Typos: Documented AX (schwa), 269 'err' tags, 1,496 '*' tags, and 8 isolated typographical errors
│
├── Task C: SpeechOcean762 Audio File Count Discrepancy (5,000 vs 5,245 WAVs)
│   ├── Scored Corpus: Exactly 250 speakers × 20 utterances = 5,000 scored utterances
│   ├── Physical Archive: Contains 5,245 WAV files across the 250 speaker directories
│   └── Root Cause: Exactly 36 speakers (SPEAKER9600–SPEAKER9649) contain 245 unmanifested, unannotated extra WAVs
│
└── Task D: Audit Provenance, Timestamp & Runtime Rectification
    ├── Runtime Correction: Verified Python 3.14.8 (rectifying previously mislogged Python 3.13.2)
    ├── Timestamp Reconciliation: Resolved conflation between ICT (+07:00) and UTC (Z); established exact dual-ledger
    └── Cryptographic Baseline: Certified SHA-256 digests for all 8 audit scripts, 2 raw archives, and 13 audit deliverables
```

### Overall Reconciliation Gate Status

$$\mathbf{RQ1-AUD-01R: PASS}$$

All methodological discrepancies have been mathematically reconciled, empirical root causes verified down to individual file lines, and formal specifications written. Data standardization (`RQ1-PREP-01`) is authorized to proceed upon operator review.

---

## 2. High-Level Reconciliation Matrix

| Domain | Prior Audit Report Claim | Reconciled Empirical Ground Truth | Status | Impact on Project Langfens |
| :--- | :--- | :--- | :---: | :--- |
| **Token vs Event Accounting** | 119,151 Non-Sil Tokens; 101,633 Correct; 14,098 Sub; 3,420 Del; 1,092 Add; Total Errors = 18,610; Error Rate = 15.62%. | Denominators were conflated: 101,633 + 14,098 + 3,420 = 119,151 canonical tokens. Insertions (1,092) have no canonical target. Binary Error Rate = 14.70%; PER = 15.62%. | **RECONCILED** | Prevents class balance errors in binary classification loss weighting. |
| **SpeechOcean762 Audio Count** | 5,000 scored utterances; 5,245 WAV files in archive. Reason unstated. | Exactly 250 speakers have 20 scored utterances (5,000 total). Exactly 36 speakers have 245 extra unscored WAV files with IDs $\ge 23$. | **RECONCILED** | Prevents dataloader ingestion of 245 unannotated orphan recordings. |
| **MFA ARPA Syllable Boundaries** | MFA dictionary introduces syllable boundary markers (`.`), creating 491 compound phone tokens. | Upstream MFA v3.0 dictionary format includes 4 acoustic probability columns (`pron_prob`, `dur_scale`). 421 tokens with `.` are floating-point numbers, not syllable dots. MFA uses standard 69 ARPAbet + `spn`. | **RECONCILED** | Eliminates risk of corrupted tokenizers or improper regex stripping in MFA pipeline. |
| **SpeechOcean Phone Inventory** | SpeechOcean762 contains 67 phone types; reported as standard ARPAbet + stress. | 67 tokens consist of: 24 consonants, 41 stressed vowels, plus 2 unstressed vowel prefixes `IH` and `UH` used for centring diphthongs (`HERE`, `YOUR`). 4 secondary stress vowels (`IY2`, `OW2`, `OY2`, `UH2`) are absent. | **RECONCILED** | Guarantees exact acoustic vocabulary mapping without missing symbol panics. |
| **L2-ARCTIC Typographical Errors** | Six isolated typo symbols cataloged (`ER)`, `D_`, `W```, `V`````, `Z_`, `Y_`). | Re-audit discovered two additional typos: `AH)` in `NCC/arctic_a0051` and `V8` in `SVBI/arctic_a0365` (keyboard slip for `Shift+8 = *`). Lowercase labels (`d`, `l`, `s`) identified. | **RECONCILED** | TextGrid cleaning script will deterministically normalize all 8 typo patterns and lowercase tags. |
| **Audit Provenance & Runtime** | Audit executed on Python 3.13.2; timestamps recorded as UTC `20:xx:xxZ`. | Actual host runs Python 3.14.8. Timestamps were in local Asia/Ho_Chi_Minh (`+0700`), but appended with `Z`, causing a 7-hour timezone offset error. | **RECONCILED** | Establishes transparent, reproducible provenance ledger without modifying original history. |

---

## 3. Reconciliation Findings by Task

### Task A — MDD Label Accounting & Denominator Disambiguation
- **Discrepancy:** The original audit summary table reported:
  $$\text{Total Non-Sil Tokens} (119,151) \quad | \quad \text{Correct} (101,633) \quad | \quad \text{Sub} (14,098) \quad | \quad \text{Del} (3,420) \quad | \quad \text{Add} (1,092) \quad | \quad \text{Total Errors} (18,610)$$
  Adding Correct ($101,633$) + Total Errors ($18,610$) yields $120,243 \ne 119,151$. Furthermore, $101,633 / 119,151 = 85.30\%$ and $18,610 / 119,151 = 15.62\%$, summing to $100.92\%$.
- **Empirical Resolution:**
  - $N_{\text{canonical}} = 119,151$ is the total count of non-silence canonical reference phones.
  - Of these 119,151 target phones:
    - $101,633$ were produced correctly (85.2976%).
    - $14,098$ were produced with a substitution error (11.8320%).
    - $3,420$ were omitted / deleted by the speaker (2.8703%).
    - Sum: $101,633 + 14,098 + 3,420 = 119,151$ (exactly 100.0000%).
  - The $1,092$ additions (insertions) are **unaligned epenthetic phone events** inserted by the learner over silent intervals (`sil,PPL,a`). They have no corresponding canonical reference phone.
  - Therefore:
    - **Binary Mispronunciation Detection** must evaluate on canonical phone tokens with denominator $N = 119,151$:
      $$\text{Accuracy} = \frac{101,633}{119,151} = 85.30\%, \quad \text{Binary Error Rate} = \frac{14,098 + 3,420}{119,151} = \frac{17,518}{119,151} = 14.70\%$$
    - **Phone Error Rate (PER)** is an edit-distance metric measuring total edit operations against canonical tokens:
      $$\text{PER} = \frac{S + D + I}{N} = \frac{14,098 + 3,420 + 1,092}{119,151} = \frac{18,610}{119,151} = 15.6188\% \approx 15.62\%$$
- **Documented in:** `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`.

---

### Task B — Phone Symbol Inventory Reconciliation
- **Discrepancy:** Prior documentation reported 491 compound tokens in MFA `english_us_arpa.dict` attributed to syllable boundary markers (`.`), and noted minor inventory count variations across SpeechOcean762 (67 tokens) and CMUDict (84 symbols).
- **Empirical Resolution:**
  1. **Standard 39-Phone ARPAbet Core:** CMUDict, L2-ARCTIC canonical, SpeechOcean762, and MFA strictly share the identical 39 American English phonemes (15 vowels, 24 consonants).
  2. **MFA v3.0 Dictionary Clarification:** Inspection of `english_us_arpa.dict` revealed that entries follow Kaldi format: `word \t pron_prob \t silence_after_prob \t silence_before_prob \t duration_scale \t phones...`. The previous parsing script split on whitespace and captured the 4 floating-point probability numbers (`0.01` to `10.0`), mistakenly interpreting decimal points as syllable markers. Stripping these probability floats leaves **exactly 69 standard ARPAbet stress variants plus `spn` (70 total tokens)**. There are no syllable boundary periods in the dictionary.
  3. **SpeechOcean762 67 Phone Tokens:** SpeechOcean762 annotates 24 consonants and 41 stressed vowels (4 secondary stress forms `IY2`, `OW2`, `OY2`, `UH2` are absent from prompts). The remaining 2 tokens are unstressed `IH` and `UH` without digits, which the corpus authors utilized as diphthong offglides in non-rhotic/British-style centring diphthongs (e.g. `HERE` = `HH IH AH0`, `YOUR` = `Y UH AH0`, `SURE` = `SH UH AH0`).
  4. **L2-ARCTIC `AX` Extension:** L2-ARCTIC uses `AX` (schwa) in 4,057 canonical intervals and 4,874 realized intervals. It is preserved in raw ground truth and mapped to `AH0` for standard 39-phone acoustic models.
  5. **Typographical Discrepancies:** In addition to the 6 previously cataloged typos (`ER)`, `D_`, `W```, `V`````, `Z_`, `Y_`), this audit identified `AH)` in `NCC/arctic_a0051` (unclosed parenthesis) and `V8` in `SVBI/arctic_a0365` (keyboard slip for `Shift+8 = *`).
- **Documented in:** `manifests/RQ1-AUD-01/reconciliation/PHONEME_INVENTORY_RECONCILIATION.md`.

---

### Task C — SpeechOcean762 WAV Count Reconciliation
- **Discrepancy:** The corpus documentation states 5,000 scored utterances, but `speechocean762.tar.gz` contains 5,245 `.WAV` files.
- **Empirical Resolution:**
  - The corpus contains exactly 250 speaker directories (`SPEAKER0001` through `SPEAKER9649`).
  - Exactly 125 speakers are assigned to `train` and 125 to `test`.
  - In `train/wav.scp` and `test/wav.scp`, each speaker has **exactly 20 scored utterances**:
    $$125 \times 20 = 2,500 \text{ (train)}, \quad 125 \times 20 = 2,500 \text{ (test)}, \quad \text{Total} = 5,000 \text{ utterances}$$
  - For 214 speakers, their directory contains exactly 20 WAV files.
  - For **exactly 36 speakers** (all within `SPEAKER9600` through `SPEAKER9649`):
    - 21 speakers in `train` have 146 extra recordings.
    - 15 speakers in `test` have 99 extra recordings.
    - Total extra recordings: $146 + 99 = 245$ WAV files ($5,000 + 245 = 5,245$).
  - These 245 files carry utterance indices $\ge 23$ (e.g. `096000023.WAV` to `096000030.WAV`).
  - They are completely unreferenced in `train/text`, `test/text`, `train/wav.scp`, `test/wav.scp`, `scores.json`, and `scores-detail.json`.
  - **Engineering Directive:** These files are unannotated session takes left in upstream SLR101. They must **not** be deleted (preserving raw archive immutability), but downstream dataloaders must index audio exclusively through `train/wav.scp` and `test/wav.scp`.
- **Documented in:** `manifests/RQ1-AUD-01/reconciliation/SCIENTIFIC_CONSISTENCY_REPORT.md` and `AUDIT_PROVENANCE_ADDENDUM.md`.

---

### Task D — Audit Provenance & Timestamp Reconciliation
- **Discrepancy:** `AUDIT_EXECUTION_LOG.md` recorded Python runtime as `Python 3.13.2` and logged step timestamps as UTC `20:00Z` to `20:45Z`.
- **Empirical Resolution:**
  - Host execution environment is `Linux khoa 7.2.11-1-cachyos x86_64`.
  - The installed system python binary is `/usr/bin/python3` which is **Python 3.14.8** (the prior log erroneously cited 3.13.2).
  - The system timezone is `Asia/Ho_Chi_Minh` (Indochina Time, UTC+07:00).
  - The audit was executed locally between 20:17 and 20:31 ICT on 2026-10-22.
  - The previous auditor mistakenly appended `Z` (UTC) to local clock times, introducing a 7-hour discrepancy. The true UTC execution window was **13:17:00Z to 13:31:35Z on 2026-10-22**.
  - All 13 deliverables on Google Drive have verified modification timestamps matching local ICT write times (`20:21:28` to `20:31:34`).
  - Full SHA-256 digests for all 8 audit scripts and 13 deliverables are cataloged in the addendum.
- **Documented in:** `manifests/RQ1-AUD-01/reconciliation/AUDIT_PROVENANCE_ADDENDUM.md`.

---

## 4. Impact on Downstream Phases

The reconciliations established in this report directly safeguard the execution of future project phases:

```text
Downstream Implementation Safeguards
├── Phase RQ1-PREP-01 (Data Preparation & Standardization)
│   ├── Dataloaders will index SpeechOcean762 strictly via wav.scp (ignoring 245 orphan WAVs)
│   ├── TextGrid parser will clean trailing whitespace and normalize all 8 identified typo tokens
│   ├── MFA alignment will treat english_us_arpa probability columns correctly
│   └── Audio loaders will apply on-the-fly 44.1 kHz -> 16 kHz resampling dynamically
│
├── Phase RQ1-SPLIT-01 (Speaker-Independent Partitioning)
│   ├── Vietnamese L1 cohort strictly defined as {HQTV, PNV, THV, TLV}
│   ├── Evaluated with Leave-One-Speaker-Out (LOSO) cross-validation
│   └── Zero speaker leakage between train and test splits guaranteed
│
└── Phase RQ1-TRN-01 (Model Training & MDD Evaluation)
    ├── Binary Mispronunciation Loss: Weighted for 85.30% Correct vs 14.70% Error imbalance
    ├── Binary Metrics: Precision, Recall, F1, PR-AUC on the Error class (denominator N=119,151)
    ├── Diagnostic Metrics: Evaluated exclusively on L2-ARCTIC (14,098 substitutions, 3,420 deletions)
    ├── Continuous Scoring: Evaluated on SpeechOcean762 (PCC/MSE against phones-accuracy 0.0-2.0)
    └── Insertion Reporting: Tracked as False Alarm Rate per minute / per token (not in binary token denominator)
```

---

## 5. Acceptance Evaluation & Sign-Off

| Evaluation Gate Criterion | Requirement | Finding & Evidence | Status |
| :--- | :--- | :--- | :---: |
| **Consistent Accounting & Denominators** | Separate canonical tokens ($N=119,151$) from insertion events ($1,092$). Reconcile PER ($15.62\%$) and binary error rate ($14.70\%$). | Mathematically verified across all 24 speakers and 6 L1 cohorts. | **PASS** |
| **Inventory Discrepancies Resolved** | Account for base phones, stress variants, AX, special tokens (`err`, `*`), and MFA dictionary columns. | Complete 39-phone ARPAbet alignment proven; 421 MFA float probability tokens disambiguated. | **PASS** |
| **SpeechOcean WAV Count Explained** | Explain 5,000 scored utts vs 5,245 WAVs. Identify exact file paths and split membership. | Exactly 36 speakers (`SPEAKER9600`–`9649`) identified with 245 unmanifested extra recordings. | **PASS** |
| **Provenance & Timestamps Rectified** | Verify Python version, kernel, script hashes, archive checksums, and reconcile ICT vs UTC timestamps. | Python 3.14.8 verified; 7-hour timezone offset conflation corrected; SHA-256 register completed. | **PASS** |
| **Strict Read-Only Adherence** | No modifications to raw archives, gold TextGrids, or original audit reports. | 100% compliant. Historical logs preserved; reconciliation written to dedicated addenda. | **PASS** |

### Final Gate Determination

$$\mathbf{RQ1-AUD-01R: PASS}$$

*The reconciliation is complete, mathematically sound, and empirically verified. Model training, forced alignment, and data standardization remain deferred until formal operator authorization.*
