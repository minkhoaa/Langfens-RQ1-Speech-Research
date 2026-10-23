# LANGFENS — RQ1-PREP-01: Full Corpus Data Validation Report

**Execution Gate:** RQ1-PREP-01 — Pipeline Validation & Data Contract Enforcement  
**Corpus Targets:**
- L2-ARCTIC Release v5.0 (24 Non-Native Speakers, 3,599 Gold Manual Praat TextGrids)
- SpeechOcean762 (SLR101, 250 Manifested Speakers, 5,000 Utterances)  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/DATA_VALIDATION_REPORT.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`  
**Audit Mode:** STRICT READ-ONLY COMPREHENSIVE RECONCILIATION  
**Overall Validation Gate:** `PASS`

---

## 1. Executive Summary & Verification Outcomes

This report documents the exhaustive, non-destructive validation of the data preparation pipeline implemented for Langfens RQ1 (Phoneme-level Mispronunciation Detection & Diagnosis). Every manual gold annotation and audio recording was verified against the theoretical specifications defined in `RQ1-STD-01` and `CORRECTED_LABEL_ACCOUNTING.md`.

All ten core verification tests passed with **zero unexplained variance**:
1. **Representative TextGrid Parsing:** Confirmed correct extraction of tiers, intervals, and timestamps.
2. **Strict vs. Lenient Binary MDD Labels:** Primary strict ($E=17,518$) and pedagogical lenient ($E=16,406$) policies evaluated without token leakage.
3. **Error Category Separation:** Clean separation between substitutions ($14,098$), deletions ($3,420$), epenthetic additions ($1,092$), correct productions ($101,633$), and acoustic silence ($15,647$).
4. **Special Tokens & Uncertainty Handling:** Verified handling of distorted `err` tags ($258$ in substitutions, masked from diagnosis), asterisk flags ($1,472$ in substitutions), `AX` folding to `AH`, silence symbols, and keyboard slip typo `V8`.
5. **Raw Label Preservation:** Original labels and interval bounds retained untouched in record provenance.
6. **Speaker & Utterance Consistency:** All 24 L2-ARCTIC speakers and 250 SpeechOcean762 speakers verified.
7. **Audio-Annotation Alignment:** Every annotated TextGrid resolves to a valid, decodable WAV file.
8. **Partition Isolation:** Confirmed zero cross-partition overlap across all 4 folds and evaluation regimes.
9. **Numerical & Acoustic Finiteness:** Verified float32 mono waveforms, sinc bandlimited 16 kHz resampling, and absence of NaNs/Infs.
10. **Deterministic Pipeline Execution:** Confirmed bitwise identical output across repeated parser runs.

---

## 2. Exhaustive L2-ARCTIC v5.0 Gold Census Reconciliation

Parsing all 3,599 gold manual Praat TextGrids across the 24 speakers yields the definitive physical counts below:

### 2.1 Global Interval Decomposition
| Interval Category | TextGrid Syntax / Definition | Empirical Census Count | Reconciled Target (`CORRECTED_LABEL_ACCOUNTING.md`) | Variance ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Acoustic Silence Segments** | `sil`, `sp`, `spn`, `""` | **15,647** | 15,647 | 0 | **VERIFIED** |
| **Correct Phone Productions** | Single ARPAbet token (e.g., `IH0`, `K`) | **101,633** | 101,633 | 0 | **VERIFIED** |
| **Categorical Substitutions** | 3-part tag (`CPL,PPL,s`) | **14,098** | 14,098 | 0 | **VERIFIED** |
| **Phonemic Deletions** | 3-part tag (`CPL,sil,d`) | **3,420** | 3,420 | 0 | **VERIFIED** |
| **Epenthetic Additions** | 3-part tag (`sil,PPL,a`) | **1,092** | 1,092 | 0 | **VERIFIED** |
| **Total Evaluated Intervals** | **All intervals in `phones` tier** | **135,890** | **135,890** | **0** | **VERIFIED** |

$$\begin{aligned}
N_{\text{total\_intervals}} &= 15,647 + 101,633 + 14,098 + 3,420 + 1,092 = \mathbf{135,890} \\
N_{\text{canonical}} &= C + S + D = 101,633 + 14,098 + 3,420 = \mathbf{119,151} \\
N_{\text{non\_canonical}} &= \text{Silence} + \text{Additions} = 15,647 + 1,092 = \mathbf{16,739}
\end{aligned}$$

### 2.2 Mutually Exclusive Substitution Census ($S = 14,098$)
| Substitution Subcategory | Syntax Definition | Census Count | Proportion of Substitutions | Proportion of Canonical Tokens ($N$) | Strict Label ($b_i$) | Lenient Label ($b_i$) | Diagnosis Target |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Substitutions (Different Base)** | `CPL,PPL,s` ($\text{base}(C) \ne \text{base}(P)$, no `*`, no `err`) | **12,364** | 87.7004% | 10.3768% | 1 (Error) | 1 (Error) | `base(P)` |
| **Clean Substitutions (Same Base / Stress / Typo)** | `CPL,PPL,s` ($\text{base}(C) = \text{base}(P)$, no `*`, no `err`) | **4** | 0.0284% | 0.0034% | 1 (Error) | 1 (Error) | `base(P)` |
| **Base-Preserving Allophones** | `CPL,PPL*,s` ($\text{base}(C) = \text{base}(P)$) | **1,112** | 7.8876% | 0.9333% | 1 (Error) | **0 (Correct)** | Strict: `base(P)`<br>Lenient: N/A |
| **Category-Shifting Accented** | `CPL,PPL*,s` ($\text{base}(C) \ne \text{base}(P)$) | **360** | 2.5536% | 0.3021% | 1 (Error) | 1 (Error) | `base(P)` |
| **Distorted / Uncertain Realizations** | `CPL,err,s` | **258** | 1.8300% | 0.2165% | 1 (Error) | 1 (Error) | **MASKED (None)** |
| **Total Substitutions** | **All intervals with `err_type == 's'`** | **14,098** | **100.0000%** | **11.8320%** | — | — | — |

$$\mathbf{Sum\ Check:} \quad 12,364 + 4 + 1,112 + 360 + 258 = \mathbf{14,098} \quad (\Delta = 0)$$

---

## 3. Binary MDD Labeling Policy Verification

### 3.1 Primary Strict MDD Policy (Benchmark Baseline)
$$\begin{aligned}
N_{\text{canonical}} &= \mathbf{119,151} \\
C_{\text{strict}} &= \mathbf{101,633} \quad (85.2976\%) \\
E_{\text{canon, strict}} &= S_{\text{strict}} + D_{\text{strict}} = 14,098 + 3,420 = \mathbf{17,518} \quad (14.7024\%) \\
\text{Imbalance Ratio} &= \frac{101,633}{17,518} = \mathbf{5.80 : 1}
\end{aligned}$$

### 3.2 Pedagogical Lenient MDD Policy (Sensitivity Analysis)
In lenient mode, base-preserving allophones ($1,112$ tokens) are reclassified as Correct ($b_i = 0$):
$$\begin{aligned}
N_{\text{canonical}} &= \mathbf{119,151} \quad (\mathbf{STRICTLY\ INVARIANT}) \\
C_{\text{lenient}} &= 101,633 + 1,112 = \mathbf{102,745} \quad (86.2309\%) \\
E_{\text{canon, lenient}} &= (14,098 - 1,112) + 3,420 = 12,986 + 3,420 = \mathbf{16,406} \quad (13.7691\%) \\
\text{Imbalance Ratio} &= \frac{102,745}{16,406} = \mathbf{6.26 : 1}
\end{aligned}$$

### 3.3 Verification of Invariant Denominator Theorem
$$\forall \text{ Policy } p \in \{\text{Strict}, \text{Lenient}\}, \quad N_{\text{canonical}} = C_p + E_p = \mathbf{119,151} \quad (\Delta = 0)$$

---

## 4. SpeechOcean762 Partition & Isolation Verification

| Metric | Manifested Partition Count | Unmanifested Archive Exclusions | Supervised Dataset Status |
| :--- | :---: | :---: | :---: |
| **Train Utterances** | 2,000 (100 speakers) | — | Ingested |
| **Validation Utterances** | 500 (25 speakers) | — | Ingested |
| **Test Utterances (Official)** | 2,500 (125 speakers) | — | Ingested (Preserved) |
| **Total Supervised Samples** | **5,000 (250 speakers)** | — | Verified |
| **Unmanifested WAV Files** | 0 | **245 files** | **STRICTLY EXCLUDED** |
| **Cross-Partition Overlap** | **0** | **0** | **ZERO LEAKAGE** |

All 5,000 manifested utterances load without missing audio references or label corruption. Continuous scores (accuracy, completeness, fluency, prosody, total) are preserved intact without fabricated discrete substitution classes.

---

## 5. Verification Gate Status

- `ANNOTATION_PARSER_VALIDATED`: **PASS**
- `LABEL_SEMANTICS_VALIDATED`: **PASS**
- `AUDIO_PIPELINE_VALIDATED`: **PASS**
- `MANIFEST_ISOLATION_VALIDATED`: **PASS**
- `DATALOADER_VALIDATED`: **PASS**
- `DATA_CONTRACT_VALIDATION`: **PASS**
