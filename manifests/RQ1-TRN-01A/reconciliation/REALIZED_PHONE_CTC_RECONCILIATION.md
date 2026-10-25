# LANGFENS — RQ1-TRN-01A-R: Realized-Phone CTC Target Reconciliation Report

**Execution Gate:** RQ1-TRN-01A-R — Pretrained Backbone & Inference Validity Gate  
**Deliverable Associated Path:** `manifests/RQ1-TRN-01A/reconciliation/REALIZED_PHONE_CTC_RECONCILIATION.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/`  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & PyTorch Architect  
**Gate Evaluation Status:** `REALIZED_CTC_RECONCILIATION: PASS`  

---

## 1. Executive Summary & Code Review Findings

In `RQ1-TRN-01A`, CTC targets were constructed by extracting `realized_phone` solely from `canonical_intervals`. This introduced three fundamental phonetic defects:
1. **Omission of Epenthetic Additions:** Unprompted additions (tagged `[sil,PPL,a]`) were excluded because they lacked canonical token indices, failing to supervise genuine acoustic emissions.
2. **Silent Inclusion of Deletions:** In canonical intervals where a sound was deleted (`[CPL,sil,d]`), the interval was processed, which risked emitting silence or bogus tokens.
3. **Absence of Chronological Ordering:** Non-canonical events were dropped, breaking the true temporal alignment of the acoustic waveform.

To resolve this, we performed a complete chronological reconstruction of realized phone targets across all 3,599 Gold TextGrids of L2-ARCTIC v5.0.

---

## 2. Chronological Reconstruction Rules

When building the realized-phone CTC supervision target for an utterance:
1. **Physical Silence:** Traversed intervals with `error_type == "silence"` are **strictly excluded**. Silence is not an active phoneme in the standard 39-ARPAbet CTC vocabulary.
2. **Acoustic Deletions:** Intervals with `error_type == "deletion"` are **strictly excluded**. The speaker produced silence or skipped the sound; there is no acoustic phone to supervise in the CTC head.
3. **Correct Productions:** Intervals with `error_type == "correct"` are **included** with their normalized 39-ARPAbet symbol.
4. **Substitutions:** Clean substitutions with valid 39-ARPAbet realized phones are **included**. Substitutions tagged `'err'` (uncertain / unidentifiable) are **excluded** to prevent fabricating arbitrary phoneme targets.
5. **Epenthetic Additions (Insertions):** Valid 39-ARPAbet realized additions are **included** in their true chronological position.
6. **CTC Blank Token Constraint:** Token ID $0$ is strictly reserved for `<BLANK>` and **never** enters the target sequence. All valid targets reside strictly in $[1, 39]$.

---

## 3. Full Census Reconciliation Table (3,599 Utterances)

Audited over all 24 non-native speakers in L2-ARCTIC v5.0 using `audit_scripts/audit_ctc_realized_accounting.py`:

| Interval Category | Event Count | CTC Supervision Action | Justification |
| :--- | :--- | :--- | :--- |
| **Total Raw Intervals** | **135,890** | Traversed chronologically | Complete inventory of all annotated segments |
| **Physical Silence** | **15,647** | **EXCLUDED** | Non-speech pause; excluded from 39-ARPAbet phone vocabulary |
| **Acoustic Deletions** | **3,420** | **EXCLUDED** | Omitted by speaker; no acoustic realization exists |
| **Canonical Correct** | **101,633** | **INCLUDED** | Valid standard 39-ARPAbet phone emission |
| **Clean Substitutions ($S_{\text{eval}}$)** | **13,840** | **INCLUDED** | Genuine non-native substitution with identifiable ARPAbet symbol |
| **Masked `'err'` Substitutions** | **258** | **EXCLUDED** | Listener uncertainty; ground-truth phone unknown; fabrication prohibited |
| **Clean Epenthetic Additions** | **1,081** | **INCLUDED** | Genuine inserted acoustic phone (e.g. epenthetic stop or vowel) |
| **Masked `'err'` Additions** | **11** | **EXCLUDED** | Unidentifiable insertion; ground-truth phone unknown |
| **Non-Standard Typo Normalizations** | **8** | **NORMALIZED** | 8 cataloged typos (`ER)`, `D_`, `W```, `V`````, `Z_`, `Y_`, `AH)`, `V8`) normalized to standard 39 ARPAbet |
| **Total Chronological CTC Targets** | **116,554** | **ACTIVE SUPERVISION** | $101,633 + 13,840 + 1,081 = \mathbf{116,554}$ tokens |

---

## 4. Mathematical Invariance & Verification Checks

### 4.1 Canonical Denominator Invariance Check
$$N_{\text{canonical}} = C + S + D = 101,633 + 14,098 + 3,420 = \mathbf{119,151} \quad (\text{Strict Invariance Preserved})$$

### 4.2 Exact Token Accounting Identity
$$\text{Total CTC Targets} = C + S_{\text{clean}} + A_{\text{clean}} = 101,633 + 13,840 + 1,081 = \mathbf{116,554} \quad (\text{Exact Match})$$

### 4.3 CTC Feasibility & Blank Integrity
- **Blank ID in Active Target Sequence:** **0 violations** (Zero blank tokens present).
- **Target Sequence Length Range:** $[9, 72]$ tokens per utterance.
- **Acoustic Frame vs Target Feasibility:** For all 3,599 utterances, downsampled acoustic frames $T_{\text{frames}} \ge L_{\text{target}} + \text{repeats}$, satisfying CTC alignment mathematical preconditions.
