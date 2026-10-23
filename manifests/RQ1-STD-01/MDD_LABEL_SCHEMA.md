# LANGFENS — RQ1-STD-01: MDD Label Accounting & Semantics Schema

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Task A — Finalize Label Semantics, Accounting Denominators & Ambiguity Resolution  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Phonetics Specialist  
**Corpus Target:** L2-ARCTIC Release v5.0 (3,599 Gold Manual Praat TextGrids across 24 Speakers)  
**Deliverable Associated Path:** `manifests/RQ1-STD-01/MDD_LABEL_SCHEMA.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`
- `manifests/RQ1-AUD-01/reconciliation/PHONEME_INVENTORY_RECONCILIATION.md`
- `manifests/RQ1-AUD-01/reconciliation/SCIENTIFIC_CONSISTENCY_REPORT.md`
- `manifests/RQ1-AUD-01/L2_ARCTIC_ANNOTATION_AUDIT.md`

---

## 1. Executive Overview & Schema Scope

Mispronunciation Detection and Diagnosis (MDD) algorithms must map continuous acoustic speech frames into structured linguistic decisions. In non-native speech corpora—specifically L2-ARCTIC v5.0—annotators recorded a complex continuum of phonetic behaviors: flawless productions, regional allophonic colorings, categorical phonemic substitutions, outright omissions, epenthetic additions, unintelligible distortions, and human transcription slips.

This document formally specifies:
1. An unambiguous structural schema categorizing all interval annotations.
2. The definitive scientific resolution of the `PPL*` (asterisk tag) accounting contradiction identified in earlier reports.
3. Strict vs. Lenient operational labeling policies, with mathematically verified invariant denominators.
4. Formal masking contracts for uncertain realized phones (`err`).
5. Independent acoustic event handling for epenthetic insertions ($I = 1,092$).
6. Rigorous policies for typographical corrections, explicitly designating the `V8 → V*` interpretation as provisional.

---

## 2. Comprehensive Phone Annotation Taxonomy

Every time interval in the gold Praat TextGrids (`phones` tier) belongs to exactly one category in the following hierarchical taxonomy:

```text
L2-ARCTIC Annotation Taxonomy (135,890 Total Evaluated Intervals)
├── 1. Acoustic Silence Segments (16,739 intervals)
│   ├── Leading, trailing, and inter-word pauses ('sil', 'sp', 'spn', '""')
│   └── Excluded from all canonical phone evaluations
│
└── 2. Canonical Phone Evaluations (119,151 non-silence prompt tokens)
    ├── A. Correct Productions (101,633 tokens; 85.30%)
    │   └── Single ARPAbet symbol matching canonical prompt target (e.g., 'IH0', 'K', 'S')
    │
    ├── B. Categorical Phonemic Substitutions (14,098 tokens; 11.83%)
    │   ├── Standard 3-part tag: 'CPL,PPL,s' (11,048 tokens; e.g., 'DH,D,s', 'ER,AH,s')
    │   ├── Accented / Allophonic realization: 'CPL,PPL*,s' (1,770 tokens total)
    │   │   ├── Base-Preserving Allophone: 'CPL,CPL*,s' (1,281 tokens; e.g., 'R,R*,s', 'T,T*,s')
    │   │   └── Category-Shifting Accented: 'CPL,PPL*,s' (489 tokens; e.g., 'AE1,AA*,s', 'V,B*,s')
    │   └── Unintelligible / Distorted Substitution: 'CPL,err,s' (269 tokens; e.g., 'R,err,s')
    │
    ├── C. Phonemic Deletions (3,420 tokens; 2.87%)
    │   └── 3-part tag located at omitted interval: 'CPL,sil,d' (e.g., 'T,sil,d', 'D,sil,d')
    │
    └── D. Epenthetic Acoustic Insertions (1,092 events corpus-wide; NOT IN CANONICAL DENOMINATOR)
        ├── Standard phone insertion over silence: 'sil,PPL,a' (1,088 events; e.g., 'sil,AH0,a')
        └── Unintelligible insertion over silence: 'sil,err,a' (4 events)
```

### Detailed Formal Interval Schema

| Category | Raw TextGrid Syntax | Canonical Token Exists? | Primary Binary Label ($b_i$) | Realized Target for CTC | Diagnosis Evaluation Target |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Silence** | `sil`, `sp`, `spn`, `""` | No (Acoustic pause) | Excluded | `<SIL>` or `<BLANK>` | N/A |
| **Correct Phone** | `PHONE` (e.g., `K`, `IY1`) | Yes ($y_i = \text{PHONE}$) | **Correct (0)** | Base phone | N/A (True Negative) |
| **Clean Substitution** | `CPL,PPL,s` | Yes ($y_i = \text{CPL}$) | **Error (1)** | `PPL` (base phone) | `PPL` (39-phone vocabulary) |
| **Base-Preserving Allophone** | `CPL,CPL*,s` | Yes ($y_i = \text{CPL}$) | Strict: **Error (1)**<br>Lenient: **Correct (0)** | `CPL` (base phone) | Strict: `CPL`<br>Lenient: N/A |
| **Cross-Category Accented** | `CPL,PPL*,s` ($C \ne P$) | Yes ($y_i = \text{CPL}$) | **Error (1)** (Both policies) | `PPL` (base phone) | `PPL` (base phone) |
| **Uncertain Realization** | `CPL,err,s` | Yes ($y_i = \text{CPL}$) | **Error (1)** | `<UNK_PHONE>` | **Masked** (Excluded from 39-class) |
| **Phoneme Deletion** | `CPL,sil,d` | Yes ($y_i = \text{CPL}$) | **Error (1)** | `<SIL>` / Blank | `<SIL>` (Deletion target) |
| **Epenthetic Insertion** | `sil,PPL,a` | No (Between tokens) | N/A (Event task) | `PPL` (inserted token) | Task 4 (Windowed Event Detection) |
| **Uncertain Insertion** | `sil,err,a` | No (Between tokens) | N/A (Event task) | `<UNK_PHONE>` | Task 4 (Windowed Event Detection) |
| **Typographical Error** | e.g., `ER)`, `V8`, `D_` | Yes | Governed by rule | Normalized base phone | Normalized base phone |

---

## 3. Resolution of the `PPL*` Accounting Contradiction

### The Forensic Contradiction Identified in RQ1-AUD-01R
During the Phase B/C audit reconciliation, a severe semantic ambiguity was documented in `MDD_LABEL_ACCOUNTING_SPEC.md`:
- **Section 2 (Task 1 Ground Truth Grounding):** Stated that $b_i = 0 \iff$ interval label is a single ARPAbet phone *or accented phone* `PPL*`. Under this phrasing, all 1,770 asterisk tokens would be classified as **Correct ($b_i = 0$)**.
- **Section 3 (Special Token Taxonomy):** Stated that `CPL,PPL*,s` is *Strict: Mispronounced (1)* vs *Lenient: Correct (0)*.
- **Section 1 & 4 (Ledger Totals):** Computed $S = 14,098$ and $C = 101,633$. In the audit parser (`audit_scripts/audit_l2arctic_annotations.py`), every interval where `err_type == 's'` was incremented into `sub_count` ($14,098$). Because all 1,770 asterisk annotations have `err_type == 's'`, **they were counted 100% as substitutions (Errors, $b_i = 1$) in the primary corpus ledger**.

### Physical Evidence Breakdown of Asterisk Tags
Exhaustive re-parsing of all 3,599 gold TextGrids reveals that the 1,770 asterisk annotations partition into two phonetically distinct cohorts:

$$\begin{aligned}
N_{\text{asterisk}} &= \mathbf{1,770} \quad \text{tokens across 24 speakers} \\
N_{\text{same\_base}} &= \mathbf{1,281} \quad \text{tokens } (72.37\%) \quad \text{where } \text{base}(\text{CPL}) = \text{base}(\text{PPL}) \\
N_{\text{diff\_base}} &= \mathbf{489} \quad \text{tokens } (27.63\%) \quad \text{where } \text{base}(\text{CPL}) \ne \text{base}(\text{PPL})
\end{aligned}$$

#### Top Base-Preserving Patterns ($N = 1,281$)
Annotators explicitly judged that the learner produced the target phoneme, but with a perceptible foreign allophone:
- `R → R*,s`: 468 occurrences (e.g., alveolar trill [r] or flap [ɾ] instead of English retroflex [ɹ]).
- `T → T*,s`: 212 occurrences (e.g., dentalized [t̪], unreleased [t̚], or heavily aspirated [tʰ]).
- `HH → HH*,s`: 99 occurrences (e.g., velar/pharyngeal fricative [x] instead of glottal [h]).
- `D → D*,s`: 89 occurrences (e.g., dentalized [d̪]).
- `L → L*,s`: 72 occurrences (e.g., clear /l/ in syllable coda instead of dark /ɫ/).
- `N → N*,s`: 56 occurrences; `ER0 → ER*,s`: 49 occurrences; `P → P*,s`: 34 occurrences; `W → W*,s`: 25 occurrences; `V → V*,s`: 24 occurrences.

#### Top Cross-Category Patterns ($N = 489$)
Annotators judged that the learner produced a **different English phoneme**, which additionally had non-native coloring:
- `AE1 → AA*,s`: 87 occurrences (Target /æ/ replaced by accented open-back /ɑ/).
- `AH1 → AA*,s`: 81 occurrences (Target /ʌ/ replaced by accented open-back /ɑ/).
- `ER0 → R*,s`: 35 occurrences (Syllabic rhotic vowel reduced to consonantal trill/flap).
- `AH0 → AA*,s`: 33 occurrences; `AH0 → OW*,s`: 28 occurrences; `V → B*,s`: 27 occurrences (Voiced labiodental fricative replaced by bilabial stop /b/).

---

## 4. Primary vs. Sensitivity MDD Labeling Policies

To prevent silent reclassification while supporting pedagogical and linguistic analyses, Project Langfens establishes two formal labeling policies:

```text
Binary MDD Labeling Policies
├── Policy 1: Primary Strict Policy (Corpus Default & Baseline Comparable)
│   ├── Every interval tagged with 's' or 'd' is classified as Error (b_i = 1)
│   ├── All 1,770 asterisk tokens are treated as Errors (Substitutions)
│   └── Directly matches standard MDD literature (Peng et al., 2021; Tu et al., 2025)
│
└── Policy 2: Pedagogical Lenient Sensitivity Policy (Allophonic Acceptance)
    ├── Base-preserving allophones (CPL,CPL*,s; 1,281 tokens) reclassified as Correct (b_i = 0)
    ├── Cross-category substitutions (CPL,PPL*,s; 489 tokens) remain Errors (b_i = 1)
    └── Evaluates model sensitivity to phonemic boundaries vs allophonic accent
```

### Mathematical Formulation & Reconciled Class Counts

#### Policy 1: Primary Strict Policy (Formal Standard)

$$\begin{aligned}
N_{\text{canonical}} &= 119,151 \\
C_{\text{strict}} &= 101,633 \quad (85.2976\% \approx 85.30\%) \\
S_{\text{strict}} &= 14,098 \quad (11.8320\% \approx 11.83\%) \\
D_{\text{strict}} &= 3,420 \quad (2.8703\% \approx 2.87\%) \\
E_{\text{canon, strict}} &= S_{\text{strict}} + D_{\text{strict}} = 14,098 + 3,420 = \mathbf{17,518} \quad (14.7024\% \approx 14.70\%) \\
\text{Imbalance Ratio} &= \frac{C_{\text{strict}}}{E_{\text{canon, strict}}} = \frac{101,633}{17,518} = \mathbf{5.80 : 1}
\end{aligned}$$

#### Policy 2: Pedagogical Lenient Policy (Sensitivity Benchmark)
Under Policy 2, the 1,281 base-preserving allophones are reclassified as Correct ($b_i = 0$), acknowledging that the speaker achieved the intended phonemic contrast:

$$\begin{aligned}
N_{\text{canonical}} &= 119,151 \quad (\mathbf{STRICTLY\ INVARIANT}) \\
C_{\text{lenient}} &= 101,633 + 1,281 = \mathbf{102,914} \quad (86.3728\% \approx 86.37\%) \\
S_{\text{lenient}} &= 14,098 - 1,281 = \mathbf{12,817} \quad (10.7569\% \approx 10.76\%) \\
D_{\text{lenient}} &= 3,420 \quad (2.8703\% \approx 2.87\%) \\
E_{\text{canon, lenient}} &= S_{\text{lenient}} + D_{\text{lenient}} = 12,817 + 3,420 = \mathbf{16,237} \quad (13.6272\% \approx 13.63\%) \\
\text{Imbalance Ratio} &= \frac{C_{\text{lenient}}}{E_{\text{canon, lenient}}} = \frac{102,914}{16,237} = \mathbf{6.34 : 1}
\end{aligned}$$

#### Policy 3: Ultra-Lenient Variant (Exploratory / All-Asterisks Accepted)
For comprehensive sensitivity analysis, if an application accepts all 1,770 asterisk annotations as acceptable non-native speech:

$$\begin{aligned}
C_{\text{ultra}} &= 101,633 + 1,770 = \mathbf{103,403} \quad (86.7832\% \approx 86.78\%) \\
E_{\text{canon, ultra}} &= 17,518 - 1,770 = \mathbf{15,748} \quad (13.2168\% \approx 13.22\%) \\
S_{\text{ultra}} &= 14,098 - 1,770 = \mathbf{12,328} \quad (10.3465\% \approx 10.35\%)
\end{aligned}$$

### Invariant Denominator Proof
Across all three policies, the canonical reference denominator is **strictly invariant**:

$$\forall \text{Policy } p \in \{\text{Strict}, \text{Lenient}, \text{Ultra}\}, \quad N_{\text{canonical}} = C_p + S_p + D_p = \mathbf{119,151}$$

*Proof:* Reclassification operations are pure permutations between sets $C$ and $S$. No tokens are added, deleted, or removed from the prompt sequence.

---

## 5. Treatment of Epenthetic Insertions ($I = 1,092$)

### Why Insertions Cannot Enter the Canonical Denominator
In binary MDD, the task is formal token-level classification:
$$\forall y_i \in Y = (y_1, y_2, \dots, y_M), \quad \hat{b}_i \in \{0, 1\}$$
- Canonical reference sentences consist strictly of $N_{\text{canonical}} = 119,151$ prompt phones.
- Epenthetic insertions occur in the **acoustic speech signal over silence intervals** between reference phones (e.g. inserting an epenthetic schwa: `/s/ + [ə] + /p/` for *spoon*).
- There is **no $i$-th prompt token** corresponding to an insertion.
- Summing insertions into the binary denominator ($119,151 + 1,092 = 120,243$) creates a mathematically undefined denominator that mixes prompt text tokens with acoustic hypothesis events, corrupting class conditional probabilities ($P(\text{Correct}) + P(\text{Error}) \ne 1.0$).

### Mandatory Event-Based Protocol for Insertions (Task 4)
Insertions must be evaluated strictly as an **Independent Continuous Acoustic Event Detection Task**:
1. **Target Segment:** Gold TextGrid intervals tagged `sil,PPL,a` ($1,088$ events) and `sil,err,a` ($4$ events).
2. **Matching Rule:** A predicted insertion $\hat{a}$ at time $t_{\hat{a}}$ is a True Positive ($TP_{\text{ins}}$) if and only if there exists a gold insertion event $a$ spanning $[t_{\text{start}}, t_{\text{end}}]$ such that:
   $$t_{\text{start}} - \tau \le t_{\hat{a}} \le t_{\text{end}} + \tau, \quad \text{where tolerance } \tau = 50\text{ ms}$$
3. **Primary Metric:**
   - **Insertion False Alarm Rate (FAR):** Alarms per minute of speech:
     $$\text{FAR}_{\text{ins}} = \frac{FP_{\text{ins}}}{T_{\text{audio}} \text{ (minutes)}}$$
   - **Insertion Precision, Recall, and F1-score** computed over the event sets.

---

## 6. Treatment of Uncertain Targets (`err` Tags)

### Forensic Accounting
Across all gold TextGrids, the symbol `err` appears exactly 273 times:
- `CPL,err,s`: **269 occurrences** on canonical phone substitutions.
- `sil,err,a`: **4 occurrences** on epenthetic insertions.

### Linguistic Meaning in L2-ARCTIC
The annotation manual specifies that `err` was applied when the human phonetician observed an acoustic event that was:
1. Severely distorted, muffled, or mumbled.
2. An idiosyncratic non-English sound (e.g., Arabic pharyngealized consonants, Spanish bilabial fricative [β], Hindi retroflex click) that could not be mapped to any standard ARPAbet symbol.
3. Completely ambiguous between two or more candidate phonemes.

### Processing Rules for Downstream Models

```text
Handling Rules for 'err' Annotations
├── 1. Binary Mispronunciation Detection (Task 1)
│   ├── Target: 'err' indicates an unequivocal pronunciation failure
│   └── Action: b_i = 1 (Error / Mispronounced) across ALL policies (Strict and Lenient)
│
├── 2. 39-Class Substitution Diagnosis (Task 2)
│   ├── Problem: 'err' has no ground truth in the 39-phone ARPAbet vocabulary
│   ├── Action: EXCLUDE / MASK from the 39-class diagnosis confusion matrix
│   └── Diagnosis Denominator: S_eval = 14,098 - 269 = 13,829 tokens
│
└── 3. Non-Native Phoneme Recognition CTC (Task 5)
    ├── Option A (Standard 39-Phone Head): Mask CTC loss at 'err' frame (loss weight = 0.0)
    └── Option B (Extended 43-Token Head): Supervise with dedicated <UNK_PHONE> token
```

---

## 7. Typographical Anomalies & Manual Corrections

### The Catalog of 8 Praat Typographical Errors
Auditing identified eight isolated typos introduced during manual Praat logging:

| Speaker | Utterance ID | Interval (s) | Raw TextGrid Label | Systematic Category | Provisional Interpretation | Conservative Masking Action |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| `ABA` | `arctic_a0025` | 2.580 – 2.740 | `ER)` | Unclosed parenthesis | Strip `)` $\rightarrow$ `ER` | Map to standard `ER` |
| `BWC` | `arctic_a0434` | 0.990 – 1.050 | `D_` | Trailing underscore | Strip `_` $\rightarrow$ `D` | Map to standard `D` |
| `TXHC` | `arctic_b0418` | 1.870 – 1.930 | `W`` | Trailing backtick | Strip backtick $\rightarrow$ `W` | Map to standard `W` |
| `TXHC` | `arctic_b0488` | 0.880 – 0.930 | `V```` | Trailing backticks | Strip backticks $\rightarrow$ `V` | Map to standard `V` |
| `HJK` | `arctic_a0104` | 0.690 – 0.740 | `Z_` | Trailing underscore | Strip `_` $\rightarrow$ `Z` | Map to standard `Z` |
| `ERMS` | `arctic_a0097` | 0.760 – 0.810 | `Y_` | Trailing underscore | Strip `_` $\rightarrow$ `Y` | Map to standard `Y` |
| `NCC` | `arctic_a0051` | 3.990 – 4.070 | `sil,AH),a` | Unclosed parenthesis | Strip `)` $\rightarrow$ `sil,AH,a` | Map to inserted `AH` |
| `SVBI` | `arctic_a0365` | 0.970 – 1.000 | `V,V8,s` | Shift+8 keyboard slip | Normalize to `V,V*,s` $\rightarrow$ `V` | **Flag as PROVISIONAL** (Allow masking) |

### Lowercase Canonical Phone Tags
Three intervals contain lowercase letters in the canonical position:
- `TLV/arctic_b0481`: `d,sil,d` (Interval 3.060–3.110s) $\rightarrow$ Uppercase to `D,sil,d` (Deletion).
- `TLV/arctic_b0534`: `l,sil,d` (Interval 0.130–0.200s) $\rightarrow$ Uppercase to `L,sil,d` (Deletion).
- `YKWK/arctic_a0178`: `s` (Interval 2.110–2.190s) $\rightarrow$ Uppercase to `S` (Correct).

---

## 8. Deep-Dive: Rigorous Policy for `V8 → V*` in `SVBI/arctic_a0365`

### Contextual Linguistic & Physical Evidence
In `SVBI/arctic_a0365.TextGrid`:
- Prompt Sentence: *"These rumors may even originate with us"* (`arctic_a0365.txt`).
- Target Word: **"even"** (Canonical pronunciation: `IY1 V AH0 N` or `IY1 V IH0 N`).
- Interval 12: `IY1` ($0.87 - 0.97\text{ s}$)
- Interval 13: `V,V8,s` ($0.97 - 1.00\text{ s}$)
- Interval 14: `IH0` ($1.00 - 1.07\text{ s}$)
- Interval 15: `N` ($1.07 - 1.21\text{ s}$)

### The Keyboard Slip Hypothesis
On standard US QWERTY keyboards:
$$\text{Key } 8 \iff \text{Key } * \quad (\text{Shift} + 8 = *)$$
The annotator intended to type `V,V*,s` (indicating that speaker `SVBI` produced an accented labiodental fricative [v] with Indian English allophonic coloring, such as a labiodental approximant [ʋ]), but released the `Shift` key milliseconds early, registering `V8`.

### Engineering Mandate: Provisional Status
1. **Designation:** This interpretation is classified as **PROVISIONAL**.
2. **Scientific Rationale:** Without acoustic waveform inspection or consultation with the original annotator, it cannot be proven beyond doubt that `V8` did not represent an internal annotator code or audio artifact.
3. **Operational Implementation:**
   - Preprocessing scripts (`RQ1-PREP-01`) must expose a configuration parameter:
     `HANDLE_PROVISIONAL_V8: "NORMALIZE_TO_V_STAR" | "TREAT_AS_UNCERTAIN_MASK"`
   - Under `"NORMALIZE_TO_V_STAR"`: Binary label = Error (1); Diagnosis target = `V`.
   - Under `"TREAT_AS_UNCERTAIN_MASK"`: Binary label = Error (1); Diagnosis target = Masked from evaluation.
   - Default setting for Project Langfens: `"NORMALIZE_TO_V_STAR"` with explicit uncertainty logging in the sample contract.
