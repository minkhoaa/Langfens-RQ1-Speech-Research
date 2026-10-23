# LANGFENS — RQ1-AUD-01R: MDD Label Accounting & Evaluation Specification

**Execution Gate:** RQ1-AUD-01R — Scientific Consistency Reconciliation  
**Document Type:** Formal Engineering & Scientific Evaluation Specification  
**Focus Area:** Task A — Mispronunciation Detection & Diagnosis (MDD) Label Accounting  
**Benchmark Target:** L2-ARCTIC Release v5.0 (3,599 Gold Praat TextGrids across 24 Speakers)  
**Evaluation Lead:** Senior Speech ML Research Engineer & Research Reproducibility Reviewer  

---

## 1. Executive Summary & Mathematical Reconciliation

In speech Mispronunciation Detection and Diagnosis (MDD), a frequent methodological flaw is the conflation of **token-level binary classification** (is a canonical phone pronounced correctly or not?) with **event-level Levenshtein edit distance** (insertions, deletions, and substitutions).
This specification formally resolves the counting discrepancies present in earlier reports and defines strict, independent mathematical formulations and denominators for all evaluation tasks.

### Core Corpus Totals Ledger

$$\begin{aligned}
N_{\text{canonical}} &= 119,151 \quad \text{(Total non-silence canonical reference phone tokens)} \\
C &= 101,633 \quad \text{(Correctly pronounced canonical phone tokens, } 85.30\% \text{)} \\
S &= 14,098 \quad \text{(Substituted canonical phone tokens, } 11.83\% \text{)} \\
D &= 3,420 \quad \text{(Deleted canonical phone tokens, } 2.87\% \text{)} \\
I &= 1,092 \quad \text{(Epenthetic insertion phone events over silent segments, } 0.92\% \text{)} \\
E_{\text{canon}} &= S + D = 17,518 \quad \text{(Total canonical tokens with pronunciation errors, } 14.70\% \text{)} \\
E_{\text{event}} &= S + D + I = 18,610 \quad \text{(Total Levenshtein error events)}
\end{aligned}$$

### Reconciliation Proof

$$\begin{array}{rcccl}
\text{Canonical Decomposition:} & C + S + D & = & 101,633 + 14,098 + 3,420 = \mathbf{119,151} & (100.0000\%) \\
\text{Binary Class Balance:} & C + E_{\text{canon}} & = & 101,633 + 17,518 = \mathbf{119,151} & (100.0000\%) \\
\text{Levenshtein Error Sum:} & S + D + I & = & 14,098 + 3,420 + 1,092 = \mathbf{18,610} & (\text{Event Count})
\end{array}$$

#### Why the Prior Report Added to 100.92%
In the original Phase C audit report, the summary table listed:
$$\text{Correct (85.30\%)} + \text{Error Rate (15.62\%)} = 100.92\%$$
This occurred because the report computed:
$$\text{"Error Rate"} = \frac{E_{\text{event}}}{N_{\text{canonical}}} = \frac{18,610}{119,151} = 15.6188\% \approx 15.62\%$$
This metric is actually the **Phone Error Rate (PER)**. Because the numerator includes insertions ($I = 1,092$) which have **no corresponding canonical reference token**, the sum of operations $(C + S + D + I) / N = (119,151 + 1,092) / 119,151 = 120,243 / 119,151 = 100.916\% \approx 100.92\%$.

---

## 2. Formal Task Definitions, Denominators & Metrics

To guarantee scientific rigor, future model evaluations in Project Langfens must report metrics under five strictly separated mathematical regimes:

```text
Evaluation Regimes
├── 1. Binary Canonical-Phone Mispronunciation Detection (Token-Level)
├── 2. Substitution Diagnosis (Token-Level Conditional)
├── 3. Deletion Detection (Token-Level Conditional)
├── 4. Insertion-Event Detection (Acoustic Interval-Level)
└── 5. Phone Error Rate / Levenshtein Distance (Sequence-Level)
```

---

### Task 1: Binary Canonical-Phone Mispronunciation Detection
- **Objective:** For every canonical phone token $y_i \in Y$ in the prompt sentence, predict binary label $\hat{b}_i \in \{0, 1\}$, where:
  - $b_i = 0$ (Negative / Correct): Phone was produced correctly without categorical phonological error.
  - $b_i = 1$ (Positive / Mispronounced): Phone was substituted or omitted.
- **Evaluation Unit:** Canonical reference phone token.
- **Independent Denominator:**
  $$N_{\text{canonical}} = C + S + D = 119,151$$
- **Ground Truth Grounding:**
  - $b_i = 0 \iff$ Interval label is a single ARPAbet phone (e.g. `IH0`, `K`, `S`) or accented phone `PPL*`.
  - $b_i = 1 \iff$ Interval label is `CPL,PPL,s` (substitution) or `CPL,sil,d` (deletion).
- **Class Distribution:**
  - **Negative (Correct):** $C = 101,633$ ($85.2976\% \approx 85.30\%$)
  - **Positive (Error):** $E_{\text{canon}} = S + D = 17,518$ ($14.7024\% \approx 14.70\%$)
  - Imbalance Ratio: $5.80 : 1$ Negative-to-Positive.
- **Standard Evaluation Metrics:**
  - True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$) computed on the **Error (Positive = 1)** class.
  - **Detection Precision:** $\text{Prec} = \frac{TP}{TP + FP}$
  - **Detection Recall:** $\text{Rec} = \frac{TP}{TP + FN}$
  - **Detection F1-Score:** $\text{F1} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$
  - **PR-AUC (Precision-Recall Area Under Curve):** Mandatory primary ranking metric due to the 85:15 class imbalance.
  - **ROC-AUC & Equal Error Rate (EER):** For threshold-free posterior classifiers.
- **Forbidden Practice:** Never evaluate binary detection using unweighted Accuracy ($\frac{TP+TN}{N}$). A trivial baseline predicting all zeros scores $85.30\%$ accuracy while detecting 0 errors.

---

### Task 2: Phoneme Substitution Diagnosis
- **Objective:** Given that a canonical phone $y_i = c$ is flagged as substituted ($b_i = 1$), predict the specific perceived phonetic realization $\hat{p}_i \in \mathcal{V}_{\text{ARPAbet}}$.
- **Evaluation Unit:** Substituted canonical phone token.
- **Independent Denominators:**
  1. *Global Substitution Denominator:* $S = 14,098$ tokens.
  2. *Class-Conditional Denominator:* $N_c = \sum_{p \ne c} \text{Count}(c \rightarrow p)$ (e.g. for canonical `DH`, $N_{DH} = 1,489$ substitutions).
- **Ground Truth Grounding:**
  - Interval label matches `c, p, s` in Praat TextGrid.
- **Evaluation Metrics:**
  - **Substitution Diagnosis Accuracy:**
    $$\text{Diag-Acc} = \frac{\sum_{i=1}^S \mathbb{I}(\hat{p}_i = p_i)}{S}$$
  - **Hierarchical Error Rate (HER):** Measures articulatory distance (Place, Manner, Voicing feature overlap) between predicted phone $\hat{p}$ and actual realized phone $p$.
  - **Phoneme Confusion Matrix:** $M_{c, p} = \text{Count}(\text{True } c \rightarrow \text{Predicted } p)$.

---

### Task 3: Phoneme Deletion Detection
- **Objective:** Given a canonical phone $y_i = c$, detect whether the speaker omitted the phone entirely, resulting in silence or acoustic absence.
- **Evaluation Unit:** Canonical reference phone token.
- **Independent Denominators:**
  1. *Global Deletion Count:* $D = 3,420$ tokens ($2.8703\%$ of canonical tokens).
  2. *Canonical Error Denominator:* $\frac{D}{S + D} = \frac{3,420}{17,518} = 19.52\%$ of all error tokens are deletions.
- **Ground Truth Grounding:**
  - Interval label matches `c, sil, d` in Praat TextGrid.
- **Evaluation Metrics:**
  - **Deletion Detection Recall:** $\text{Rec}_{\text{del}} = \frac{TP_{\text{del}}}{D}$
  - **Deletion Detection Precision:** $\text{Prec}_{\text{del}} = \frac{TP_{\text{del}}}{TP_{\text{del}} + FP_{\text{del}}}$
  - **Vietnamese Deletion Recall:** Evaluated separately on the Vietnamese cohort (`HQTV`, `PNV`, `THV`, `TLV`), where $D_{\text{Viet}} = 1,566$ ($45.79\%$ of all corpus deletions).

---

### Task 4: Insertion-Event Detection
- **Objective:** Detect acoustic segments in the speech signal where the learner inserted an intrusive, epenthetic, or extraneous phone not present in the canonical prompt.
- **Evaluation Unit:** Continuous time intervals / Realized acoustic phone frames.
- **Independent Denominators:**
  - **Total Insertion Events:** $I = 1,092$ events corpus-wide.
  - **Audio Duration Denominator:** $T_{\text{gold}} \approx 3.8 \text{ hours}$ (Average rate: $\approx 4.8 \text{ insertions per audio minute}$).
  - **Canonical Reference Denominator:** $\frac{I}{N_{\text{canonical}}} = \frac{1,092}{119,151} = 0.9165\% \approx 0.92\%$ insertions per canonical phone.
- **Why Insertions CANNOT be in the Binary Classification Denominator:**
  - In binary mispronunciation detection, the input to the classifier is the sequence of canonical phones $(y_1, y_2, \dots, y_M)$ from the prompt transcript.
  - An insertion occurs *between* canonical tokens (e.g. between $y_i$ and $y_{i+1}$).
  - There is no $i$-th canonical token to label as "an insertion".
  - If insertions were added to the denominator, the denominator would represent an arbitrary mix of reference words and hypothesis frames.
- **Evaluation Metrics:**
  - **Insertion False Alarm Rate (FAR):** Number of false insertion alarms per minute of speech.
  - **Insertion Precision & Recall:** Computed over aligned time-window intervals (e.g. within $\pm 50 \text{ ms}$ tolerance of the gold TextGrid insertion boundary).

---

### Task 5: Phone Error Rate (PER)
- **Objective:** Measure overall sequence-level phonetic transcript distance between the canonical target sequence $Y$ and the realized pronunciation sequence $P$, computed via dynamic programming (Wagner-Fischer Levenshtein alignment).
- **Formal Definition:**
  $$\text{PER} = \frac{S + D + I}{N_{\text{canonical}}} \times 100\%$$
- **Corpus-Wide Baseline Value:**
  $$\text{PER} = \frac{14,098 + 3,420 + 1,092}{119,151} = \frac{18,610}{119,151} = \mathbf{15.6188\% \approx 15.62\%}$$
- **Key Scientific Distinction:**
  - $\text{PER} = 15.62\%$ is **not** the percentage of mispronounced words or tokens.
  - It is an edit-distance ratio that can theoretically exceed $100\%$ when insertions are high.
  - In contrast, the **Canonical Mispronunciation Rate** is bounded strictly in $[0, 1]$:
    $$\text{MR}_{\text{canon}} = \frac{S + D}{N_{\text{canonical}}} = \frac{17,518}{119,151} = \mathbf{14.7024\% \approx 14.70\%}$$

---

## 3. Treatment Specification for Special Tokens

L2-ARCTIC and SpeechOcean762 contain non-standard symbols and rater conventions. The table below specifies their exact engineering handling for downstream tokenization, loss computation, and evaluation:

```text
Special Token Taxonomy
├── 1. The 'err' / 'ERR' Tag (269 occurrences)
├── 2. The Deviation Asterisk '*' (1,496 occurrences)
├── 3. Silence Tokens ('sil', 'sp', 'spn', empty intervals)
├── 4. Typographical Errors (8 cataloged cases)
└── 5. Lowercase Phone Symbols (d, l, s, etc.)
```

### Detailed Token Handling Policies

| Token Syntax | Freq in L2 | Corpus Meaning | Binary Detection Policy | Substitution Diagnosis Policy | CTC Training Policy |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `CPL,err,s` | 269 | Perceived sound is unrecognizable, distorted, or non-English | Label = **Mispronounced (1)** | **Exclude** from 39-phone confusion matrix, or map to `<UNK_PHONE>` | Map target to `<UNK>` token; mask CTC loss if `<UNK>` unsupported |
| `sil,err,a` | 4 | Unidentifiable insertion | Excluded from canonical tokens | Evaluated under Task 4 (Insertion) | Target `<UNK>` |
| `CPL,PPL*,s` | 1,496 | Phone produced with perceptible foreign accent, but base identity retained | Dependent on operational threshold: <br>• Strict: **Mispronounced (1)** <br>• Lenient: **Correct (0)** | Strip `*` for standard 39-phone vocabulary; preserve `*` in auxiliary accent head | Strip `*`; map to base phone `PPL` |
| `sil`, `sp`, `spn`, `""` | 16,739 | Acoustic pause or silence segment | **Excluded** from canonical non-silence tokens $N$ | N/A | Mapped to `<BLANK>` or `<SIL>` token in CTC |
| `CPL,sil,d` | 3,420 | Canonical phone omitted at silent interval | Label = **Mispronounced (1)** | Target is `<SIL>` / Deletion | Supervision for deletion token or blank duration |
| `sil,PPL,a` | 1,088 | Phone inserted over silence interval | Excluded from canonical tokens | Evaluated under Task 4 | Inserted phone supervised in realized CTC transcript |
| `ER)` | 1 | Praat unclosed parenthesis typo in `ABA/arctic_a0025` | Handled as `ER` | Map to `ER` | Strip `)` $\rightarrow$ `ER` |
| `D_` | 1 | Trailing underscore in `BWC/arctic_a0434` | Handled as `D` | Map to `D` | Strip `_` $\rightarrow$ `D` |
| `W`` | 1 | Backtick typo in `TXHC/arctic_b0418` | Handled as `W` | Map to `W` | Strip backtick $\rightarrow$ `W` |
| `V```` | 1 | Triple backtick in `TXHC/arctic_b0488` | Handled as `V` | Map to `V` | Strip backticks $\rightarrow$ `V` |
| `Z_` | 1 | Trailing underscore in `HJK/arctic_a0104` | Handled as `Z` | Map to `Z` | Strip `_` $\rightarrow$ `Z` |
| `Y_` | 1 | Trailing underscore in `ERMS/arctic_a0097` | Handled as `Y` | Map to `Y` | Strip `_` $\rightarrow$ `Y` |
| `AH)` | 1 | Unclosed parenthesis in `NCC/arctic_a0051` (`sil,AH),a`) | Excluded from canonical tokens | Map to `AH` (insertion) | Strip `)` $\rightarrow$ `AH` |
| `V8` | 1 | Keyboard slip (`Shift+8`) for `V*` in `SVBI/arctic_a0365` | Label = **Mispronounced (1)** | Normalize to `V*` $\rightarrow$ base `V` | Map to base `V` |
| `d`, `l`, `s` | 3 | Lowercase canonical phones (`TLV/b0481`, `TLV/b0534`, `YKWK/a0178`) | Handled as standard `D`, `L`, `S` | Map to uppercase `D`, `L`, `S` | Uppercase normalization `str.upper()` |

---

## 4. Reconciled L1 & Speaker Performance Ledger

Below is the verified, empirical accounting across all 24 speakers, separating canonical evaluation tokens ($N = C + S + D$) from Levenshtein error events ($E = S + D + I$):

| Speaker | Sex | L1 Background | Canonical Non-Sil ($N$) | Correct ($C$) | Sub ($S$) | Del ($D$) | Ins ($I$) | Binary Error Tokens ($S+D$) | Binary Error Rate (%) | Total Levenshtein Errors | Phone Error Rate (PER %) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ABA` | M | Arabic | 4,935 | 4,502 | 402 | 31 | 78 | 433 | 8.77% | 511 | 10.35% |
| `SKA` | F | Arabic | 4,958 | 4,328 | 571 | 59 | 116 | 630 | 12.71% | 746 | 15.05% |
| `YBAA` | M | Arabic | 4,934 | 4,572 | 331 | 31 | 30 | 362 | 7.34% | 392 | 7.94% |
| `ZHAA` | F | Arabic | 4,937 | 4,434 | 404 | 99 | 50 | 503 | 10.19% | 553 | 11.20% |
| **Arabic Cohort** | — | — | **19,764** | **17,836** | **1,708** | **220** | **274** | **1,928** | **9.76%** | **2,202** | **11.14%** |
| `BWC` | M | Chinese | 4,947 | 4,045 | 696 | 206 | 44 | 902 | 18.23% | 946 | 19.12% |
| `LXC` | F | Chinese | 4,940 | 4,015 | 709 | 216 | 58 | 925 | 18.72% | 983 | 19.90% |
| `NCC` | F | Chinese | 4,947 | 4,356 | 536 | 55 | 31 | 591 | 11.95% | 622 | 12.57% |
| `TXHC` | M | Chinese | 4,944 | 4,324 | 492 | 128 | 73 | 620 | 12.54% | 693 | 14.02% |
| **Chinese Cohort** | — | — | **19,778** | **16,740** | **2,433** | **605** | **206** | **3,038** | **15.36%** | **3,244** | **16.40%** |
| `ASI` | M | Hindi | 4,934 | 4,053 | 761 | 120 | 26 | 881 | 17.86% | 907 | 18.38% |
| `RRBI` | M | Hindi | 4,945 | 4,429 | 451 | 65 | 47 | 516 | 10.43% | 563 | 11.39% |
| `SVBI` | F | Hindi | 4,919 | 4,113 | 717 | 89 | 28 | 806 | 16.39% | 834 | 16.95% |
| `TNI` | F | Hindi | 4,939 | 4,380 | 491 | 68 | 8 | 559 | 11.32% | 567 | 11.48% |
| **Hindi Cohort** | — | — | **19,737** | **16,975** | **2,420** | **342** | **109** | **2,762** | **13.99%** | **2,871** | **14.55%** |
| `HJK` | F | Korean | 4,920 | 4,586 | 330 | 4 | 7 | 334 | 6.79% | 341 | 6.93% |
| `HKK` | M | Korean | 4,919 | 4,498 | 318 | 103 | 44 | 421 | 8.56% | 465 | 9.45% |
| `YDCK` | F | Korean | 4,929 | 4,421 | 450 | 58 | 24 | 508 | 10.31% | 532 | 10.79% |
| `YKWK` | M | Korean | 4,918 | 4,327 | 462 | 129 | 30 | 591 | 12.02% | 621 | 12.63% |
| **Korean Cohort** | — | — | **19,686** | **17,832** | **1,560** | **294** | **105** | **1,854** | **9.42%** | **1,959** | **9.95%** |
| `EBVS` | M | Spanish | 5,009 | 3,939 | 876 | 194 | 93 | 1,070 | 21.36% | 1,163 | 23.22% |
| `ERMS` | M | Spanish | 5,013 | 4,102 | 855 | 56 | 43 | 911 | 18.17% | 954 | 19.03% |
| `MBMPS` | F | Spanish | 5,017 | 4,424 | 530 | 63 | 26 | 593 | 11.82% | 619 | 12.34% |
| `NJS` | F | Spanish | 5,011 | 4,370 | 561 | 80 | 38 | 641 | 12.79% | 679 | 13.55% |
| **Spanish Cohort** | — | — | **20,050** | **16,835** | **2,822** | **393** | **200** | **3,215** | **16.03%** | **3,415** | **17.03%** |
| `HQTV` | M | Vietnamese | 5,038 | 3,752 | 809 | 477 | 72 | 1,286 | 25.53% | 1,358 | 26.96% |
| `PNV` | F | Vietnamese | 5,031 | 4,154 | 631 | 246 | 16 | 877 | 17.43% | 893 | 17.75% |
| `THV` | F | Vietnamese | 5,030 | 3,698 | 926 | 406 | 67 | 1,332 | 26.48% | 1,399 | 27.81% |
| `TLV` | M | Vietnamese | 5,037 | 3,811 | 789 | 437 | 43 | 1,226 | 24.34% | 1,269 | 25.19% |
| **Vietnamese Cohort** | — | — | **20,136** | **15,415** | **3,155** | **1,566** | **198** | **4,721** | **23.45%** | **4,919** | **24.43%** |
| **CORPUS TOTAL** | — | — | **119,151** | **101,633** | **14,098** | **3,420** | **1,092** | **17,518** | **14.70%** | **18,610** | **15.62%** |

### Critical Takeaways from Reconciled Ledger
1. **Binary Error Rate vs PER:** Corpus-wide Binary Error Rate on canonical tokens is **14.70%** (17,518 / 119,151), whereas Phone Error Rate is **15.62%** (18,610 / 119,151).
2. **Vietnamese L1 Deletion Profile:** The Vietnamese cohort exhibits an extreme concentration of deletions ($1,566$ out of $3,420$, or $45.79\%$). In contrast, insertions in Vietnamese are modest ($198$ out of $1,092$, or $18.13\%$).
3. **Imbalance Variance Across L1s:** The positive (error) class prevalence varies from $9.42\%$ in Korean to $23.45\%$ in Vietnamese. LOSO cross-validation folds must calibrate decision thresholds per L1 background.
