# LANGFENS — RQ1-STD-01R: Validated Split Protocol & Partitioning Addendum

**Execution Gate:** RQ1-STD-01R — Specification Consistency & Pre-Implementation Gate  
**Addendum To:** `manifests/RQ1-STD-01/SPEAKER_SPLIT_PROTOCOL.md`  
**Focus Area:** Task B (4-Fold Evaluation Protocol) & Task C (Vietnamese LOSO Specification)  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Scientific Reproducibility Reviewer  
**Corpus Target:** L2-ARCTIC Release v5.0 (24 Speakers across 6 L1s; 3,599 Gold Annotations)  
**Deliverable File:** `manifests/RQ1-STD-01/reconciliation/VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/reconciliation/`  
**Execution State:** CANDIDATE SPECIFICATION ONLY (No Production Splits Generated or Frozen)  

---

## 1. Executive Summary & Defect Rectification Overview

In speech machine learning and pronunciation assessment, evaluation integrity depends entirely on leak-free, demographically balanced speaker partitioning.
This addendum resolves two critical mathematical and structural defects identified in `manifests/RQ1-STD-01/SPEAKER_SPLIT_PROTOCOL.md`:
1. **The 4-Fold Gender Imbalance Defect (Task B):** The previous 4-fold test partition claimed to provide $3\text{ Male} / 3\text{ Female}$ balance in prose, but its actual allocation table assigned $5\text{ Male} / 1\text{ Female}$ to Fold 1, and $1\text{ Male} / 5\text{ Female}$ to Fold 3.
2. **The 26-Speaker Vietnamese LOSO Contradiction (Task C):** The previous Vietnamese LOSO protocol defined training ($23$ speakers), validation ($2$ speakers), and test ($1$ speaker), summing to $26$ speakers in a $24$-speaker corpus. Validation speakers were double-counted in both the training pool and the validation set.

This document provides mathematically verified, programmatically validated candidate partition allocations that eliminate all speaker leakage, maintain demographic parity, and enforce partition disjointness.

---

## 2. Demographic Inventory of the 24 L2-ARCTIC Speakers

The L2-ARCTIC v5.0 corpus contains exactly 24 speakers evenly distributed across six native language (L1) backgrounds and two biological sexes:

| L1 Background | Male Speakers ($N=2$) | Female Speakers ($N=2$) | Total Speakers | Gold Utterances | Total Canonical Tokens | Total Error Tokens | L1 Mean Error Rate (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabic** | `ABA`, `YBAA` | `SKA`, `ZHAA` | 4 | 599 | 19,764 | 1,928 | 9.76% |
| **Chinese** | `BWC`, `TXHC` | `LXC`, `NCC` | 4 | 600 | 19,778 | 3,038 | 15.36% |
| **Hindi** | `ASI`, `RRBI` | `SVBI`, `TNI` | 4 | 600 | 19,737 | 2,762 | 13.99% |
| **Korean** | `HKK`, `YKWK` | `HJK`, `YDCK` | 4 | 600 | 19,686 | 1,854 | 9.42% |
| **Spanish** | `EBVS`, `ERMS` | `MBMPS`, `NJS` | 4 | 600 | 20,050 | 3,215 | 16.03% |
| **Vietnamese** | `HQTV`, `TLV` | `PNV`, `THV` | 4 | 600 | 20,136 | 4,721 | **23.45%** |
| **Corpus Total** | **12 Male** | **12 Female** | **24** | **3,599** | **119,151** | **17,518** | **14.70%** |

*(Note: Speaker `YBAA` has 149 gold annotated utterances; all other 23 speakers have exactly 150 gold annotated utterances, yielding $23 \times 150 + 149 = 3,599$ utterances).*

---

## 3. Task B: Validation & Correction of 4-Fold Cross-Validation Architecture

### 3.1 Forensic Audit of Prior Proposal
In `SPEAKER_SPLIT_PROTOCOL.md` (lines 114–120), the proposed test partitions were:
- **Fold 1 Test:** `{ABA (M), BWC (M), ASI (M), HJK (F), EBVS (M), HQTV (M)}` $\rightarrow$ **5 Male, 1 Female**.
- **Fold 2 Test:** `{SKA (F), LXC (F), RRBI (M), HKK (M), ERMS (M), PNV (F)}` $\rightarrow$ **3 Male, 3 Female**.
- **Fold 3 Test:** `{YBAA (M), NCC (F), SVBI (F), YDCK (F), MBMPS (F), THV (F)}` $\rightarrow$ **1 Male, 5 Female**.
- **Fold 4 Test:** `{ZHAA (F), TXHC (M), TNI (F), YKWK (M), NJS (F), TLV (M)}` $\rightarrow$ **3 Male, 3 Female**.

**Finding:**
While the prose asserted that each fold satisfied demographic balance ("3M, 3F; 1 per L1"), the table contained severe gender skew. A model evaluated on Fold 1 would be evaluated almost exclusively on male vocal tract characteristics, while Fold 3 evaluated predominantly female speakers.

### 3.2 Combinatorial Proof of Feasibility
To evaluate whether four folds can simultaneously satisfy all constraints:
1. **Constraint 1 (Fold Cardinality):** Exactly 6 test speakers per fold ($4 \times 6 = 24$ speakers).
2. **Constraint 2 (L1 Representation):** Exactly 1 speaker per L1 per fold.
3. **Constraint 3 (Biological Sex Balance):** Exactly 3 male and 3 female speakers per fold.
4. **Constraint 4 (Partition Disjointness):** Each speaker appears in exactly one test fold.
5. **Constraint 5 (Internal Partition Disjointness):** Train, validation, and test sets are mutually disjoint.
6. **Constraint 6 (Role Cardinalities):** Each fold consists of 16 train, 2 validation, and 6 test speakers.

**Proof of Existence:**
Each of the 6 L1s contains exactly 2 Male and 2 Female speakers. Over 4 test folds, each L1 must contribute a Male speaker to exactly 2 folds, and a Female speaker to exactly 2 folds.
Representing the sex assignment as a $4 \times 6$ binary matrix $M \in \{0, 1\}^{4 \times 6}$ (where $1 = \text{Male}, 0 = \text{Female}$, rows = folds, columns = L1s):
- Row constraint: $\sum_{j=1}^6 M_{i, j} = 3 \quad \forall i \in \{1, 2, 3, 4\}$ (3 Males per fold).
- Column constraint: $\sum_{i=1}^4 M_{i, j} = 2 \quad \forall j \in \{1, \dots, 6\}$ (2 Males per L1).

The number of valid binary matrices satisfying these exact marginal sums is given by the coefficient in the bipartite matching expansion:
$$N_{\text{valid matrices}} = \mathbf{1,860}$$
Because $1,860 > 0$, valid assignments are mathematically guaranteed to exist.

### 3.3 Validated Candidate 4-Fold Allocation Table
We construct a canonical allocation that satisfies all constraints, balances error rates across folds, and specifies exact validation speaker assignments:

```text
Validated Candidate 4-Fold Architecture
├── Fold 1: Test = {ABA (M), LXC (F), ASI (M), HJK (F), EBVS (M), PNV (F)}   [3M, 3F]
│   ├── Val   = {RRBI (M, Hind), NJS (F, Span)}                              [1M, 1F]
│   └── Train = 16 remaining speakers                                        [8M, 8F]
│
├── Fold 2: Test = {SKA (F), BWC (M), SVBI (F), HKK (M), MBMPS (F), HQTV (M)} [3M, 3F]
│   ├── Val   = {YBAA (M, Arab), TNI (F, Hind)}                              [1M, 1F]
│   └── Train = 16 remaining speakers                                        [8M, 8F]
│
├── Fold 3: Test = {YBAA (M), NCC (F), RRBI (M), YDCK (F), ERMS (M), THV (F)} [3M, 3F]
│   ├── Val   = {TXHC (M, Chin), ZHAA (F, Arab)}                             [1M, 1F]
│   └── Train = 16 remaining speakers                                        [8M, 8F]
│
└── Fold 4: Test = {ZHAA (F), TXHC (M), TNI (F), YKWK (M), NJS (F), TLV (M)}   [3M, 3F]
    ├── Val   = {HKK (M, Kor), NCC (F, Chin)}                                [1M, 1F]
    └── Train = 16 remaining speakers                                        [8M, 8F]
```

### 3.4 Detailed Metric and Speaker Cardinality per Fold

| Fold ID | Partition Role | Speaker IDs | Sex Ratio | L1s Represented | Gold Utts | Canonical Tokens ($N$) | Errors ($E$) | Error Rate (%) |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | **Test** | `ABA`, `LXC`, `ASI`, `HJK`, `EBVS`, `PNV` | 3M / 3F | Arab, Chin, Hind, Kor, Span, Viet | 900 | 29,769 | 4,520 | 15.18% |
| | **Val** | `RRBI`, `NJS` | 1M / 1F | Hind, Span | 300 | 9,956 | 1,157 | 11.62% |
| | **Train** | `SKA`, `YBAA`, `ZHAA`, `BWC`, `NCC`, `TXHC`, `SVBI`, `TNI`, `HKK`, `YDCK`, `YKWK`, `ERMS`, `MBMPS`, `HQTV`, `THV`, `TLV` | 8M / 8F | All 6 L1s | 2,399 | 79,426 | 11,841 | 14.91% |
| **Fold 2** | **Test** | `SKA`, `BWC`, `SVBI`, `HKK`, `MBMPS`, `HQTV` | 3M / 3F | Arab, Chin, Hind, Kor, Span, Viet | 900 | 29,798 | 4,638 | 15.56% |
| | **Val** | `YBAA`, `TNI` | 1M / 1F | Arab, Hind | 299 | 9,873 | 921 | 9.33% |
| | **Train** | `ABA`, `ZHAA`, `LXC`, `NCC`, `TXHC`, `ASI`, `RRBI`, `HJK`, `YDCK`, `YKWK`, `EBVS`, `ERMS`, `NJS`, `PNV`, `THV`, `TLV` | 8M / 8F | All 6 L1s | 2,400 | 79,480 | 11,959 | 15.05% |
| **Fold 3** | **Test** | `YBAA`, `NCC`, `RRBI`, `YDCK`, `ERMS`, `THV` | 3M / 3F | Arab, Chin, Hind, Kor, Span, Viet | 899 | 29,798 | 4,220 | 14.16% |
| | **Val** | `TXHC`, `ZHAA` | 1M / 1F | Chin, Arab | 300 | 9,881 | 1,123 | 11.37% |
| | **Train** | `ABA`, `SKA`, `BWC`, `LXC`, `ASI`, `SVBI`, `TNI`, `HJK`, `HKK`, `YKWK`, `EBVS`, `MBMPS`, `NJS`, `HQTV`, `PNV`, `TLV` | 8M / 8F | All 6 L1s | 2,400 | 79,472 | 12,175 | 15.32% |
| **Fold 4** | **Test** | `ZHAA`, `TXHC`, `TNI`, `YKWK`, `NJS`, `TLV` | 3M / 3F | Arab, Chin, Hind, Kor, Span, Viet | 900 | 29,786 | 4,140 | 13.90% |
| | **Val** | `HKK`, `NCC` | 1M / 1F | Kor, Chin | 300 | 9,866 | 1,012 | 10.26% |
| | **Train** | `ABA`, `SKA`, `YBAA`, `BWC`, `LXC`, `ASI`, `RRBI`, `SVBI`, `HJK`, `YDCK`, `EBVS`, `ERMS`, `MBMPS`, `HQTV`, `PNV`, `THV` | 8M / 8F | All 6 L1s | 2,399 | 79,499 | 12,366 | 15.55% |

### 3.5 Verification Matrix for Candidate 4-Fold Split

| Constraint Check | Verification Requirement | Observed Candidate Status | Verification Result |
| :--- | :--- | :--- | :---: |
| **Test Speaker Count** | Exactly 6 test speakers per fold | Fold 1: 6, Fold 2: 6, Fold 3: 6, Fold 4: 6 | **PASS** |
| **L1 Representation** | Exactly 1 speaker per L1 per fold in test | All 6 L1s present in every test fold | **PASS** |
| **Biological Sex Balance** | Exactly 3M / 3F per fold in test | All 4 test folds contain exactly 3M and 3F | **PASS** |
| **Coverage & Disjointness** | Union of test sets = 24; Pairwise disjoint | $\bigcup_{k=1}^4 S_{\text{test}}^{(k)} = S_{\text{all}}$, $|S_{\text{test}}^{(i)} \cap S_{\text{test}}^{(j)}| = 0$ | **PASS** |
| **Validation Cardinality** | Exactly 2 validation speakers per fold | 2 speakers per fold (1M, 1F, distinct L1s) | **PASS** |
| **Training Cardinality** | Exactly 16 training speakers per fold | 16 speakers per fold (8M, 8F, all 6 L1s) | **PASS** |
| **Zero Partition Leakage** | $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$ | Strictly mutually disjoint across all folds | **PASS** |
| **Corpus Total Conservation** | $\sum \text{Test Tokens} = 119,151$; $\sum \text{Test Errors} = 17,518$ | $29,769 + 29,798 + 29,798 + 29,786 = 119,151$<br>$4,520 + 4,638 + 4,220 + 4,140 = 17,518$ | **PASS** |

---

## 4. Task C: Correction of Vietnamese Evaluation Protocol

### 4.1 Forensic Audit of Prior Vietnamese LOSO Specification
In `SPEAKER_SPLIT_PROTOCOL.md` (lines 131–140), the protocol defined:
- **Training Set:** 20 non-Vietnamese speakers + 3 Vietnamese speakers = **23 speakers**.
- **Validation Set:** 2 non-Vietnamese speakers = **2 speakers**.
- **Test Set:** 1 held-out Vietnamese speaker = **1 speaker**.

$$\Sigma_{\text{speakers}} = 23 + 2 + 1 = \mathbf{26 \text{ speakers}}$$
Because the corpus contains only **24 speakers**, the 2 validation speakers were counted **both in the 20 non-Vietnamese training speakers and in the validation set**. This represents an intolerable form of **validation-in-training data leakage**.

### 4.2 Disambiguation of Vietnamese Evaluation Regimes
To prevent conflating distinct scientific questions, Project Langfens establishes two strictly separated evaluation regimes for the Vietnamese cohort (`HQTV`, `PNV`, `THV`, `TLV`):

```text
Vietnamese Evaluation Architecture
├── Regime A: Cross-Vietnamese Generalization (Intra-L1 Speaker LOSO; 4 Rounds)
│   ├── Evaluates: Model generalization to an unseen Vietnamese learner when other Vietnamese learners are in training
│   ├── Training Set: 18 non-Vietnamese + 3 other Vietnamese = 21 speakers
│   ├── Validation Set: 2 non-Vietnamese speakers (for checkpoint & threshold tuning)
│   └── Test Set: Exactly 1 held-out Vietnamese speaker (HQTV, PNV, THV, or TLV)
│
└── Regime B: Zero-Shot Vietnamese-L1 Transfer (Cross-Lingual Accent Transfer)
    ├── Evaluates: Model ability to detect Vietnamese pronunciation errors WITHOUT any Vietnamese training data
    ├── Training Set: Exactly 18 non-Vietnamese speakers
    ├── Validation Set: Exactly 2 non-Vietnamese speakers
    └── Test Set: All 4 Vietnamese speakers (HQTV, PNV, THV, TLV; 600 utterances)
```

---

## 5. Formal Specification of Regime A: Cross-Vietnamese Generalization (4 Rounds)

In Regime A, four evaluation rounds are executed, holding out each Vietnamese speaker in turn.

### 5.1 Protocol Principles
1. **Validation Speaker Carve-Out:** The 2 validation speakers (`RRBI` [M, Hindi] and `NJS` [F, Spanish]) are explicitly removed from the development pool. They never enter the training set.
2. **Threshold Calibration Isolation:** Decision thresholds $\theta^*$ must be calibrated **strictly on the 2 non-Vietnamese validation speakers**. The held-out Vietnamese test speaker must never be evaluated during validation or threshold tuning.
3. **Partition Disjointness:** In every round, $|S_{\text{train}}| = 21$, $|S_{\text{val}}| = 2$, $|S_{\text{test}}| = 1$, summing to exactly 24 speakers.

### 5.2 Round Allocations and Exact Cardinalities

| Round ID | Test Speaker | Sex | Test Tokens ($N$) | Test Errors ($E$) | Test Error Rate (%) | Validation Speakers ($N=2$) | Training Speakers ($N=21$) | Train Tokens | Train Errors |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: |
| **Round 1** | `HQTV` | M | 5,038 | 1,286 | 25.53% | `RRBI` (M), `NJS` (F) | 18 non-Viet + `PNV`, `THV`, `TLV` | 104,157 | 15,075 |
| **Round 2** | `PNV` | F | 5,031 | 877 | 17.43% | `RRBI` (M), `NJS` (F) | 18 non-Viet + `HQTV`, `THV`, `TLV` | 104,164 | 15,484 |
| **Round 3** | `THV` | F | 5,030 | 1,332 | 26.48% | `RRBI` (M), `NJS` (F) | 18 non-Viet + `HQTV`, `PNV`, `TLV` | 104,165 | 15,029 |
| **Round 4** | `TLV` | M | 5,037 | 1,226 | 24.34% | `RRBI` (M), `NJS` (F) | 18 non-Viet + `HQTV`, `PNV`, `THV` | 104,158 | 15,135 |

$$\begin{aligned}
\text{Validation Set (Invariant across rounds):} \quad & S_{\text{val}} = \{\text{RRBI}, \text{NJS}\} \\
& \text{Utterances} = 300, \quad \text{Tokens} = 9,956, \quad \text{Errors} = 1,157 \ (11.62\%)
\end{aligned}$$

---

## 6. Formal Specification of Regime B: Zero-Shot Vietnamese-L1 Transfer

Regime B evaluates the model's inductive transfer capability: can representations learned from Arabic, Chinese, Hindi, Korean, and Spanish transfer zero-shot to Vietnamese speech without encountering any Vietnamese phonetic patterns during training?

### 6.1 Protocol Architecture
- **Training Pool ($N=18$):** Drawn exclusively from the 20 non-Vietnamese speakers, excluding validation speakers:
  `{ABA, SKA, YBAA, ZHAA, BWC, LXC, NCC, TXHC, ASI, SVBI, TNI, HJK, HKK, YDCK, YKWK, EBVS, ERMS, MBMPS}`
- **Validation Pool ($N=2$):** `{RRBI (M, Hindi), NJS (F, Spanish)}`.
- **Test Pool ($N=4$):** All four Vietnamese speakers: `{HQTV, PNV, THV, TLV}`.

### 6.2 Partition Cardinalities

| Partition Role | Speakers Included | Speaker Count | Gold Utterances | Canonical Tokens ($N$) | Errors ($E$) | Error Rate (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 18 non-Vietnamese speakers | **18** | 2,699 | 89,059 | 11,640 | 13.07% |
| **Val Split** | `RRBI` (M), `NJS` (F) | **2** | 300 | 9,956 | 1,157 | 11.62% |
| **Test Split** | `HQTV`, `PNV`, `THV`, `TLV` | **4** | **600** | **20,136** | **4,721** | **23.45%** |
| **Corpus Total** | **Full 24-Speaker Corpus** | **24** | **3,599** | **119,151** | **17,518** | **14.70%** |

---

## 7. Mathematical Partition-Disjointness Proof

Let $\mathcal{S}$ denote the set of all 24 speakers in L2-ARCTIC v5.0:
$$\mathcal{S} = \{s_1, s_2, \dots, s_{24}\}, \quad |\mathcal{S}| = 24$$

### 7.1 Proof for Candidate 4-Fold CV
For every fold $k \in \{1, 2, 3, 4\}$, let $S_{\text{test}}^{(k)}$, $S_{\text{val}}^{(k)}$, and $S_{\text{train}}^{(k)}$ be the assigned subsets:
1. **Pairwise Disjointness within Fold:**
   $$S_{\text{test}}^{(k)} \cap S_{\text{val}}^{(k)} = \emptyset, \quad S_{\text{test}}^{(k)} \cap S_{\text{train}}^{(k)} = \emptyset, \quad S_{\text{val}}^{(k)} \cap S_{\text{train}}^{(k)} = \emptyset$$
2. **Exhaustive Partition of Corpus:**
   $$S_{\text{test}}^{(k)} \cup S_{\text{val}}^{(k)} \cup S_{\text{train}}^{(k)} = \mathcal{S}$$
3. **Cardinality Conservation:**
   $$|S_{\text{test}}^{(k)}| + |S_{\text{val}}^{(k)}| + |S_{\text{train}}^{(k)}| = 6 + 2 + 16 = 24 = |\mathcal{S}|$$
4. **Test Set Partition of Corpus:**
   $$\bigcup_{k=1}^4 S_{\text{test}}^{(k)} = \mathcal{S} \quad \text{and} \quad \forall i \ne j, \ S_{\text{test}}^{(i)} \cap S_{\text{test}}^{(j)} = \emptyset$$

### 7.2 Proof for Vietnamese LOSO (Regimes A & B)
Let $\mathcal{V} = \{\text{HQTV}, \text{PNV}, \text{THV}, \text{TLV}\}$ and $\mathcal{NV} = \mathcal{S} \setminus \mathcal{V}$, where $|\mathcal{V}| = 4$ and $|\mathcal{NV}| = 20$.
Let $S_{\text{val}} \subset \mathcal{NV}$ with $|S_{\text{val}}| = 2$.
Let $S_{\text{train, base}} = \mathcal{NV} \setminus S_{\text{val}}$, with $|S_{\text{train, base}}| = 18$.

**For Regime A (Round $r$ with held-out speaker $v_r \in \mathcal{V}$):**
- $S_{\text{test}} = \{v_r\}$
- $S_{\text{val}} = S_{\text{val}} \subset \mathcal{NV}$
- $S_{\text{train}} = S_{\text{train, base}} \cup (\mathcal{V} \setminus \{v_r\})$
- **Disjointness:**
  $$\{v_r\} \cap S_{\text{val}} = \emptyset \quad (\text{since } v_r \in \mathcal{V} \text{ and } S_{\text{val}} \subset \mathcal{NV})$$
  $$\{v_r\} \cap S_{\text{train}} = \emptyset \quad (\text{since } v_r \notin S_{\text{train, base}} \text{ and } v_r \notin \mathcal{V} \setminus \{v_r\})$$
  $$S_{\text{val}} \cap S_{\text{train}} = \emptyset \quad (\text{since } S_{\text{val}} \cap S_{\text{train, base}} = \emptyset \text{ and } S_{\text{val}} \cap \mathcal{V} = \emptyset)$$
- **Cardinality:**
  $$|S_{\text{test}}| + |S_{\text{val}}| + |S_{\text{train}}| = 1 + 2 + (18 + 3) = 1 + 2 + 21 = \mathbf{24}$$

**For Regime B (Zero-Shot Transfer):**
- $S_{\text{test}} = \mathcal{V}$ ($|\mathcal{V}| = 4$)
- $S_{\text{val}} = S_{\text{val}}$ ($|S_{\text{val}}| = 2$)
- $S_{\text{train}} = S_{\text{train, base}}$ ($|S_{\text{train, base}}| = 18$)
- **Disjointness:**
  $$\mathcal{V} \cap S_{\text{val}} = \emptyset, \quad \mathcal{V} \cap S_{\text{train, base}} = \emptyset, \quad S_{\text{val}} \cap S_{\text{train, base}} = \emptyset$$
- **Cardinality:**
  $$|\mathcal{V}| + |S_{\text{val}}| + |S_{\text{train, base}}| = 4 + 2 + 18 = \mathbf{24}$$

$\blacksquare$ *Both protocols are provably disjoint, leak-free, and mathematically complete.*

---

## 8. Operating Decision Threshold Calibration Protocol

To prevent threshold calibration leakage:
1. **Prohibition:** Decision threshold $\theta$ must never be tuned on test speakers.
2. **Procedure:**
   - The neural model outputs posterior error probability $\hat{p}_i = P(\text{Error} \mid x_i)$.
   - Operating threshold $\theta^*$ is calibrated strictly on the Validation Split:
     $$\theta^* = \arg\max_{\theta \in [0.01, 0.99]} \text{F1}_{\text{val}}(\theta)$$
   - The calibrated threshold $\theta^*$ is frozen.
   - Test predictions are generated via $\hat{b}_i = \mathbb{I}(\hat{p}_i \ge \theta^*)$ and evaluated.
3. **Threshold-Free Primary Metrics:** PR-AUC and ROC-AUC must always be reported as primary ranking metrics to ensure model comparison remains completely independent of threshold selection.
