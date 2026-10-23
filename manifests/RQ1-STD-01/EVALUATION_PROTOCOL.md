# LANGFENS — RQ1-STD-01: Multi-Granular Evaluation Metrics & Protocol Specification

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Task E — Evaluation Metrics, Statistical Significance & CAPT Pedagogical Benchmarks  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Phonetics Specialist  
**Evaluated Benchmarks:**
1. **L2-ARCTIC Release v5.0** (Categorical MDD, Substitutions, Deletions, Insertions)
2. **SpeechOcean762** (Continuous Pronunciation Quality Scoring & Multi-Aspect Assessment)
**Deliverable Associated Path:** `manifests/RQ1-STD-01/EVALUATION_PROTOCOL.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`
- `manifests/RQ1-AUD-01/reconciliation/SCIENTIFIC_CONSISTENCY_REPORT.md`
- `manifests/RQ1-AUD-01/SPEECHOCEAN762_ANNOTATION_AUDIT.md`
- `manifests/RQ1-AUD-01/MDD_RESEARCH_FEASIBILITY.md`

---

## 1. Executive Summary & Evaluation Regimes

To maintain absolute scientific rigor, Project Langfens separates speech pronunciation assessment into seven distinct mathematical regimes. Each regime operates with its own well-defined input representation, ground truth target, evaluation unit, independent denominator, and primary ranking metric.

```text
The Seven Evaluation Regimes
├── 1. Binary Canonical-Phone Mispronunciation Detection (Token-Level)
├── 2. Phoneme Substitution Diagnosis (Conditional Token-Level)
├── 3. Phoneme Deletion Detection (Conditional Token-Level)
├── 4. Epenthetic Insertion-Event Detection (Continuous Acoustic Time Windows)
├── 5. Realized Phoneme Recognition (Sequence-Level Model PER)
├── 6. Multi-Aspect Pronunciation Quality Scoring (Continuous Regression)
└── 7. Educational CAPT Feedback Quality (Pedagogical Evaluation Rubric)
```

---

## 2. Regime 1: Binary Mispronunciation Detection

### Task Definition
For every canonical reference phone token $y_i \in Y$ in prompt sentence $Y = (y_1, y_2, \dots, y_M)$, predict binary label $\hat{b}_i \in \{0, 1\}$, where:
- $b_i = 0$ (**Negative / Correct**): Speaker produced the target phoneme without categorical phonological error.
- $b_i = 1$ (**Positive / Error**): Speaker substituted, distorted, or deleted the target phoneme.

### Mathematical Formulations & Metrics

$$\begin{aligned}
\text{Evaluation Unit} &= \text{Canonical reference prompt phone token} \\
\text{Primary Denominator} &= N_{\text{canonical}} = C + S + D = \mathbf{119,151} \quad (\text{Strict Invariance}) \\
\text{Positive Class (Error)} &= E_{\text{canon}} = S + D = \mathbf{17,518} \quad (14.70\%) \\
\text{Negative Class (Correct)} &= C = \mathbf{101,633} \quad (85.30\%)
\end{aligned}$$

Given predictions $\hat{b}_i \in \{0, 1\}$ and ground truth $b_i \in \{0, 1\}$, compute counts on the **Positive (Error = 1)** class:
- **True Positives ($TP$):** Number of true errors correctly flagged as errors ($b_i=1, \hat{b}_i=1$).
- **False Positives ($FP$):** Number of correct phones falsely flagged as errors ($b_i=0, \hat{b}_i=1$).
- **True Negatives ($TN$):** Number of correct phones correctly passed ($b_i=0, \hat{b}_i=0$).
- **False Negatives ($FN$):** Number of true errors missed by the model ($b_i=1, \hat{b}_i=0$).

#### Core Evaluation Metrics
1. **Detection Precision:**
   $$\text{Prec} = \frac{TP}{TP + FP}$$
2. **Detection Recall:**
   $$\text{Rec} = \frac{TP}{TP + FN}$$
3. **Detection F1-Score:**
   $$\text{F1} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$
4. **Precision-Recall Area Under Curve (PR-AUC):**
   $$\text{PR-AUC} = \int_0^1 \text{Prec}(R) \, dR$$
   *Mandatory Primary Metric:* Because of the $5.80 : 1$ class imbalance, PR-AUC evaluates ranking quality independently of operating decision thresholds.
5. **False Acceptance Rate (FAR) & False Rejection Rate (FRR):**
   In educational CAPT:
   - **False Acceptance (FA):** System misses a student's error ($FN$), reinforcing improper articulation:
     $$\text{FAR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$$
   - **False Rejection (FR):** System falsely flags a correct sound ($FP$), causing learner frustration:
     $$\text{FRR} = \frac{FP}{TN + FP} = 1 - \text{Specificity}$$
6. **ROC-AUC & Equal Error Rate (EER):** Computed across posterior probabilities for threshold-free comparison.

#### Strictly Prohibited Metric: Unweighted Accuracy
Evaluating binary MDD using standard unweighted accuracy ($\frac{TP+TN}{N}$) is **strictly forbidden**. A trivial baseline predicting $\hat{b}_i = 0$ for all tokens achieves $85.30\%$ accuracy while possessing zero diagnostic utility.

---

## 3. Regime 2: Phoneme Substitution Diagnosis

### Task Definition
Given that canonical reference phone $y_i = c$ is flagged as substituted ($b_i = 1$), predict the specific perceived phonetic realization $\hat{p}_i \in \mathcal{V}_{39}$.

### Denominator & Masking Rules

$$\begin{aligned}
S_{\text{raw}} &= 14,098 \quad \text{substitution intervals in L2-ARCTIC} \\
N_{\text{err}} &= 269 \quad \text{intervals labeled 'CPL,err,s' (unrecognizable/distorted)} \\
S_{\text{eval}} &= S_{\text{raw}} - N_{\text{err}} = 14,098 - 269 = \mathbf{13,829} \quad \text{evaluable 39-class substitutions}
\end{aligned}$$

- **Masking Contract:** The 269 `CPL,err,s` tokens must be masked from the 39-phone diagnosis confusion matrix and diagnosis accuracy calculation.

### Metrics
1. **Substitution Diagnosis Accuracy ($\text{Diag-Acc}$):**
   $$\text{Diag-Acc} = \frac{\sum_{i=1}^{S_{\text{eval}}} \mathbb{I}(\hat{p}_i = p_i)}{S_{\text{eval}}}$$
2. **Class-Conditional Substitution Accuracy:** Computed per canonical phoneme (e.g. for `DH` stopping: $\text{Acc}_{DH \rightarrow D} = \frac{\text{Count}(\text{True } DH \rightarrow \hat{D})}{\text{Total } DH \text{ substitutions}}$).
3. **Phoneme Confusion Matrix ($M$):** A $39 \times 39$ row-normalized contingency matrix where entry $(c, p)$ represents $P(\text{Predicted } p \mid \text{True } c)$.
4. **Hierarchical Error Rate (HER) / Articulatory Distance:**
   Assigns partial credit based on phonetic feature overlap (Place, Manner, Voicing):
   $$\text{Dist}(p_1, p_2) = \frac{1}{3} \left[ \mathbb{I}(\text{Place}_1 \ne \text{Place}_2) + \mathbb{I}(\text{Manner}_1 \ne \text{Manner}_2) + \mathbb{I}(\text{Voicing}_1 \ne \text{Voicing}_2) \right]$$

---

## 4. Regime 3: Phoneme Deletion Detection

### Task Definition
Given canonical prompt phone $y_i = c$, detect whether the speaker omitted the phone entirely, resulting in silence or elision.

### Denominator & Reconciled Baseline

$$\begin{aligned}
D_{\text{corpus}} &= \mathbf{3,420} \quad \text{deletion tokens corpus-wide (2.87% of all canonical tokens)} \\
D_{\text{Viet}} &= \mathbf{1,566} \quad \text{deletion tokens in Vietnamese cohort (45.79% of corpus deletions)}
\end{aligned}$$

### Metrics
1. **Deletion Detection Recall ($\text{Rec}_{\text{del}}$):**
   $$\text{Rec}_{\text{del}} = \frac{TP_{\text{del}}}{D} = \frac{TP_{\text{del}}}{3,420}$$
2. **Deletion Detection Precision ($\text{Prec}_{\text{del}}$):**
   $$\text{Prec}_{\text{del}} = \frac{TP_{\text{del}}}{TP_{\text{del}} + FP_{\text{del}}}$$
3. **Vietnamese Deletion Recall ($\text{Rec}_{\text{del, Viet}}$):**
   Evaluated strictly on the Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`):
   $$\text{Rec}_{\text{del, Viet}} = \frac{TP_{\text{del, Viet}}}{1,566}$$
   *Mandatory Benchmark Requirement:* Any MDD model deployed in Project Langfens must explicitly report this Vietnamese deletion recall metric.

---

## 5. Regime 4: Epenthetic Insertion-Event Detection

### Task Definition & Independent Status
Insertions ($I = 1,092$) occur between reference tokens in acoustic silence segments. They have no index in prompt sequence $Y$ and cannot enter token classification denominators.

### Windowed Event-Matching Protocol
- **Gold Events:** Intervals tagged `sil,PPL,a` ($1,088$) and `sil,err,a` ($4$) in Praat TextGrids.
- **Matching Rule:** A predicted insertion $\hat{a}$ at timestamp $t_{\hat{a}}$ is a True Positive ($TP_{\text{ins}}$) if and only if there exists a gold insertion spanning $[t_{\text{start}}, t_{\text{end}}]$ such that:
  $$t_{\text{start}} - \tau \le t_{\hat{a}} \le t_{\text{end}} + \tau, \quad \text{where tolerance } \tau = 50\text{ ms}$$
- **Metrics:**
  1. **Insertion False Alarm Rate ($\text{FAR}_{\text{ins}}$):** False alarms per minute of speech:
     $$\text{FAR}_{\text{ins}} = \frac{FP_{\text{ins}}}{T_{\text{audio}} \text{ (minutes)}} \quad (\text{Total } T_{\text{audio}} \approx 3.8 \text{ hours})$$
  2. **Insertion Precision & Recall:** Computed over the discrete event sets.

---

## 6. Regime 5: Realized Phoneme Recognition (Model PER vs Corpus PER)

### Sequence-Level Levenshtein Edit Distance
Model phoneme recognition quality is evaluated by aligning the predicted phoneme sequence $\hat{P}$ against canonical prompt $Y$ or gold realized transcript $P$ using Wagner-Fischer dynamic programming:

$$\text{PER} = \frac{S + D + I}{N_{\text{ref}}} \times 100\%$$

### Critical Scientific Distinction: Model PER vs Corpus Baseline PER
- **Corpus Annotation Baseline PER ($15.62\%$):**
  $$\text{PER}_{\text{corpus}} = \frac{14,098 + 3,420 + 1,092}{119,151} = \frac{18,610}{119,151} = \mathbf{15.6188\% \approx 15.62\%}$$
  This is the ground-truth phonetic distance between native canonical English and non-native learner productions.
- **Model Hypothesis PER ($\text{PER}_{\text{model}}$):**
  Measures the recognition error of the neural acoustic model when predicting the speaker's true realization $P$:
  $$\text{PER}_{\text{model}} = \frac{S_{\text{hyp}} + D_{\text{hyp}} + I_{\text{hyp}}}{N_{\text{realized}}} \times 100\%$$
- **Rule:** Never conflate the two. A model with $\text{PER}_{\text{model}} = 10\%$ is an acoustic recognizer, not an annotator that detected $10\%$ mispronunciations.

---

## 7. Regime 6: Pronunciation Quality Scoring (SpeechOcean762)

### Granularities & Target Dimensions
SpeechOcean762 evaluates pronunciation quality across three hierarchical granularities:
- **Phone-Level:** `phones-accuracy` ($0.0 - 2.0$, step 0.2; 5-expert consensus).
- **Word-Level:** `accuracy` ($0 - 10$), `stress` ($5 \text{ or } 10$), `total` ($0 - 10$).
- **Sentence-Level:** `accuracy` ($0 - 10$), `fluency` ($0 - 10$), `prosodic` ($0 - 10$), `completeness` ($0.0 - 10.0$), `total` ($0 - 10$).

### Formal Scoring Metrics
For continuous predicted score $\hat{s}_i$ and human consensus score $s_i$ over $N$ evaluation items:
1. **Pearson Correlation Coefficient (PCC / $r$):**
   $$r = \frac{\sum_{i=1}^N (s_i - \bar{s})(\hat{s}_i - \bar{\hat{s}})}{\sqrt{\sum_{i=1}^N (s_i - \bar{s})^2} \sqrt{\sum_{i=1}^N (\hat{s}_i - \bar{\hat{s}})^2}}$$
2. **Spearman Rank Correlation Coefficient ($\rho$):**
   $$\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}, \quad \text{where } d_i = \text{rank}(s_i) - \text{rank}(\hat{s}_i)$$
3. **Mean Squared Error (MSE) & Mean Absolute Error (MAE):**
   $$\text{MSE} = \frac{1}{N}\sum_{i=1}^N (\hat{s}_i - s_i)^2, \quad \text{MAE} = \frac{1}{N}\sum_{i=1}^N |\hat{s}_i - s_i|$$

---

## 8. Regime 7: Educational CAPT Feedback Quality Rubric

To evaluate feedback generated for learners, Project Langfens defines a four-dimensional qualitative evaluation rubric:

```text
Educational Feedback Evaluation Rubric
├── 1. Articulatory Correctness (Accuracy of acoustic & phonetic diagnosis)
├── 2. Pedagogical Clarity (Accessible plain English; avoiding phonetic jargon)
├── 3. Communicative Relevance (Prioritizing high-impact phonological errors)
└── 4. Physical Actionability (Actionable tongue, lip, and voicing cues)
```

### Rubric Scoring Scale ($1 - 5$)

| Dimension | Score 1 (Unacceptable) | Score 3 (Marginal) | Score 5 (Exemplary) |
| :--- | :--- | :--- | :--- |
| **Correctness** | Diagnoses nonexistent error or hallucinated phone | Correct phone flagged, but wrong substitution identified | Exactly identifies substituted/omitted phone and acoustic trigger |
| **Clarity** | Dense phonetics jargon (e.g. *"Coronal anterior laminal friction"*) | Understandable, but awkward or verbose | Clear, friendly, accessible to A2/B1 English language learners |
| **Relevance** | Nitpicks minor allophonic accent while ignoring deletion | Identifies error, but fails to contextualize semantic impact | Prioritizes meaning-altering phonemic errors (e.g. *bat* vs *bad*) |
| **Actionability** | Vagaries (e.g. *"Pronounce it better"*) | High-level cue (e.g. *"Make it longer"*) | Concrete physical cue (e.g. *"Touch your tongue behind your teeth"*) |

---

## 9. Aggregation Standards & Statistical Significance

### Macro vs. Micro Aggregation
- **Micro-Average:** Pools all phone tokens across all test speakers into a single contingency table before computing precision, recall, and F1. Reflects corpus-wide performance.
- **Macro-Average:** Computes metrics independently per speaker or per L1 group, then computes the arithmetic mean. Prevents high-error speakers (e.g. `HQTV`) from dominating metrics.
- **Reporting Rule:** Research publications in Project Langfens must report **both Micro- and Macro-averages** across all test folds.

### 95% Bootstrap Confidence Intervals
All reported test metrics must include **$95\%$ non-parametric bootstrap confidence intervals** generated via:
$$B = 1,000 \text{ resamples at the utterance level (clustering by speaker)}$$
$$\text{CI}_{95\%} = \left[ q_{0.025}, \, q_{0.975} \right]$$
Resampling at the utterance level accounts for intra-utterance phonetic correlation and provides rigorous scientific bounds.
