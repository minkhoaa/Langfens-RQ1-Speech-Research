# Metric Specification: English L2 Phoneme-Level Pronunciation Assessment

This document specifies the exact mathematical definitions, denominators, aggregation methods, and implementation standards for all metrics used in this research lab.

---

## 1. Prediction Units and Denominators

### 1.1 Target Phonemes ($N_{\text{target}}$)
- **Unit of Analysis:** Every expected canonical phoneme from the prompt text within eligible utterances.
- **Classes:** Mutually exclusive set $\mathcal{C}_{\text{target}} = \{\text{CORRECT}, \text{OMISSION}, \text{SUBSTITUTION}\}$.
- **Ground Truth:** Derived from publisher-verified corpus annotations.
- **Predicted Quantities:**
  - Probability distribution $(p_{\text{CORR}}, p_{\text{OMIS}}, p_{\text{SUBS}})$ where $\sum_{c} p_c = 1$.
  - Discrete decision $\hat{y} \in \mathcal{C}_{\text{target}}$ based on frozen decision policy.
  - Predicted substitution phoneme $\hat{s}$ when $\hat{y} = \text{SUBSTITUTION}$.

### 1.2 Inter-Phoneme Gaps ($N_{\text{gap}}$)
- **Unit of Analysis:** Every inter-phoneme interval between adjacent target phonemes, plus utterance start (before phone 1) and utterance end (after phone $N$). For an utterance of $K$ target phones, there are exactly $K+1$ gap positions.
- **Class:** Binary event $\mathcal{C}_{\text{gap}} = \{\text{NO\_INSERTION}, \text{INSERTION}\}$.
- **Ground Truth:** Non-zero phone inserted in acoustic realization with no target counterpart.
- **Predicted Quantities:**
  - Probability $p_{\text{INS}} \in [0, 1]$.
  - Discrete decision $\hat{y}_{\text{gap}} \in \{0, 1\}$ at threshold $\tau_{\text{INS}}$.

---

## 2. Core Metrics per Class

### 2.1 Average Precision (AP / PR-AUC)
For each class $c \in \{\text{CORRECT}, \text{OMISSION}, \text{SUBSTITUTION}, \text{INSERTION}\}$:
$$\text{AP}_c = \sum_{k=1}^K (R_k - R_{k-1}) P_k$$
where $P_k$ and $R_k$ are precision and recall at the $k$-th threshold of the sorted predicted probabilities. Calculated via `sklearn.metrics.average_precision_score`.

### 2.2 Precision, Recall, and F1-Score
At the frozen calibration operating point:
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 2.3 Macro Metrics
$$\text{Macro-F1}_{\text{target}} = \frac{1}{3}(F1_{\text{CORR}} + F1_{\text{OMIS}} + F1_{\text{SUBS}})$$
$$\text{Macro-AP}_{\text{target}} = \frac{1}{3}(\text{AP}_{\text{CORR}} + \text{AP}_{\text{OMIS}} + \text{AP}_{\text{SUBS}})$$

---

## 3. Calibration and Safety Guardrails

### 3.1 Expected Calibration Error (ECE)
Binned probability evaluation into $M = 10$ equally spaced intervals $I_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
**Guardrail:** $\text{ECE} \le 0.15$.

### 3.2 False Warning Rate per Utterance (FWR)
$$\text{FWR} = \frac{\text{FP}_{\text{OMIS}} + \text{FP}_{\text{SUBS}} + \text{FP}_{\text{INS}}}{N_{\text{utterances}}}$$
**Guardrail:** $\text{FWR} \le 0.20$ false errors per utterance.

### 3.3 Coverage
$$\text{Coverage} = \frac{N_{\text{evaluated}}}{N_{\text{eligible\_total}}}$$
**Constraint:** $\text{Coverage} \ge 0.80$, preferred $\ge 0.90$.

---

## 4. Milestone Gates

- **Milestone 60 (`MILESTONE_60`):**
  $\min(\text{AP}_c, F1_c) \ge 0.60$ for all four core classes simultaneously, $\text{Coverage} \ge 0.80$.
- **Outstanding Milestone (`OUTSTANDING_CONFIRMED`):**
  $\min(\text{AP}_c, \text{Precision}_c, \text{Recall}_c, F1_c) \ge 0.80$ for all four core classes simultaneously, $\text{Coverage} \ge 0.80$, guardrails satisfied, verified independently across seeds and speakers.
