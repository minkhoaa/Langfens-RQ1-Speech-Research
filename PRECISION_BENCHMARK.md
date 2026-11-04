# LANGFENS RQ1 — Precision Benchmark Report (Agent A)

**Document ID:** `LANGFENS-RQ1-PRECISION-BENCHMARK-01`  
**Evaluation Target:** Recovered P-SUP V2 Benchmark (Fold-1 VAL, 300 utterances, speakers NJS and RRBI)  
**Contract Baseline:** `Evaluation Contract V1` (`manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`)  
**Data Provenance:** Staged Authorized Zip SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`  
**Execution Context:** CPU FP32 Deterministic Replay, Batch Size 1, Frozen Acoustic Features  

---

## 1. Executive Summary & Core Research Questions

This benchmark investigates why Subtype Supervision (`T-OP`) exhibits extremely low omission precision on the L2-ARCTIC validation cohort, despite achieving high Average Precision ($\text{AP}_{\text{omit}} = 0.0986$ on Protocol Z, $+207.6\%$ relative to Baseline R E0) and high recall ($\text{Recall@FPR15} = 56.55\%$ on genuine prompt-only Protocol Z and $73.10\%$ on oracle Protocol O).

### Key Findings
1. **Severe Imbalance Dominance:** The validation cohort exhibits an extreme class imbalance: out of 8,911 omission-evaluable positions under Protocol Z, only 137 are gold omissions (prevalence **$1.54\%$**). Consequently, even an achieved False Positive Rate (FPR) of $12.14\%$ generates **1,068 false alarms on correct phones**, overpowering the 82 true positives and driving subset omission precision down to **$7.13\%$** ($82 / 1,150$).
2. **Deployed Event vs Subset Precision:** When substitution negatives are included (a deployed scenario where a learner substitutes a phone, but the detector mistakenly fires an omission alert), an additional 59 false alarms occur, yielding a deployed event precision of **$6.78\%$** ($82 / 1,209$).
3. **Supported Score Semantics:** 
   - `s_omit` (T-OP dedicated omission head): Measures identified omission probability. At $15\%$ CAL ceiling, it detects 82 true positives with 1,068 correct false alarms and only 59 substitution false alarms.
   - `s_error` (Legacy binary error score, shared by T-OP and B-Control): Flags general error mass. Because B-Control's subtype logits are unidentified, it must rely on `s_error`. At $15\%$ CAL ceiling, B-Control legacy detects only 41 omissions (recall $28.28\%$) while firing 1,056 correct false alarms and **488 substitution false alarms**, yielding a deployed omission precision of only **$2.59\%$** ($41 / 1,585$).
4. **Precision–Recall Frontier Ceilings:** On retrospective validation data, a precision target of $\ge 30\%$ is attainable only up to a recall of **$2.76\%$** (4 detections); a precision target of $\ge 50\%$ is attainable only for a single detection (**$0.69\%$ recall**); precision targets of $\ge 60\%$ and $\ge 70\%$ are **completely unattainable** across all thresholds.

---

## 2. Full Cohort Accounting & Protocol Closure

To avoid selective reporting, all 9,956 gold positions in Fold-1 VAL (and 9,993 total evaluation records) are strictly accounted for:

| Partition Stratum | Protocol O (Oracle Canonical) | Protocol Z (Genuine Prompt-Only) | Accounting Treatment |
|---|---:|---:|---|
| **Aligned Correct ($y=0$)** | 8,799 | 8,774 | Evaluated as True Negatives or False Positives |
| **Aligned Substitutions ($y=1$)** | 1,012 | 1,008 | Separate negative pool for omission; positive for substitution |
| **Aligned Omissions ($y=2$)** | 145 | 137 | True Positives or False Negatives |
| **Unaligned Gold Abstentions** | 0 | 37 (25 cor, 4 sub, 8 del) | Accounted as Missed Positives (FN) or No-Alarm Negatives (TN) |
| **Extra Prompt Positions** | 0 | 37 | Extra prompt tokens unaligned to gold; excluded from gold AP |
| **Total Gold Positions** | **9,956** | **9,956** | Fully reconciled ($9,919\text{ aligned} + 37\text{ abstentions}$) |
| **Total Evaluation Records** | **9,956** | **9,993** | $9,919\text{ aligned} + 37\text{ unmapped gold} + 37\text{ extra prompt}$ |

In all tables below:
- **Aligned Population:** $N=9,919$ ($8,774\text{ correct}, 1,008\text{ substitution}, 137\text{ omission}$).
- **All-Gold Population:** $N=9,956$ ($8,799\text{ correct}, 1,012\text{ substitution}, 145\text{ omission}$).
- Unaligned gold omissions ($N=8$) are treated as False Negatives in all-gold operating point calculations.
- Unaligned gold correct ($N=25$) and substitutions ($N=4$) are counted as No-Alarm True Negatives in primary confusion tables, with FPR upper bounds disclosing them.

---

## 3. Calibrated Operating Points Across 7 FPR Ceilings

Thresholds $\tau$ were derived **strictly on Calibration (CAL) correct scores** from speakers NCC and THV (300 utterances, $N=8,036$ correct scores under Protocol Z; $N=8,054$ under Protocol O). **Zero threshold tuning was performed on the Validation (VAL) set.**

The target ceilings are: **$0.1\%$, $0.5\%$, $1.0\%$, $2.0\%$, $5.0\%$, $10.0\%$, $15.0\%$**.

### Table 3.1: Protocol Z — T-OP Fixed Budget (3,600 updates, `s_omit` dedicated)
*Population: 8,799 gold correct, 1,012 gold substitution, 145 gold omission (all-gold accounting).*

| CAL Ceiling | $\tau_{\text{omit}}$ | Achieved CAL FPR | VAL Aligned TP | VAL All-Gold Recall | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | Total Alarms | Achieved VAL $\text{FPR}_{\text{cor}}$ | Subset Precision | Deployed Precision | False Alarms / 1,000 phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.1%** | 0.247430 | 0.10% | 1 | 0.69% (1/145) | 2 | 0 | 3 | 0.02% | **33.33%** | **33.33%** | 0.2 |
| **0.5%** | 0.205967 | 0.50% | 3 | 2.07% (3/145) | 7 | 0 | 10 | 0.08% | **30.00%** | **30.00%** | 0.7 |
| **1.0%** | 0.184287 | 1.00% | 7 | 4.83% (7/145) | 24 | 0 | 31 | 0.27% | **22.58%** | **22.58%** | 2.4 |
| **2.0%** | 0.165763 | 1.99% | 10 | 6.90% (10/145) | 48 | 1 | 59 | 0.55% | **17.24%** | **16.95%** | 4.9 |
| **5.0%** | 0.133940 | 4.99% | 26 | 17.93% (26/145) | 182 | 7 | 215 | 2.07% | **12.50%** | **12.09%** | 19.0 |
| **10.0%** | 0.097335 | 9.99% | 60 | 41.38% (60/145) | 571 | 28 | 659 | 6.49% | **9.51%** | **9.10%** | 60.2 |
| **15.0%** | 0.066045 | 15.00% | 82 | 56.55% (82/145) | 1,068 | 59 | 1,209 | 12.14% | **7.13%** | **6.78%** | 113.2 |

*Note:* Aligned recall at 15% is $82 / 137 = 59.85\%$; all-gold recall including 8 unaligned omissions is $82 / 145 = 56.55\%$.

---

### Table 3.2: Protocol Z — B-Control Fixed Budget (3,600 updates, `s_error` legacy)
*Matched binary control arm where subtype logits are unidentified.*

| CAL Ceiling | $\tau_{\text{error}}$ | Achieved CAL FPR | VAL Aligned TP | VAL All-Gold Recall | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | Total Alarms | Achieved VAL $\text{FPR}_{\text{cor}}$ | Subset Precision | Deployed Precision | False Alarms / 1,000 phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.1%** | 0.571745 | 0.10% | 0 | 0.00% (0/145) | 6 | 1 | 7 | 0.07% | 0.00% | 0.00% | 0.7 |
| **0.5%** | 0.548440 | 0.50% | 0 | 0.00% (0/145) | 19 | 12 | 31 | 0.22% | 0.00% | 0.00% | 3.1 |
| **1.0%** | 0.525835 | 1.00% | 0 | 0.00% (0/145) | 50 | 39 | 89 | 0.57% | 0.00% | 0.00% | 8.9 |
| **2.0%** | 0.488415 | 1.99% | 2 | 1.38% (2/145) | 151 | 141 | 294 | 1.72% | 1.31% | 0.68% | 29.3 |
| **5.0%** | 0.313781 | 4.99% | 7 | 4.83% (7/145) | 463 | 345 | 815 | 5.26% | 1.49% | 0.86% | 81.2 |
| **10.0%** | 0.263482 | 9.99% | 18 | 12.41% (18/145) | 688 | 384 | 1,090 | 7.82% | 2.55% | 1.65% | 107.7 |
| **15.0%** | 0.223345 | 15.00% | 41 | 28.28% (41/145) | 1,056 | 488 | 1,585 | 12.00% | 3.74% | 2.59% | 155.1 |

---

### Table 3.3: Protocol Z — T-OP Fixed Budget (`s_error` legacy scoring)
*Evaluating T-OP under the identical legacy scoring rule as B-Control.*

| CAL Ceiling | $\tau_{\text{error}}$ | Achieved CAL FPR | VAL Aligned TP | VAL All-Gold Recall | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | Total Alarms | Achieved VAL $\text{FPR}_{\text{cor}}$ | Subset Precision | Deployed Precision | False Alarms / 1,000 phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.1%** | 0.557259 | 0.10% | 0 | 0.00% (0/145) | 2 | 1 | 3 | 0.02% | 0.00% | 0.00% | 0.3 |
| **0.5%** | 0.531387 | 0.50% | 0 | 0.00% (0/145) | 20 | 8 | 28 | 0.23% | 0.00% | 0.00% | 2.8 |
| **1.0%** | 0.513248 | 1.00% | 0 | 0.00% (0/145) | 37 | 28 | 65 | 0.42% | 0.00% | 0.00% | 6.5 |
| **2.0%** | 0.468254 | 1.99% | 2 | 1.38% (2/145) | 146 | 142 | 290 | 1.66% | 1.35% | 0.69% | 28.9 |
| **5.0%** | 0.319003 | 4.99% | 10 | 6.90% (10/145) | 454 | 344 | 808 | 5.16% | 2.16% | 1.24% | 80.2 |
| **10.0%** | 0.272109 | 9.99% | 25 | 17.24% (25/145) | 713 | 408 | 1,146 | 8.10% | 3.39% | 2.18% | 112.6 |
| **15.0%** | 0.235092 | 15.00% | 41 | 28.28% (41/145) | 1,039 | 480 | 1,560 | 11.81% | 3.80% | 2.63% | 152.6 |

---

### Table 3.4: Comparison Across Selection Checkpoints (`best.pt`) at Nominal 15% CAL Ceiling

| Protocol | Model Configuration | Score Key | $\tau$ | TP | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | All-Gold Recall | Subset Precision | Deployed Precision | VAL $\text{FPR}_{\text{cor}}$ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Z** | **T-OP Fixed (3,600 up)** | `s_omit` | 0.066045 | **82** | 1,068 | **59** | **56.55%** | **7.13%** | **6.78%** | 12.14% |
| **Z** | **T-OP Best (2,700 up)** | `s_omit` | 0.053000 | 81 | 1,027 | 64 | 55.86% | 7.31% | 6.91% | 11.67% |
| **Z** | **B-Control Fixed (3,600 up)** | `s_error` | 0.223345 | 41 | 1,056 | 488 | 28.28% | 3.74% | 2.59% | 12.00% |
| **Z** | **B-Control Best (2,250 up)** | `s_error` | 0.217067 | 24 | 897 | 451 | 16.55% | 2.61% | 1.75% | 10.19% |
| **O** | **T-OP Fixed (3,600 up)** | `s_omit` | 0.066318 | **91** | 1,083 | **66** | **62.76%** | **7.75%** | **7.34%** | 12.31% |
| **O** | **T-OP Best (2,700 up)** | `s_omit` | 0.054015 | 89 | 1,038 | 68 | 61.38% | 7.90% | 7.45% | 11.80% |
| **O** | **B-Control Fixed (3,600 up)** | `s_error` | 0.223545 | 52 | 1,069 | 496 | 35.86% | 4.64% | 3.22% | 12.15% |
| **O** | **B-Control Best (2,250 up)** | `s_error` | 0.213850 | 32 | 931 | 448 | 22.07% | 3.32% | 2.27% | 10.58% |

---

## 4. Retrospective Precision–Recall Frontiers & Target Milestones

Using dense empirical precision-recall curve analysis across the entire score range on Protocol Z, we evaluated whether higher precision can be achieved by raising the decision threshold.

### Table 4.1: Retrospective Precision Milestones for Omission Detection

| Model / Score Family | Max Observed Precision | Recall at Target $P \ge 30\%$ | Recall at Target $P \ge 50\%$ | Recall at Target $P \ge 70\%$ |
|---|---:|---:|---:|---:|
| **T-OP Fixed (`s_omit`)** | **50.00%** (at $\tau = 0.2528$) | **2.76%** (4/145, $P=33.33\%$) | **0.69%** (1/145, $P=50.00\%$) | **NOT ATTAINABLE** |
| **T-OP Best (`s_omit`)** | **42.86%** (at $\tau = 0.2173$) | **2.07%** (3/145, $P=37.50\%$) | **NOT ATTAINABLE** | **NOT ATTAINABLE** |
| **B-Control Fixed (`s_error`)** | **14.29%** (at $\tau = 0.5822$) | **NOT ATTAINABLE** | **NOT ATTAINABLE** | **NOT ATTAINABLE** |
| **B-Control Best (`s_error`)** | **11.11%** (at $\tau = 0.6120$) | **NOT ATTAINABLE** | **NOT ATTAINABLE** | **NOT ATTAINABLE** |
| **T-OP Fixed (`s_error`)** | **14.29%** (at $\tau = 0.5815$) | **NOT ATTAINABLE** | **NOT ATTAINABLE** | **NOT ATTAINABLE** |

### Mathematical Frontier Analysis
Why is $P \ge 60\%$ or $\ge 70\%$ unattainable for omission?
- The model outputs highest omission logits ($s_{\text{omit}} \ge 0.20$) on a mixture of true deletions and unreleased/reduced coronal consonants.
- The single highest score in the dataset is $s_{\text{omit}} = 0.2902$ on an aligned gold `CORRECT` token (`RRBI arctic_a0384 D`), followed by $s_{\text{omit}} = 0.2528$ on a gold `OMIT` token (`RRBI arctic_a0384 D`), followed immediately by three gold `CORRECT` tokens ($s_{\text{omit}} = 0.2482, 0.2443, 0.2204$).
- Because false positives contaminate even the topmost order statistics of the predictor, precision cannot climb above $50.0\%$ at any threshold, regardless of how aggressively thresholding is tuned.

---

## 5. False Alarms per 1,000 Phones & Deployed Error Burden

In practical computer-assisted pronunciation training (CAPT), learners cannot tolerate high false alarm frequencies. False alarms per 1,000 phones are computed as:
$$\text{FA}_{1000} = \frac{\text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}}{N_{\text{total gold}}} \times 1,000$$

- At **$1.0\%$ CAL ceiling**: T-OP generates **2.4 false alarms per 1,000 phones** (31 total alarms across 300 utterances, $\sim 0.10$ false alarms per sentence), while detecting 7 true omissions.
- At **$5.0\%$ CAL ceiling**: T-OP generates **19.0 false alarms per 1,000 phones** (215 alarms across 300 utterances, $\sim 0.72$ false alarms per sentence), detecting 26 omissions.
- At **$15.0\%$ CAL ceiling**: T-OP generates **113.2 false alarms per 1,000 phones** (1,209 alarms across 300 utterances, **$4.03$ false alarms per sentence**), detecting 82 omissions.

### Pedagogical Verdict
A false alarm rate of 4 false alarms per sentence ($113.2 / 1\text{k}$) destroys learner trust in an automated feedback system. Selective prediction or threshold raising alone can suppress false alarms down to 2.4 / 1k, but only at the cost of collapsing omission recall from $56.55\%$ down to $4.83\%$.

---

## 6. Audit Trail & Verification Manifest

- Full numerical curves and calibration points are preserved in `PRECISION_RECALL_FRONTIER.json`.
- All 9,993 token evaluations are indexed in `CASE_MANIFEST.json`.
- SHA-256 verification of inputs:
  - Staged zip: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`
  - Protocol Z offline evaluation: `e2c84eeb1d12dd3a48e8941783cf6551b9e28f3ea7f4d4453b30fa8c8da029b9`
  - Protocol Z runtime predictions: `79c8d5040a6c22c0d16550774a3f25c798991206103e6de5ce57c7b0dcb940e4`
- Audit script: `audit_scripts/compute_precision_benchmark.py` (executed deterministically on CPU FP32).
