# LANGFENS RQ1-TRN-04 — Precision–Recall Tradeoff & Operating Point Analysis

**Document ID:** `LANGFENS-RQ1-TRN-04-PR-TRADEOFF-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist & Evaluator  
**Date:** 2026-11-04  
**Cohort:** L2-ARCTIC Fold-1 VAL (Protocol Z, 300 utterances, 9,956 gold canonical tokens)  
**Calibration Platform:** L2-ARCTIC Fold-1 CAL (NCC, THV, 300 utterances, 8,054 correct tokens)  

---

## 1. Operating Point Confusion Matrices across CAL Ceilings

To preserve rigorous non-contaminating evaluation, all operating thresholds ($\tau_{\text{CAL}}$) were determined prospectively on the independent CAL partition (NCC/THV) using only correct-token scores ($y=0$). These thresholds were then evaluated blindly on Fold-1 VAL (NJS/RRBI):

### 1.1 CAL 15% Ceiling (Nominal Operational Target)
- **CAL Target:** Allow up to $15.0\%$ False Positive Rate on CAL correct tokens.

| Arm | $\tau_{\text{CAL}}$ | TP | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | TN | FN | Deployed Prec | All-Gold Rec | Aligned Rec | FA / 1k Phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C0** | 0.058205 | 87 | 1,007 | 59 | 7,792 | 58 | **7.55%** | **60.00%** | 63.50% | 107.07 |
| **T1** | 0.106096 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |
| **T2** | 0.061087 | 74 | 914 | 55 | 7,885 | 71 | **7.09%** | **51.03%** | 54.01% | 97.33 |
| **T3** | 0.104263 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |
| *T-OP Ref* | 0.066045 | 82 | 1,068 | 59 | 7,731 | 63 | 6.78% | 56.55% | 59.85% | 113.20 |

### 1.2 CAL 10% Ceiling (Intermediate Threshold)
- **CAL Target:** Allow up to $10.0\%$ False Positive Rate on CAL correct tokens.

| Arm | $\tau_{\text{CAL}}$ | TP | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | TN | FN | Deployed Prec | All-Gold Rec | Aligned Rec | FA / 1k Phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C0** | 0.091770 | 58 | 502 | 26 | 8,297 | 87 | **9.90%** | **40.00%** | 42.34% | 53.03 |
| **T1** | 0.122748 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |
| **T2** | 0.092321 | 49 | 468 | 28 | 8,331 | 96 | **8.99%** | **33.79%** | 35.77% | 49.82 |
| **T3** | 0.121685 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |

### 1.3 CAL 5% Ceiling (Conservative Precision-Biased Threshold)
- **CAL Target:** Allow up to $5.0\%$ False Positive Rate on CAL correct tokens.

| Arm | $\tau_{\text{CAL}}$ | TP | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | TN | FN | Deployed Prec | All-Gold Rec | Aligned Rec | FA / 1k Phones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C0** | 0.135848 | 29 | 171 | 6 | 8,628 | 116 | **14.08%** | **20.00%** | 21.17% | 17.78 |
| **T1** | 0.138860 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |
| **T2** | 0.132617 | 18 | 130 | 7 | 8,669 | 127 | **11.61%** | **12.41%** | 13.14% | 13.76 |
| **T3** | 0.140210 | 145 | 8,799 | 1,012 | 0 | 0 | **1.46%** | **100.00%** | 100.00% | 985.44 |

---

## 2. Analysis of the Fundamental Concave Tradeoff

### 2.1 The Precision-Destruction Gradient in C0 and T2
As thresholds are raised from CAL 15% to CAL 5%:
1. **In Arm C0:**
   - Precision rises from $7.55\% \to 9.90\% \to 14.08\%$ ($+6.53\text{ pp}$).
   - But recall collapses from $60.00\% \to 40.00\% \to 20.00\%$ (**$-40.00\text{ pp}$ loss, destroying $66.7\%$ of all true omissions**).
   - Exchange rate: To gain $+1.0\text{ pp}$ of deployed precision, the system sacrifices **$-6.12\text{ pp}$ of true omission recall**.
2. **In Arm T2:**
   - Precision rises from $7.09\% \to 8.99\% \to 11.61\%$ ($+4.52\text{ pp}$).
   - Recall collapses from $51.03\% \to 33.79\% \to 12.41\%$ (**$-38.62\text{ pp}$ loss, destroying $75.7\%$ of true omissions**).
   - Exchange rate: To gain $+1.0\text{ pp}$ of precision, T2 sacrifices **$-8.54\text{ pp}$ of recall**.
3. **Comparative Efficiency:**
   - At every single threshold ceiling ($5\%, 10\%, 15\%$), **Arm C0 achieves strictly superior precision AND strictly superior recall compared to Arm T2**:
     - At CAL 15%: C0 achieves $7.55\%$ prec / $60.0\%$ rec vs T2's $7.09\%$ prec / $51.0\%$ rec.
     - At CAL 10%: C0 achieves $9.90\%$ prec / $40.0\%$ rec vs T2's $8.99\%$ prec / $33.8\%$ rec.
     - At CAL 5%: C0 achieves $14.08\%$ prec / $20.0\%$ rec vs T2's $11.61\%$ prec / $12.4\%$ rec.
   - **Conclusion:** **Arm C0 strictly Pareto-dominates Arm T2 across the entire operational calibration frontier.**

### 2.2 The Pathological Logit Collapse in T1 and T3
In both focal-loss arms (T1 and T3), the precision–recall curve underwent complete structural degeneration:
- The CAL correct token score percentiles were compressed into the range $[0.10, 0.14]$.
- However, the neural head assigned omission probabilities $\ge 0.15$ to almost the entire test cohort.
- Consequently, at all three calibrated operating points (CAL 5%, 10%, 15%), **100% of all 9,956 tokens triggered an omission alarm**, resulting in the exact base-rate precision of the dataset:
  $$\text{Base Rate Precision} = \frac{145}{9,956} = 1.456\% \approx 1.46\%$$
- This proves that class-conditioned focal loss cannot be calibrated with standard correct-token percentile ceilings.
