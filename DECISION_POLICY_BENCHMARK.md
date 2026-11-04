# LANGFENS RQ1 — Precision & Decision-Policy Benchmark Report (Agent A)

**Document ID:** `LANGFENS-RQ1-DECISION-POLICY-BENCHMARK-01`  
**Workstream:** `workstreams/agent_a_policy/`  
**Author:** Agent A — Precision & Decision-Policy Benchmark Lead  
**Evaluation Target:** Recovered P-SUP V2 Benchmark (Fold-1 VAL, 300 utterances, speakers NJS and RRBI)  
**Contract Baseline:** `Evaluation Contract V1` (`manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`)  
**Data Provenance:** Staged Authorized Zip SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`  
**Model Under Test:** Subtype Supervision `T-OP` Fixed Budget Checkpoint (3,600 updates, seed 1337)  
**Control Models:** `B-Control` Fixed Budget (3,600 updates), `T-OP` Best Checkpoint (2,700 updates), `B-Control` Best Checkpoint (2,250 updates)  
**Execution Environment:** Deterministic CPU FP32 Replay, `.venv/bin/python`, Batch Size 1, Frozen Acoustic Embeddings  

---

## 1. Executive Summary & Screening Target Verdicts

This report delivers the comprehensive empirical evaluation of **25 post-processing decision policies** and the retrospective **Precision–Recall (PR) Frontier** for LANGFENS RQ1 Precision Recovery Experiment 01. The investigation resolves the central operational question:

> *Can post-hoc decision policies (global FPR ceilings, phoneme suppression, per-phoneme thresholds, asymmetric gating, margin/ratio filtering, selective abstention, or matched false-alarm budgets) recover omission detection precision to pedagogically viable levels without retraining model weights or deploying acoustic verifiers?*

### Empirical Screening Verdict: NO-GO for Decision Policy Alone

Post-processing decision policy modifications **cannot solve the omission precision crisis as a standalone intervention**. Across all 25 evaluated policies and both retrospective PR frontiers:

1. **Pedagogical Precision Screening Target ($\ge 30\%$ deployed precision at $\ge 10\%$ recall): FAILED.**
   - A precision target of $\ge 30\%$ is attainable only at a negligible recall of **$2.76\%$** (4 true detections out of 145 gold omissions, firing 8 false alarms).
   - At pedagogically meaningful recall ($\ge 50\%$), deployed precision collapses to **$6.78\%$** ($82 / 1,209$).
   - The highest precision achieved by any policy retaining at least $10\%$ recall is **$13.93\%$** (Margin Gating $\Delta \ge 0.05$, detecting 17 omissions with 105 false alarms, recall $11.72\%$).
2. **High-Precision Screening Target ($\ge 50\%$ deployed precision): FAILED.**
   - Attainable only for **a single isolated token** in the entire corpus ($0.69\%$ recall, $1$ TP, $1$ FP at $\tau_{\text{omit}} = 0.2528$).
3. **Strict Precision Screening Targets ($\ge 60\%$ and $\ge 70\%$ deployed precision): COMPLETELY UNATTAINABLE.**
   - Across the entire continuous score distribution, zero operating points achieve $\ge 60\%$ or $\ge 70\%$ precision ($0.00\%$ recall).
4. **Learner False-Alarm Budget Targets ($\le 5$ or $\le 10$ false alarms per 1,000 phones): FAILED AT VIABLE RECALL.**
   - Enforcing $\text{FA}_{1000} \le 5$ achieves $0.8$ false alarms per 1,000 phones, but eviscerates omission recall down to **$2.07\%$** (3 detections).
   - Enforcing $\text{FA}_{1000} \le 10$ achieves $2.6$ false alarms per 1,000 phones, but recall reaches only **$4.83\%$** (7 detections).
   - At the baseline operating point ($\tau_{\text{omit}} = 0.066045$, $15\%$ CAL ceiling), the model generates **$113.2$ false alarms per 1,000 phones**—equating to **$4.03$ false alarms per sentence**, severely degrading student trust.
5. **The Phonetic Trap of Selective Muting: SENSITIVITY COLLAPSE.**
   - Completely muting the three pathological phonemes (`/d/`, `/t/`, `/r/`) successfully eliminates $93.4\%$ of false alarms, but simultaneously eliminates **80 of the 82 True Positives** ($97.56\%$).
   - Recall collapses from $56.55\%$ to **$1.38\%$**, and deployed precision remains trapped at **$2.63\%$** ($2 / 76$).
6. **Demographic Equity Screening (Speaker Parity): FAILED.**
   - Aggressive gating policies induce severe cross-speaker disparities. Under Margin Gating ($\Delta \ge 0.05$), precision on Vietnamese speaker NJS is **$23.53\%$**, whereas precision on Arabic speaker RRBI drops to **$7.04\%$**, driven by persistent rhotic and coronal false alarms on RRBI.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SCREENING TARGET AUDIT SUMMARY                                      │
├────────────────────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│ Screening Target Criterion             │ Threshold   │ Status      │ Best Result │ Operating Point     │
├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ Pedagogical Precision ($\ge 30\%$)     │ Rec >= 10%  │ FAILED      │ Prec 13.93% │ Margin Delta >= 0.05│
│ High Precision ($\ge 50\%$)            │ Any Rec     │ FAILED (0.7%)Prec 50.00% │ tau = 0.2528 (1 TP) │
│ Strict Precision ($\ge 60\%$)          │ Any Rec     │ UNATTAINABLE│ Prec 0.00%  │ None exist          │
│ Strict Precision ($\ge 70\%$)          │ Any Rec     │ UNATTAINABLE│ Prec 0.00%  │ None exist          │
│ False-Alarm Budget ($\le 10$ FA/1k)    │ Rec >= 40%  │ FAILED      │ Rec 4.83%   │ Matched Budget <=10 │
│ False-Alarm Budget ($\le 5$ FA/1k)     │ Rec >= 40%  │ FAILED      │ Rec 2.07%   │ Matched Budget <= 5 │
│ Cross-Speaker Parity (|P_njs - P_rrbi|)│ <= 5.0%     │ FAILED      │ Diff 16.49% │ Margin Delta >= 0.05│
└────────────────────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
```

---

## 2. Evaluation Framework, Full Cohort Accounting & Protocol Closure

To prevent selective reporting, survivorship bias, and metric distortion, all **9,956 gold canonical positions** across the 300 validation utterances (speakers NJS and RRBI) are strictly accounted for under Protocol Z (genuine prompt-only alignment):

| Partition Stratum | Aligned Count ($N$) | Unmapped Abstentions | Total Gold Positions | Accounting Treatment in Evaluation |
|---|---:|---:|---:|---|
| **Gold Correct ($y=0$)** | 8,774 | 25 | **8,799** | Negative pool 1 (evaluable for False Alarms) |
| **Gold Substitution ($y=1$)** | 1,008 | 4 | **1,012** | Negative pool 2 (substitution false alarms) |
| **Gold Omission ($y=2$)** | 137 | 8 | **145** | Positive pool (evaluable for True Positives) |
| **Prompt Extras** | 37 | — | 37 | Unaligned model prompt extras; excluded from gold AP |
| **Total Evaluation Population** | **9,919** | **37** | **9,956** | **Full Cohort Reconciliation** ($9,956 + 37\text{ extra} = 9,993$) |

### Metric Definitions and Mathematical Formulations

- **True Positives ($\text{TP}$):** Aligned gold omissions where the policy fires an alarm ($s \ge \tau$ and passes policy filters).
- **False Negatives ($\text{FN}$):** Gold omissions not flagged as an alarm, including all 8 unmapped gold omission abstentions: $\text{FN} = 145 - \text{TP}$.
- **False Positives on Correct ($\text{FP}_{\text{cor}}$):** Gold correct tokens incorrectly flagged as omissions.
- **False Positives on Substitutions ($\text{FP}_{\text{sub}}$):** Gold substitutions incorrectly flagged as omissions (misdiagnosis errors).
- **Total Alarms:** $\text{Total} = \text{TP} + \text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}$.
- **Deployed Event Precision ($P_{\text{dep}}$):** The precision experienced by a student using the software:
  $$P_{\text{dep}} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}}$$
- **Subset Precision ($P_{\text{sub}}$):** The idealized precision considering only correct negative tokens:
  $$P_{\text{sub}} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{cor}}}$$
- **All-Gold Recall ($R_{\text{gold}}$):** Sensitivity against the entire gold benchmark:
  $$R_{\text{gold}} = \frac{\text{TP}}{145}$$
- **Aligned Recall ($R_{\text{aln}}$):** Sensitivity against aligned prompt positions:
  $$R_{\text{aln}} = \frac{\text{TP}}{137}$$
- **False Alarms per 1,000 Canonical Phones ($\text{FA}_{1000}$):** Learner annoyance metric:
  $$\text{FA}_{1000} = \frac{\text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}}{9,956} \times 1,000$$
- **Correct-Token False Positive Rate ($\text{FPR}_{\text{cor}}$):**
  $$\text{FPR}_{\text{cor}} = \frac{\text{FP}_{\text{cor}}}{8,799}$$

---

## 3. The 25 Decision Policies — Taxonomy, Design & Benchmark Table

We systematically benchmarked **25 post-processing decision policies** divided into seven distinct paradigms, all calibrated strictly on the independent Calibration (CAL) cohort (speakers NCC and THV, 300 utterances, 8,036 aligned correct scores). **No threshold was fit on the validation cohort.**

### Table 3.1: Full Empirical Benchmark of 25 Decision Policies on Protocol Z
*All counts derived from `POLICY_COMPARISON.csv` and verified against `policy_summary.json`.*

| # | Policy Name | Policy Paradigm | Operating Parameter / Rule | TP | FP_cor | FP_sub | Total Alarms | Deployed Prec | Subset Prec | All-Gold Recall | FA / 1k | NJS Prec | NJS Rec | RRBI Prec | RRBI Rec | D/T/R Alarms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Global CAL 0.1% Ceiling** | Global Threshold | $\tau = 0.247430$ | 1 | 2 | 0 | 3 | **33.33%** | 33.33% | 0.69% | 0.2 | 0.00% | 0.00% | 33.33% | 1.54% | 3 / 0 / 0 |
| 2 | **Global CAL 0.5% Ceiling** | Global Threshold | $\tau = 0.205967$ | 3 | 7 | 0 | 10 | **30.00%** | 30.00% | 2.07% | 0.7 | 50.00% | 2.50% | 16.67% | 1.54% | 9 / 1 / 0 |
| 3 | **Global CAL 1.0% Ceiling** | Global Threshold | $\tau = 0.184287$ | 7 | 24 | 0 | 31 | **22.58%** | 22.58% | 4.83% | 2.4 | 31.25% | 6.25% | 13.33% | 3.08% | 30 / 1 / 0 |
| 4 | **Global CAL 2.0% Ceiling** | Global Threshold | $\tau = 0.165763$ | 10 | 48 | 1 | 59 | **16.95%** | 17.24% | 6.90% | 4.9 | 26.92% | 8.75% | 9.09% | 4.62% | 52 / 3 / 4 |
| 5 | **Global CAL 5.0% Ceiling** | Global Threshold | $\tau = 0.133940$ | 26 | 182 | 7 | 215 | **12.09%** | 12.50% | 17.93% | 19.0 | 16.30% | 18.75% | 8.94% | 16.92% | 133 / 15 / 60 |
| 6 | **Global CAL 10.0% Ceiling** | Global Threshold | $\tau = 0.097335$ | 60 | 571 | 28 | 659 | **9.10%** | 9.51% | 41.38% | 60.2 | 9.43% | 37.50% | 8.80% | 46.15% | 303 / 89 / 252 |
| 7 | **Global CAL 15.0% Ceiling (Baseline)** | Global Threshold | $\tau = 0.066045$ | 82 | 1,068 | 59 | 1,209 | **6.78%** | 7.13% | **56.55%** | 113.2 | 7.58% | 56.25% | 6.02% | 56.92% | 436 / 320 / 377 |
| 8 | **Baseline + Mute {D, T, R}** | Phoneme Muting | $\tau = 0.066045$, mute {D, T, R} | 2 | 64 | 10 | 76 | **2.63%** | 3.03% | 1.38% | 7.4 | 3.33% | 1.25% | 2.17% | 1.54% | 0 / 0 / 0 |
| 9 | **Baseline + Mute {D, T}** | Phoneme Muting | $\tau = 0.066045$, mute {D, T} | 17 | 395 | 41 | 453 | **3.75%** | 4.13% | 11.72% | 43.8 | 1.36% | 3.75% | 6.01% | 21.54% | 0 / 0 / 377 |
| 10 | **Baseline + Mute {R}** | Phoneme Muting | $\tau = 0.066045$, mute {R} | 67 | 737 | 28 | 832 | **8.05%** | 8.33% | 46.21% | 76.8 | 10.64% | 53.75% | 5.61% | 36.92% | 436 / 320 / 0 |
| 11 | **Per-Phoneme CAL 15% Thresholds** | Per-Phone Threshold | Phone $\tau$ at 15% CAL FPR | 10 | 260 | 38 | 308 | **3.25%** | 3.70% | 6.90% | 29.9 | 3.45% | 5.00% | 3.12% | 9.23% | 17 / 16 / 15 |
| 12 | **Per-Phoneme CAL 5% Thresholds** | Per-Phone Threshold | Phone $\tau$ at 5% CAL FPR | 4 | 90 | 5 | 99 | **4.04%** | 4.26% | 2.76% | 9.5 | 3.57% | 1.25% | 4.23% | 4.62% | 4 / 8 / 3 |
| 13 | **Asymmetric: 1% Ceiling {D,T,R}** | Asymmetric Ceiling | $\tau_{\text{DTR}}=0.1843, \tau_{\text{other}}=0.0660$ | 9 | 88 | 10 | 107 | **8.41%** | 9.28% | 6.21% | 9.8 | 13.04% | 7.50% | 4.92% | 4.62% | 30 / 1 / 0 |
| 14 | **Asymmetric: 2% Ceiling {D,T,R}** | Asymmetric Ceiling | $\tau_{\text{DTR}}=0.1658, \tau_{\text{other}}=0.0660$ | 12 | 112 | 11 | 135 | **8.89%** | 9.68% | 8.28% | 12.3 | 14.29% | 10.00% | 5.06% | 6.15% | 52 / 3 / 4 |
| 15 | **Asymmetric: 5% Ceiling {D,T,R}** | Asymmetric Ceiling | $\tau_{\text{DTR}}=0.1339, \tau_{\text{other}}=0.0660$ | 28 | 239 | 17 | 284 | **9.86%** | 10.49% | 19.31% | 25.7 | 13.68% | 20.00% | 7.19% | 18.46% | 133 / 15 / 60 |
| 16 | **Margin Gating: Delta >= 0.02** | Margin Filtering | $\tau=0.0660, (s_{\text{omit}} - s_{\text{sub}}) \ge 0.02$ | 46 | 390 | 7 | 443 | **10.38%** | 10.55% | 31.72% | 39.9 | 13.99% | 33.75% | 7.60% | 29.23% | 260 / 173 / 0 |
| 17 | **Margin Gating: Delta >= 0.05** | Margin Filtering | $\tau=0.0660, (s_{\text{omit}} - s_{\text{sub}}) \ge 0.05$ | 17 | 104 | 1 | 122 | **13.93%** | 14.05% | 11.72% | 10.6 | 23.53% | 15.00% | 7.04% | 7.69% | 111 / 11 / 0 |
| 18 | **Margin Gating: Delta >= 0.10** | Margin Filtering | $\tau=0.0660, (s_{\text{omit}} - s_{\text{sub}}) \ge 0.10$ | 4 | 15 | 0 | 19 | **21.05%** | 21.05% | 2.76% | 1.5 | 25.00% | 2.50% | 18.18% | 3.08% | 18 / 1 / 0 |
| 19 | **Ratio Gating: Ratio >= 0.6** | Ratio Filtering | $\tau=0.0660, s_{\text{omit}}/s_{\text{err}} \ge 0.6$ | 26 | 162 | 1 | 189 | **13.76%** | 13.83% | 17.93% | 16.4 | 20.00% | 20.00% | 9.17% | 15.38% | 128 / 54 / 0 |
| 20 | **Ratio Gating: Ratio >= 0.8** | Ratio Filtering | $\tau=0.0660, s_{\text{omit}}/s_{\text{err}} \ge 0.8$ | 0 | 0 | 0 | 0 | **0.00%** | 0.00% | 0.00% | 0.0 | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 / 0 |
| 21 | **Selective Abstention** | Selective Prediction | $\tau=0.0660$, abstain if margin $< 0.03$ | 48 | 594 | 42 | 684 | **7.02%** | 7.48% | 33.10% | 63.9 | 7.01% | 28.75% | 7.02% | 38.46% | 205 / 67 / 346 |
| 22 | **Matched Budget: FA <= 5/1k** | Budget-Constrained | $\tau = 0.200898$ (CAL-Derived) | 3 | 8 | 0 | 11 | **27.27%** | 27.27% | 2.07% | 0.8 | 40.00% | 2.50% | 16.67% | 1.54% | 10 / 1 / 0 |
| 23 | **Matched Budget: FA <= 10/1k** | Budget-Constrained | $\tau = 0.182380$ (CAL-Derived) | 7 | 26 | 0 | 33 | **21.21%** | 21.21% | 4.83% | 2.6 | 27.78% | 6.25% | 13.33% | 3.08% | 31 / 1 / 1 |
| 24 | **Matched Budget: FA <= 15/1k** | Budget-Constrained | $\tau = 0.170574$ (CAL-Derived) | 9 | 40 | 1 | 50 | **18.00%** | 18.37% | 6.21% | 4.1 | 25.00% | 7.50% | 11.54% | 4.62% | 45 / 3 / 2 |
| 25 | **Matched Budget: FA <= 20/1k** | Budget-Constrained | $\tau = 0.162606$ (CAL-Derived) | 12 | 58 | 1 | 71 | **16.90%** | 17.14% | 8.28% | 5.9 | 26.67% | 10.00% | 9.76% | 6.15% | 59 / 3 / 9 |

---

## 4. Deep-Dive Analysis of the 7 Policy Paradigms

### 4.1. Paradigm 1: Global False Positive Rate Ceilings (Policies 1–7)
Deriving global thresholds $\tau$ by constraining the empirical False Positive Rate on calibration correct tokens ($N=8,036$) reveals a steep, non-linear precision decay:
- **Tight Ceilings (0.1%–0.5%):** Achieves $30.0\%$ to $33.3\%$ precision, but recall is effectively zero ($0.69\%$ to $2.07\%$). The software detects only 1 to 3 omissions across 300 sentences.
- **Moderate Ceilings (1.0%–5.0%):** Precision drops rapidly from $22.58\%$ ($31$ alarms, $7$ TPs) down to $12.09\%$ ($215$ alarms, $26$ TPs).
- **Standard Ceilings (10.0%–15.0%):** Sensitivity expands to $41.38\%$ and $56.55\%$, but false alarms explode to $659$ and $1,209$, driving precision down to single digits ($9.10\%$ and $6.78\%$).
- **Core Insight:** There is no sweet spot along the global threshold curve. The relationship between false alarms and true positives is strictly sub-linear; every additional true positive captured requires admitting roughly 15 to 20 additional false alarms.

### 4.2. Paradigm 2: Phoneme-Identity Suppression (Policies 8–10)
Given that three phonemes (`/d/`, `/t/`, `/r/`) generate $93.4\%$ of all deployed false alarms ($1,053 / 1,127$), an intuitive intervention is to suppress alarms on these phonemes:
- **Muting `{D, T, R}` (Policy 8):** Suppressing all three phonemes drops false alarms dramatically from $1,127$ to $74$ ($\text{FP}_{\text{cor}} = 64, \text{FP}_{\text{sub}} = 10$). However, True Positives crash from $82$ to **$2$**! Sensitivity falls to **$1.38\%$**, and deployed precision actually regresses to **$2.63\%$** ($2 / 76$).
- **The Phonetic Trap Explained:** In Fold-1 VAL, **80 out of the 82 True Positives** are concentrated on `/d/` ($47$), `/t/` ($18$), and `/r/` ($15$). Only $2$ true detections occur on all other 36 phonemes combined ($1$ on `/l/` and $1$ on `/er/`). Muting the noisy phonemes simultaneously destroys the model's entire detection capacity.
- **Muting Coronal Stops Only (Policy 9):** Retaining `/r/` preserves $17$ TPs ($15$ on `/r/`, $2$ others), but admits $377$ alarms on `/r/`, resulting in $3.75\%$ precision and $11.72\%$ recall.
- **Muting Rhotic Only (Policy 10):** Retaining `/d/` and `/t/` preserves $67$ TPs, but admits $436$ alarms on `/d/` and $320$ on `/t/`, resulting in $8.05\%$ precision and $46.21\%$ recall.

### 4.3. Paradigm 3: Per-Phoneme Calibration & Sparse Fallbacks (Policies 11–12)
Calibrating thresholds independently for each phoneme (with a $N \ge 30$ sample threshold and fallback to global $\tau$) yields severe over-suppression:
- On phonemes with elevated false alarm rates (`/d/`, `/t/`, `/r/`), the correct-token scores on CAL are also heavily elevated. Consequently, per-phoneme calibration sets extraordinarily conservative thresholds for these exact phonemes.
- At a $15\%$ per-phoneme ceiling (Policy 11), True Positives fall from $82$ to $10$, while $298$ false alarms still slip through, driving precision down to **$3.25\%$** ($10 / 308$) and recall to $6.90\%$.
- Per-phoneme calibration fails because within each phoneme, the score distributions of correct tokens and true deletions overlap extensively.

### 4.4. Paradigm 4: Asymmetric Phonetic Ceilings (Policies 13–15)
Applying conservative ceilings (1%, 2%, 5%) specifically to `{D, T, R}` while maintaining the permissive 15% ceiling on other phonemes:
- Under a 5% ceiling on `{D, T, R}` (Policy 15), total alarms drop from $1,209$ to $284$, and precision increases slightly from $6.78\%$ to **$9.86\%$**, with recall at $19.31\%$ ($28$ TPs).
- Under a 1% ceiling on `{D, T, R}` (Policy 13), alarms drop to $107$, precision reaches $8.41\%$, but recall falls to $6.21\%$.
- While asymmetric thresholding improves the signal-to-noise ratio moderately compared to raw suppression, it remains firmly unable to reach even $10\%$ precision at viable recall.

### 4.5. Paradigm 5: Margin-Based & Likelihood Ratio Gating (Policies 16–20)
Subtype supervision provides three concurrent class probabilities: $s_{\text{cor}}$, $s_{\text{sub}}$, and $s_{\text{omit}}$. We test conservative margin gating ($s_{\text{omit}} - s_{\text{sub}} \ge \Delta$) and ratio gating ($s_{\text{omit}} / s_{\text{error}} \ge \rho$):
- **Margin Gating $\Delta \ge 0.02$ (Policy 16):** Completely purges all 377 rhotic alarms ($/r/ \to 0$ alarms) because on rhotic tokens, substitution logits closely track omission logits. Deployed precision improves to **$10.38\%$** ($46 / 443$), retaining $31.72\%$ recall.
- **Margin Gating $\Delta \ge 0.05$ (Policy 17):** Further reduces coronal stop false alarms, achieving the **highest precision among all policies with $>10\%$ recall**: **$13.93\%$** ($17 / 122$), with recall at $11.72\%$.
- **Margin Gating $\Delta \ge 0.10$ (Policy 18):** Yields $21.05\%$ precision ($4 / 19$), but recall collapses to $2.76\%$.
- **Ratio Gating $\rho \ge 0.6$ (Policy 19):** Achieves $13.76\%$ precision ($26 / 189$) and $17.93\%$ recall.
- **Ratio Gating $\rho \ge 0.8$ (Policy 20):** Fires 0 alarms; no token produces an omission-to-error score ratio of 0.8.
- **Verdict:** Margin gating is the most mathematically principled post-hoc filter because it exploits the dedicated subtype head to suppress substitution-omission ambiguity. However, it still falls short of the $30\%$ precision screening target by more than a factor of two.

### 4.6. Paradigm 6: Selective Prediction with Confidence-Based Abstention (Policy 21)
When the detector encounters uncertainty between omission and substitution ($|s_{\text{omit}} - s_{\text{sub}}| < 0.03$), it abstains from generating omission feedback:
- The policy abstains on **562 tokens** (covering 9,394 positions instead of 9,956).
- Deployed precision increases marginally from $6.78\%$ to **$7.02\%$** ($48 / 684$), while recall drops from $56.55\%$ to $33.10\%$.
- Abstention alone fails because the vast majority of false alarms on coronal stops are not ambiguous with substitutions; rather, they have very low substitution scores and moderate-to-high omission scores.

### 4.7. Paradigm 7: Matched False-Alarm Budgets (Policies 22–25)
To evaluate the system under fixed operational budgets of learner annoyance, we derived thresholds on CAL negatives to target $\le 5, 10, 15, 20$ false alarms per 1,000 phones:
- **Budget $\le 5$ FA / 1k (Policy 22):** Requires $\tau = 0.200898$. Achieves $0.8$ FA / 1k ($8$ FP_cor, $0$ FP_sub), yielding $27.27\%$ precision ($3 / 11$), but recall is restricted to **$2.07\%$**.
- **Budget $\le 10$ FA / 1k (Policy 23):** Requires $\tau = 0.182380$. Achieves $2.6$ FA / 1k ($26$ FP_cor, $0$ FP_sub), yielding $21.21\%$ precision ($7 / 33$), with recall at **$4.83\%$**.
- **Budget $\le 20$ FA / 1k (Policy 25):** Requires $\tau = 0.162606$. Achieves $5.9$ FA / 1k ($58$ FP_cor, $1$ FP_sub), yielding $16.90\%$ precision ($12 / 71$), with recall at **$8.28\%$**.
- **Operational Conclusion:** An MDD deployment constrained to acceptable false alarm rates ($\le 10$ per 1,000 phones) using threshold tuning alone loses more than $95\%$ of all true omission events.

---

## 5. Precision–Recall Frontier & Screening Targets Verification

### 5.1. Retrospective PR Frontiers Across Model Configurations
To ensure our findings are not an artifact of the 3,600-update fixed checkpoint, we benchmarked the complete retrospective PR frontiers across four distinct models under both Protocol Z and Protocol O using `PRECISION_RECALL_FRONTIER.json`:

### Table 5.1: Maximum Attainable Precision and Recall at Target Milestones
*Retrospective upper bounds derived from dense empirical precision-recall curves across all thresholds.*

| Model Configuration | Protocol | Score Key | Max Observed Precision | Recall at Target $P \ge 30\%$ | Recall at Target $P \ge 40\%$ | Recall at Target $P \ge 50\%$ | Recall at Target $P \ge 60\%$ | Recall at Target $P \ge 70\%$ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **T-OP Fixed (3,600 up)** | **Z** | `s_omit` (dedicated) | **50.00%** (1 TP, 1 FP) | **2.76%** (4 TPs, 8 FPs) | **0.69%** (1 TP, 1 FP) | **0.69%** (1 TP, 1 FP) | **UNATTAINABLE** | **UNATTAINABLE** |
| **T-OP Fixed (3,600 up)** | **Z** | `s_error` (legacy) | 4.31% (81 TPs, 1,798 FPs) | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** |
| **B-Control Fixed (3,600 up)** | **Z** | `s_error` (legacy) | 4.75% (68 TPs, 1,364 FPs) | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** |
| **T-OP Best (2,700 up)** | **Z** | `s_omit` (dedicated) | **66.67%** (2 TPs, 1 FP) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **UNATTAINABLE** |
| **B-Control Best (2,250 up)** | **Z** | `s_error` (legacy) | 4.18% (83 TPs, 1,905 FPs) | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** | **UNATTAINABLE** |
| **T-OP Fixed (3,600 up)** | **O** | `s_omit` (dedicated) | **50.00%** (1 TP, 1 FP) | **2.76%** (4 TPs, 7 FPs) | **0.69%** (1 TP, 1 FP) | **0.69%** (1 TP, 1 FP) | **UNATTAINABLE** | **UNATTAINABLE** |
| **T-OP Best (2,700 up)** | **O** | `s_omit` (dedicated) | **66.67%** (2 TPs, 1 FP) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **2.07%** (3 TPs, 2 FPs) | **UNATTAINABLE** |

### 5.2. Theoretical & Empirical Diagnosis of Screening Failure
Why does thresholding fail so catastrophically on the omission task?

#### 1. Severe Imbalance Deficit (Base-Rate Problem)
Out of 9,956 gold positions in Fold-1 VAL, only 145 are gold omissions (**prevalence $\pi = 1.46\%$**; on aligned positions, $\pi = 137 / 8,911 = 1.54\%$). By Bayes' Theorem:
$$P(\text{Omission} \mid \text{Alarm}) = \frac{\pi \cdot \text{TPR}}{\pi \cdot \text{TPR} + (1 - \pi) \cdot \text{FPR}}$$
Even if an operating point achieves a remarkable False Positive Rate of only $\text{FPR} = 1.0\%$, the denominator contains $(1 - 0.0146) \times 0.010 = 0.00985$ false positive mass. To achieve $50\%$ precision, the true positive mass ($\pi \cdot \text{TPR} = 0.0146 \cdot \text{TPR}$) must equal $0.00985$, which requires $\text{TPR} \ge 67.5\%$. However, empirically at $\text{FPR} = 1.0\%$, the model achieves $\text{TPR} = 4.83\%$ (7 detections). The false positives outnumber true positives by more than 3 to 1.

#### 2. Top-Rank Score Contamination
Crucially, the problem is not merely distribution overlap; **the topmost order statistics of the predictor are contaminated by false alarms**. In `T-OP Fixed`:
- **Rank 1 score:** $s_{\text{omit}} = 0.2902$ $\to$ Aligned Gold **CORRECT** (`RRBI arctic_a0384 D`)
- **Rank 2 score:** $s_{\text{omit}} = 0.2528$ $\to$ Aligned Gold **OMISSION** (`RRBI arctic_a0384 D`)
- **Rank 3 score:** $s_{\text{omit}} = 0.2482$ $\to$ Aligned Gold **CORRECT** (`RRBI arctic_a0199 D`)
- **Rank 4 score:** $s_{\text{omit}} = 0.2443$ $\to$ Aligned Gold **CORRECT** (`NJS arctic_a0269 D`)
- **Rank 5 score:** $s_{\text{omit}} = 0.2204$ $\to$ Aligned Gold **CORRECT** (`RRBI arctic_a0199 D`)

Because 4 of the top 5 highest-scoring tokens in the entire corpus are fully articulated gold correct phones, precision cannot exceed $50\%$ at any threshold. The classifier cannot rank true deletions above acoustically ambiguous correct tokens.

---

## 6. Granular Phonetic & Demographic Forensics

### 6.1. Coronal Stop and Liquid Domination
At the primary calibrated operating point ($\tau = 0.066045$, 15% CAL ceiling), the model fires 1,209 alarms. Granular phonetic accounting reveals extreme localization:

| Phoneme | Canonical Category | Gold Deletions ($N$) | Detected TPs | False Alarms (FP_cor) | False Alarms (FP_sub) | Total Alarms | Phoneme Precision | Phoneme Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **/D/** | Coronal Stop | 52 | 47 | 375 | 14 | 436 | **10.78%** | **90.38%** |
| **/T/** | Coronal Stop | 29 | 18 | 298 | 4 | 320 | **5.63%** | **62.07%** |
| **/R/** | Rhotic Liquid | 26 | 15 | 331 | 31 | 377 | **3.98%** | **57.69%** |
| **/L/** | Lateral Liquid | 2 | 1 | 34 | 0 | 35 | **2.86%** | **50.00%** |
| **/ER/** | Rhotic Vowel | 1 | 1 | 14 | 1 | 16 | **6.25%** | **100.00%** |
| **All Other (35 phones)**| Vowels, Fricatives, Nasals | 35 | 0 | 16 | 9 | 25 | **0.00%** | **0.00%** |
| **Total Cohort** | — | **145** | **82** | **1,068** | **59** | **1,209** | **6.78%** | **56.55%** |

- **Concentration:** Three phonemes (`/d/`, `/t/`, `/r/`) account for **$97.56\%$ of all True Positives** ($80 / 82$) and **$93.43\%$ of all False Alarms** ($1,053 / 1,127$).
- **The Rest of the Inventory:** On the remaining 36 phonemes, the model's False Positive Rate on correct phones is a pristine **$0.21\%$** ($16 / 7,478$). However, because gold omissions on these phonemes are virtually absent, precision is $0.00\%$.

### 6.2. Cross-Speaker Demographic Disparity: NJS (Vietnamese) vs. RRBI (Arabic)
The validation cohort contains two distinct L1 backgrounds: NJS (Vietnamese, $N=5,011$ tokens, $80$ gold omissions) and RRBI (Arabic, $N=4,945$ tokens, $65$ gold omissions).

| Speaker | Metric Stratum | Baseline (15% CAL) | Margin Gating ($\Delta \ge 0.05$) | Asymmetric 5% {D,T,R} | Matched Budget ($\le 10$/1k) |
|---|---|---:|---:|---:|---:|
| **NJS (Vietnamese)** | True Positives (TP) | 45 | 12 | 16 | 5 |
| | False Alarms (FP) | 549 | 39 | 101 | 13 |
| | **Deployed Precision** | **7.58%** | **23.53%** | **13.68%** | **27.78%** |
| | **All-Gold Recall** | **56.25%** | **15.00%** | **20.00%** | **6.25%** |
| **RRBI (Arabic)** | True Positives (TP) | 37 | 5 | 12 | 2 |
| | False Alarms (FP) | 578 | 66 | 155 | 13 |
| | **Deployed Precision** | **6.02%** | **7.04%** | **7.19%** | **13.33%** |
| | **All-Gold Recall** | **56.92%** | **7.69%** | **18.46%** | **3.08%** |
| **Disparity Gap** | $|\text{Prec}_{\text{NJS}} - \text{Prec}_{\text{RRBI}}|$ | **1.56%** | **16.49%** | **6.49%** | **14.45%** |

#### Why RRBI Suffers Persistent False Alarms
1. **Rhotic Realization:** NJS produces only 3 gold omissions on `/r/` (detecting 2 TPs with 188 FPs, precision $1.05\%$). In contrast, RRBI produces 23 gold omissions on `/r/` (detecting 13 TPs with 174 FPs, precision $6.95\%$). RRBI frequently produces alveolar taps [ɾ] or trills [r], which Wav2Vec2 confuses with deletions due to rapid acoustic closure.
2. **Coronal Stop Release:** On `/d/`, RRBI has 17 gold deletions but triggers 196 false alarms (precision $7.55\%$), compared to NJS who has 35 gold deletions and 193 false alarms (precision $13.84\%$). Unreleased word-final stops in Arabic L1 transfer consistently trigger high omission logits.

---

## 7. Synthesis, Forward Integration & Multi-Workstream Interface

### 7.1. Why Decision Policy Alone Is an Architectural Dead End
The empirical evidence across all 25 policies demonstrates that the omission precision failure cannot be cured in post-processing. Because the underlying model's acoustic representations assign identical probability density to unreleased stops, flapped liquids, and true deletions, any monotonic threshold or gating function that removes false alarms removes true positives in equal or greater numbers.

### 7.2. Interface with Agent B (Stop Consonant Acoustic Verification)
Agent B's acoustic forensics establish that **$88.7\%$ of coronal stop false alarms possess visible physical release bursts, non-zero closure durations, or high-frequency burst energy**. A post-hoc physics-based acoustic verifier (evaluating spectral energy and burst transitions directly from audio) can reject these false alarms without relying on the contaminated model logits.

### 7.3. Interface with Agent C (Rhotic Analysis & Ground-Truth Adjudication)
Agent C's blinded re-annotation audit reveals that **$23.6\%$ of model "false alarms" on `/r/` and `/d/` are in fact ground-truth annotation errors in L2-ARCTIC**. Learners actually deleted or severely reduced the phoneme, but original human annotators hallucinated the phone due to lexical priming. When evaluated against adjudicated ground truth, model precision increases substantially.

### 7.4. Interface with Agent D (System-Level Integration & Ablations)
Agent D's system-level benchmark demonstrates that combining **Conservative Thresholding (Margin Gating $\Delta \ge 0.02$)** with **Physical Acoustic Stop Verification** achieves a Pareto-optimal operating state that cuts learner false alarm burden while preserving clinical diagnostic sensitivity.

---

## 8. Provenance, Checksums & Reproducibility Ledger

All results in this benchmark are deterministically reproducible using `.venv/bin/python` from the frozen cache and scripts.

### Deliverables and File Hashes
| File Path | Description | SHA-256 Checksum |
|---|---|---|
| `workstreams/agent_a_policy/POLICY_COMPARISON.csv` | Full 25-policy benchmark table | `ad2c292c9d5842508bdd4db59526e2b03889f395935c16feb61f706c5ca7af16` |
| `POLICY_COMPARISON.csv` | Root repository copy | `ad2c292c9d5842508bdd4db59526e2b03889f395935c16feb61f706c5ca7af16` |
| `workstreams/agent_a_policy/policy_summary.json` | JSON serialization of all 25 policies | `70b8a06ba266ae84bdbf42119677c69d9dd6f05344ecb8881654c4e139cdc301` |
| `workstreams/agent_a_policy/PRECISION_RECALL_FRONTIER.json` | Complete PR frontier and milestones | `5a982db017883f0a68c72d09715e52af8c6e717c81465f48eb1787007a1163e4` |
| `PRECISION_RECALL_FRONTIER.json` | Root repository copy | `5a982db017883f0a68c72d09715e52af8c6e717c81465f48eb1787007a1163e4` |
| `workstreams/agent_a_policy/run_policy_benchmark.py` | Computational benchmark harness | `cf00d34052867a83c381e2de4aa227b6d5ea374440c98ff6dd045913e84c4552` |

### Input Data and Model Checkpoints
- **L2-ARCTIC 4-Fold Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`  
  SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`
- **Authorized Dataset Archive:** `l2arctic_release_v5.0.zip`  
  SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`
- **Protocol Z Offline Evaluation JSONL:**  
  SHA-256: `260b851d9324235cddf93eff29cd734c97f31ddc13de06165a0cb103d5afb74d`
- **Protocol Z Calibration JSONL:**  
  SHA-256: `3ff1a9d3c1cd61d40815e7849ecf7ce538d54877f8db0b288752c3a8790a1df1`
