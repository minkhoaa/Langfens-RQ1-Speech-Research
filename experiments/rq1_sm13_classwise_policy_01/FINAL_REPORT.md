# LANGFENS — SM13-CLASSWISE-POLICY-01: Final Evaluation Report

**Experiment ID:** `SM13-CLASSWISE-POLICY-01`  
**Target Repository:** `/home/khoa/Projects/New-Method-Research`  
**Execution Timestamp:** 2026-11-11  
**Author:** Senior Speech Machine Learning Research Scientist & Systems Auditor  
**Primary Predetermined Seed:** 101 (cross-verified on seeds 202, 303)  
**Status:** **COMPLETE & INDEPENDENTLY REPRODUCIBLE**

---

## 1. Executive Decision

### Formal Decision: **NO_GAIN**

**Technical Justification:**
1. **Severe Class Trade-Off on TRAIN OOF:** In out-of-fold cross-validation on TRAIN (12 speakers, 1,800 utterances), shifting false warning budget from Omission to Substitution incrementally increases Recall-S from ~3.3% to ~5.7% (Seed 101). However, this modest gain forces Omission recall to drop sharply from **16.38% down to 9.81%** (a 40% relative loss), causing Macro-3 F1 to fall from **0.1679 to 0.1526** under the matched joint FWR constraint ($\le 0.20$).
2. **Zero Gain at Matched Realized FWR:** At strictly matched realized FWR = 0.2000, the best classwise policy achieves Macro-3 F1 = 0.1716 versus 0.1693 for the shared-target policy (+0.0023 gain), but Recall-S is actually *lower* (3.25% vs 3.41%) because both classes are constrained to extremely conservative operating thresholds.
3. **Total Failure on DEV_CAL Diagnostic Transfer:** When candidate classwise policies frozen from TRAIN OOF ($\tau_S \approx 0.7570 - 0.8147$) are transferred to the fixed DEV_CAL benchmark (Speakers `ERMS`, `THV`), **Recall-S remains exactly 0.00%**. Root-cause investigation reveals that the CTC/E4 sequence likelihood ratio component of the 50/50 fixed fusion suffers from severe score compression on unseen speakers, capping fixed substitution probabilities on DEV_CAL at $p_S \le 0.6699$ (max on gold substitutions = $0.6538$). Consequently, any policy calibrated on TRAIN OOF with joint FWR $\le 0.20$ mathematically yields zero substitution detections on DEV_CAL.
4. **Generalization Firewall Integrity:** In-sample tuning on DEV_CAL to force Recall-S > 0 would require lowering $\tau_S$ below 0.50, which when applied to TRAIN OOF produces an catastrophic flood of 1,467 false alarms (FWR = 0.815, a 4x budget violation). Tuning on DEV_CAL is strictly barred by governance protocols.

Therefore, class-specific thresholding does **not** provide a viable mechanism to recover Substitution recall while preserving Omission, Insertion, and joint FWR $\le 0.20$.

---

## 2. Research Questions & Baseline Setup

### 2.1 The Problem
In the frozen DEV_CAL operating policy (`LANGFENS-SM13-FUSION-V1`, Seed 101):
- Operating thresholds: $\tau_T = 0.848521, \tau_G = 0.873637$
- Overall metrics: Macro-3 F1 = 0.169221; Joint FWR = 0.176667 false warnings / utterance (53 false alarms across 300 utterances)
- Classwise recall: **Recall-S = 0.00%**, Recall-O = 19.26%, Recall-I = 12.75%

The objective was to determine whether decomposing the shared target threshold $\tau_T$ into independent class thresholds $\tau_S$ and $\tau_O$, coupled with deterministic arbitration, could recover Substitution recall without regressing Omission, Insertion, or exceeding joint FWR $\le 0.20$.

### 2.2 Population & Data Partitions
- **TRAIN OOF:** 12 speakers (`ABA`, `ASI`, `BWC`, `EBVS`, `HJK`, `HKK`, `HQTV`, `LXC`, `NJS`, `PNV`, `SKA`, `SVBI`), 1,800 utterances, 59,562 target events (Correct: 50,356; Substitution: 7,381; Omission: 1,825), 61,362 gap events (No-Insertion: 60,802; Insertion: 560).
- **DEV_CAL (Diagnostic):** 2 speakers (`ERMS`, `THV`), 300 utterances, 10,043 target events (Correct: 7,800; Substitution: 1,781; Omission: 462), 10,343 gap events (No-Insertion: 10,241; Insertion: 102).
- **TEST_SEALED:** Completely unaccessed and cryptographically verified unchanged (`a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`).

---

## 3. Mathematical Formulation of Classwise Policy

### 3.1 Threshold Application & Conflict Resolution
For any target event $i$ with fused probabilities $p_i = [p_{i,C}, p_{i,S}, p_{i,O}]$ where $\sum_c p_{i,c} = 1.0$:
- Target alert flags:
  $$w_{i,S} = \mathbf{1}(p_{i,S} > \tau_S) \lor (p_{i,S} = \tau_S \land \text{tie\_hash}(i) \ge \text{cutoff}_S)$$
  $$w_{i,O} = \mathbf{1}(p_{i,O} > \tau_O) \lor (p_{i,O} = \tau_O \land \text{tie\_hash}(i) \ge \text{cutoff}_O)$$
- Decision rule:
  $$\hat{y}_{i,t} = \begin{cases}
  0 \text{ (CORRECT)}, & \text{if } \neg w_{i,S} \land \neg w_{i,O} \\
  1 \text{ (SUBSTITUTION)}, & \text{if } w_{i,S} \land \neg w_{i,O} \\
  2 \text{ (OMISSION)}, & \text{if } \neg w_{i,S} \land w_{i,O} \\
  \text{Arbitrate}(p_{i,S}, p_{i,O}, \tau_S, \tau_O, i), & \text{if } w_{i,S} \land w_{i,O}
  \end{cases}$$

### 3.2 Arbitration Rules for S/O Conflict
Five deterministic arbitration rules were implemented and audited:
1. **Score Margin (`score_margin`):** $\text{argmax}(p_S - \tau_S, p_O - \tau_O)$
2. **Normalized Margin (`norm_margin`):** $\text{argmax}\left(\frac{p_S - \tau_S}{1 - \tau_S}, \frac{p_O - \tau_O}{1 - \tau_O}\right)$
3. **Probability Comparison (`prob_comparison`):** $\text{argmax}(p_S, p_O)$
4. **Substitution Priority (`priority_substitution`):** Defaults to Substitution
5. **Omission Priority (`priority_omission`):** Defaults to Omission

*Empirical Finding:* Because $p_S + p_O \le 1.0$, whenever $\tau_S + \tau_O > 1.0$, conflict is mathematically impossible. In all nominal policies where $\tau_S \ge 0.75$ and $\tau_O \ge 0.79$, the empirical conflict count is **exactly zero** across all 59,562 events in TRAIN and all 10,043 events in DEV_CAL. All five arbitration rules yield bit-for-bit identical results in the operational regime.

### 3.3 False Warning Budget Accounting
Per EVAL-V3.1 standard, every predicted error differing from the gold label is counted as a false warning:
$$\text{false\_warnings} = \text{FP}_S + \text{FP}_O + \text{FP}_I$$
$$\text{FP}_S = (\text{C}\to\text{S}) + (\text{O}\to\text{S})$$
$$\text{FP}_O = (\text{C}\to\text{O}) + (\text{S}\to\text{O})$$
$$\text{FP}_I = (\text{NoIns}\to\text{Ins})$$
$$\text{wrong\_type\_warnings} = (\text{O}\to\text{S}) + (\text{S}\to\text{O})$$

---

## 4. Empirical Evaluation: TRAIN OOF (Seeds 101, 202, 303)

### 4.1 Bounded Classwise Budget Allocation Sweep (Seed 101, Pooled 4 Folds)

Evaluated under nominal joint false warning budget $B = 0.20$ false warnings / utterance ($N = 1,800$ utts, max nominal false warnings = 360). Gap budget fixed at $B_I = 0.040$.

| Policy Specification | $B_S$ | $B_O$ | $\tau_S$ | $\tau_O$ | $\tau_I$ | Joint FWR | FW Count | Wrong Type | Macro-3 F1 | Recall-S | F1-S | Recall-O | F1-O | Recall-I | F1-I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Shared Target Baseline** | — | — | 0.8178 | 0.8178 | 0.7380 | **0.2017** | 363 | 52 | **0.1679** | 3.36% | 0.0643 | 16.38% | 0.2569 | 11.43% | 0.1826 |
| Split 0.02 S / 0.14 O | 0.02 | 0.14 | 0.8669 | 0.7916 | 0.7380 | 0.2028 | 365 | 58 | 0.1661 | 1.83% | 0.0358 | 18.52% | 0.2799 | 11.43% | 0.1826 |
| Split 0.04 S / 0.12 O | 0.04 | 0.12 | 0.8229 | 0.8099 | 0.7380 | 0.2022 | 364 | 51 | **0.1694** | 3.27% | 0.0626 | 16.93% | 0.2631 | 11.43% | 0.1826 |
| Split 0.05 S / 0.11 O (Cand.) | 0.05 | 0.11 | 0.8147 | 0.8217 | 0.7380 | 0.2028 | 365 | 51 | 0.1679 | 3.59% | 0.0685 | 16.00% | 0.2526 | 11.43% | 0.1826 |
| Split 0.06 S / 0.10 O | 0.06 | 0.10 | 0.8022 | 0.8333 | 0.7380 | 0.2028 | 365 | 49 | 0.1645 | 4.04% | 0.0765 | 14.58% | 0.2344 | 11.43% | 0.1826 |
| Split 0.08 S / 0.08 O | 0.08 | 0.08 | 0.7769 | 0.8575 | 0.7380 | 0.2028 | 365 | 40 | 0.1587 | 4.81% | 0.0901 | 12.22% | 0.2035 | 11.43% | 0.1826 |
| Split 0.10 S / 0.06 O | 0.10 | 0.06 | 0.7570 | 0.8826 | 0.7380 | 0.2106 | 379 | 32 | 0.1526 | 5.74% | 0.1061 | 9.81% | 0.1689 | 11.43% | 0.1826 |
| Split 0.12 S / 0.04 O | 0.12 | 0.04 | 0.7397 | 0.9100 | 0.7380 | 0.2028 | 365 | 23 | 0.1431 | 6.58% | 0.1202 | 7.01% | 0.1265 | 11.43% | 0.1826 |

*Key Findings from Table:*
1. **The Inherent Trade-Off:** Increasing $B_S$ increases Recall-S from 1.83% to 6.58% and reduces wrong-type warnings from 58 to 23.
2. **The Omission Collapse:** Every unit of recall gained in Substitution costs ~1.5 to 2.0 units of Omission recall. As $B_S$ increases to 0.10, Omission recall collapses by nearly half (from 16.38% to 9.81%), and Macro-3 F1 drops from 0.1679 to 0.1526.
3. **Macro-3 F1 Invariance:** At the peak split (0.04 S / 0.12 O), Macro-3 F1 is 0.1694 vs 0.1679 (+0.0015), well within random variance, with no meaningful gain in Recall-S (3.27% vs 3.36%).

---

### 4.2 Cross-Seed Multi-Fold Verification (Seeds 101, 202, 303)

Comparing the baseline shared-target policy against the primary candidate classwise split (0.05 S / 0.11 O / 0.04 I):

| Seed | Policy Type | Joint FWR | FW Count | Wrong Type | Macro-3 F1 | Recall-S | F1-S | Recall-O | F1-O | Recall-I | F1-I |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Seed 101** | Shared Target | 0.2017 | 363 | 52 | **0.1679** | 3.36% | 0.0643 | 16.38% | 0.2569 | 11.43% | 0.1826 |
| Seed 101 | Classwise (0.05/0.11) | 0.2028 | 365 | 51 | **0.1679** | 3.59% | 0.0685 | 16.00% | 0.2526 | 11.43% | 0.1826 |
| **Seed 202** | Shared Target | 0.2050 | 369 | 54 | 0.1663 | 2.09% | 0.0406 | 16.55% | 0.2565 | 12.86% | 0.2020 |
| Seed 202 | Classwise (0.05/0.11) | 0.2100 | 378 | 49 | **0.1673** | 3.36% | 0.0642 | 14.85% | 0.2359 | 12.86% | 0.2020 |
| **Seed 303** | Shared Target | 0.2033 | 366 | 47 | **0.1828** | 3.82% | 0.0728 | 16.66% | 0.2613 | 13.75% | 0.2145 |
| Seed 303 | Classwise (0.05/0.11) | 0.2072 | 373 | 48 | 0.1825 | 3.90% | 0.0742 | 16.49% | 0.2589 | 13.75% | 0.2145 |
| **3-Seed Mean** | **Shared Target** | **0.2033** | **366.0** | **51.0** | **0.1723** | **3.09%** | **0.0592** | **16.53%** | **0.2582** | **12.68%** | **0.1997** |
| **3-Seed Mean** | **Classwise (0.05/0.11)** | **0.2067** | **372.0** | **49.3** | **0.1726** | **3.62%** | **0.0690** | **15.78%** | **0.2491** | **12.68%** | **0.1997** |
| **Delta ($\Delta$)** | | **+0.0034** | **+6.0** | **-1.7** | **+0.0003** | **+0.53%** | **+0.0098** | **-0.75%** | **-0.0091** | **0.00%** | **0.0000** |

*Summary:* The 3-seed pooled Macro-3 F1 delta is +0.0003 (+0.17% relative), completely within seed variance, while joint FWR rises slightly (+0.0034). There is no statistically or practically meaningful gain.

---

### 4.3 Matched Realized FWR Evaluation (Seed 101)

To eliminate any nominal budget slack or calibration drift, policies were evaluated across strictly matched realized false warning counts:

| Target False Warnings | Realized FWR | Shared Macro-3 F1 | Classwise Macro-3 F1 | Shared Recall-S | Classwise Recall-S | Shared Recall-O | Classwise Recall-O |
|---|---|---|---|---|---|---|---|
| 180 | 0.1000 | 0.1154 | 0.1155 (+0.0001) | 1.53% | 1.83% (+0.30%) | 10.25% | 9.86% (-0.39%) |
| 270 | 0.1500 | 0.1411 | 0.1414 (+0.0003) | 2.47% | 2.55% (+0.08%) | 13.42% | 13.37% (-0.05%) |
| 360 | 0.2000 | 0.1693 | 0.1716 (+0.0023) | 3.41% | 3.25% (-0.16%) | 16.33% | 17.15% (+0.82%) |
| 424 | 0.2356 | 0.1811 | 0.1820 (+0.0009) | 3.96% | 3.55% (-0.41%) | 17.75% | 18.74% (+0.99%) |
| 540 | 0.3000 | 0.2011 | 0.1988 (-0.0023) | 4.95% | 3.89% (-1.06%) | 20.16% | 21.48% (+1.32%) |

*Conclusion:* When false-warning rates are strictly matched, classwise thresholding fails to provide a systematic advantage over the shared-target policy at any operating point between FWR 0.10 and 0.30.

---

## 5. Diagnostic Comparison on DEV_CAL (Fixed Policy Transfer)

The prompt requires:
> *"Use DEV_CAL only for a clearly labeled, fixed-policy diagnostic comparison after freezing the candidate from TRAIN OOF. Do not retune using DEV_CAL results."*

### 5.1 Root Cause Analysis of DEV_CAL Score Distribution
Why did DEV_CAL exhibit 0% Substitution recall under the frozen policy ($\tau_T = 0.848521$)?

Inspection of raw model components on DEV_CAL reveals an acute score compression phenomenon:
- **Neural Head Output on DEV_CAL:** $p_{\text{neural}, S} \in [0.0000, 0.9957]$ (Neural head confidence is healthy and reaches 0.9957 on unseen speakers).
- **CTC/E4 Head Output on DEV_CAL:** $p_{\text{E4}, S} \in [0.0000, \mathbf{0.6508}]$ (E4 sequence likelihood ratio simplex is compressed into $[0, 0.65]$).
- **Fixed 50/50 Fusion Output on DEV_CAL:**
  $$p_{\text{fixed}, S} = 0.5 \cdot p_{\text{neural}, S} + 0.5 \cdot p_{\text{E4}, S}$$
  Because $p_{\text{E4}, S}$ never exceeds 0.6508, the fused substitution probability $p_{\text{fixed}, S}$ is strictly compressed into $[0.0000, \mathbf{0.6699}]$.
  On genuine gold substitutions ($y_t = 1$), the **maximum fused score is 0.653875**.
- In contrast, on TRAIN OOF, the CTC model was evaluated on in-domain speaker acoustic context where $p_{\text{E4}, S}$ reached 0.9999, so $p_{\text{fixed}, S}$ reached **0.9920**.

### 5.2 Diagnostic Transfer Results (DEV_CAL, $N=300$ utterances, 10,043 targets)

All candidate policies were frozen strictly from Seed 101 TRAIN OOF and evaluated without modification on DEV_CAL:

| Policy Configuration | Provenance | $\tau_S$ | $\tau_O$ | $\tau_I$ | Realized FWR | FW Count | Wrong Type | Macro-3 F1 | Recall-S | F1-S | Recall-O | F1-O | Recall-I | F1-I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Frozen DEV_CAL Baseline** | Frozen DevCal Policy | 0.8485 | 0.8485 | 0.8736 | **0.1767** | 53 | 7 | **0.1692** | **0.00%** | **0.0000** | **19.26%** | **0.2997** | 12.75% | 0.2080 |
| Cand. 1: Split 0.04 S / 0.12 O | Frozen TRAIN OOF | 0.8229 | 0.8099 | 0.8424 | 0.2400 | 72 | 10 | 0.2050 | **0.00%** | **0.0000** | 25.54% | 0.3688 | 15.69% | 0.2462 |
| Cand. 2: Split 0.05 S / 0.11 O | Frozen TRAIN OOF | 0.8147 | 0.8217 | 0.8424 | 0.2267 | 68 | 10 | 0.1997 | **0.00%** | **0.0000** | 24.03% | 0.3529 | 15.69% | 0.2462 |
| Cand. 3: Split 0.06 S / 0.10 O | Frozen TRAIN OOF | 0.8022 | 0.8333 | 0.8424 | 0.2033 | 61 | 9 | 0.1902 | **0.00%** | **0.0000** | 21.43% | 0.3246 | 15.69% | 0.2462 |
| Cand. 4: Split 0.08 S / 0.08 O | Frozen TRAIN OOF | 0.7769 | 0.8575 | 0.8424 | 0.1633 | 49 | 6 | 0.1810 | **0.00%** | **0.0000** | 18.83% | 0.2969 | 15.69% | 0.2462 |
| Cand. 5: B=0.18 Margined | Frozen TRAIN OOF | 0.8208 | 0.8267 | 0.8540 | 0.2033 | 61 | 9 | 0.1777 | **0.00%** | **0.0000** | 21.65% | 0.3268 | 12.75% | 0.2063 |

### 5.3 Diagnostic Transfer Takeaway
Every candidate classwise policy frozen from TRAIN OOF with joint FWR $\le 0.20$ requires $\tau_S \ge 0.7570$. Because the upper bound of $p_{\text{fixed}, S}$ on DEV_CAL is 0.6538, **not a single substitution alert can ever be triggered** on DEV_CAL. Meanwhile:
- Candidate policies with lower $\tau_O$ and $\tau_I$ exceed the $0.20$ FWR budget on DEV_CAL (realized FWR = $0.2267 - 0.2400$).
- Candidate policies that satisfy FWR $\le 0.20$ on DEV_CAL (Cand. 4) regress Omission recall below the baseline (18.83% vs 19.26%).

---

## 6. Speaker-Level Performance Breakdown

Evaluation across all 12 TRAIN speakers (150 utterances each) under Seed 101:

| Speaker | Acc | Targets | Gaps | AP-S | AP-O | AP-I | Shared FWR | Shared Sub R | Shared Omi R | Classwise FWR | Classwise Sub R | Classwise Omi R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ABA` | Arabic | 4,960 | 5,110 | 0.4491 | 0.3694 | 0.1802 | 0.2667 | 3.32% | 19.82% | 0.2800 | 3.99% | 19.37% |
| `ASI` | Arabic | 4,834 | 4,984 | 0.3721 | 0.4278 | 0.1610 | 0.2667 | 2.50% | 18.66% | 0.2600 | 2.80% | 17.54% |
| `BWC` | Chinese | 5,054 | 5,204 | 0.4507 | 0.3470 | 0.1820 | 0.4733 | 2.81% | 19.54% | 0.4800 | 3.01% | 19.16% |
| `EBVS` | Spanish | 4,942 | 5,092 | 0.3951 | 0.2600 | 0.1666 | 0.2267 | 2.62% | 11.26% | 0.2200 | 3.12% | 10.38% |
| `HJK` | Korean | 4,930 | 5,080 | 0.3667 | 0.2311 | 0.2223 | 0.1600 | 1.83% | 7.97% | 0.1600 | 2.44% | 7.97% |
| `HKK` | Korean | 4,976 | 5,126 | 0.3601 | 0.1883 | 0.1685 | 0.2467 | 2.41% | 8.21% | 0.2533 | 2.41% | 8.21% |
| `HQTV` | Viet | 5,049 | 5,199 | 0.4632 | 0.4912 | 0.1558 | 0.1867 | 3.73% | 17.70% | 0.1867 | 3.86% | 17.70% |
| `LXC` | Chinese | 5,124 | 5,274 | 0.4907 | 0.5186 | 0.1477 | 0.1933 | 3.59% | 18.52% | 0.1867 | 3.99% | 17.90% |
| `NJS` | Spanish | 4,904 | 5,054 | 0.4842 | 0.4658 | 0.1488 | 0.2000 | 5.08% | 17.37% | 0.2067 | 5.46% | 17.01% |
| `PNV` | Viet | 4,874 | 5,024 | 0.4354 | 0.3703 | 0.2505 | 0.2067 | 3.32% | 15.52% | 0.2067 | 3.90% | 14.54% |
| `SKA` | Arabic | 5,038 | 5,188 | 0.3846 | 0.3957 | 0.2081 | 0.2000 | 3.65% | 14.93% | 0.2067 | 3.94% | 14.93% |
| `SVBI` | Hindi | 4,877 | 5,027 | 0.4243 | 0.3475 | 0.1931 | 0.1867 | 3.03% | 14.49% | 0.1933 | 3.33% | 14.49% |
| **DEV: ERMS** | Spanish | 5,129 | 5,279 | 0.3697 | 0.4285 | 0.1706 | 0.2267 | 0.00% | 20.30% | 0.2667 | 0.00% | 23.31% |
| **DEV: THV** | Viet | 4,914 | 5,064 | 0.3712 | 0.4278 | 0.1751 | 0.1267 | 0.00% | 18.23% | 0.1867 | 0.00% | 24.74% |

---

## 7. Audit Deliverables Inventory

The complete set of verified deliverables is saved under `experiments/rq1_sm13_classwise_policy_01/`:

1. `FINAL_REPORT.md`: This comprehensive research report and audit summary.
2. `POLICY_COMPARISON.csv`: 336 rows documenting every evaluated policy, budget split, arbitration rule, realized FWR, classwise precision/recall/F1, and event counts across TRAIN OOF folds and DEV_CAL.
3. `CLASSWISE_METRICS.csv`: 76 rows containing granular per-speaker breakdowns for all 12 TRAIN speakers and 2 DEV_CAL speakers.
4. `CANDIDATE_CLASSWISE_POLICY.json`: Complete frozen machine-readable candidate policy artifact, including threshold values, tie cutoffs, budget allocations, and provenance firewall confirmations.
5. `REGRESSION_RESULTS.md`: Detailed test suite execution summary certifying 6/6 test passes, mathematical equivalence proofs, and integrity audits.
6. `run_experiments.py`: Authoritative deterministic Python execution engine.
7. `tests/test_sm13_classwise_policy_regression.py`: Dedicated Pytest regression suite.

### Cryptographic Checksums
```
dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1  experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json (Preserved)
a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6  data/derived/l2arctic/manifests/test_sealed.jsonl (Sealed & Untouched)
```

---

## 8. Final Research Recommendation

Operating policy thresholding is an **arbitration mechanism over existing posterior margins**, not a feature representation fix. It cannot synthesize high confidence where the underlying acoustic-phonetic representation provides none.

The fundamental reason for 0% Substitution recall on DEV_CAL is not an overly restrictive shared threshold, but rather the **failure of the CTC sequence likelihood ratio (E4) to scale properly on out-of-domain speakers**, severely compressing the fixed fusion simplex. To recover Substitution detection without regressing Omission or Insertion, future research must address acoustic representation alignment and likelihood ratio calibration prior to fusion, rather than attempting post-hoc classwise threshold tuning.

**Status: COMPLETE. Decision: NO_GAIN.**
