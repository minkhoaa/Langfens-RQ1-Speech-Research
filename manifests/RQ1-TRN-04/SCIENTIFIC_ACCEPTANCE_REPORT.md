# LANGFENS RQ1-TRN-04 — Scientific Acceptance & Gate Audit Report

**Document ID:** `LANGFENS-RQ1-TRN-04-ACCEPTANCE-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist & Auditor  
**Date:** 2026-11-04  
**Status:** **EMPIRICALLY RESOLVED — ALL CANDIDATES FAILED ACCEPTANCE GATES**  
**Evaluation Standard:** Evaluation Contract V1 Protocol Z (Fold-1 VAL, 300 utterances)  

---

## 1. Executive Scientific Verdict

Under the prospectively registered acceptance criteria frozen at Gate 0, **ALL THREE EXPERIMENTAL TREATMENTS ($T_1, T_2, T_3$) AS WELL AS THE CONTROL ($C_0$) FAILED BOTH LEVEL A (PILOT FEASIBILITY) AND LEVEL B (RESEARCH IMPROVEMENT TARGET) ACCEPTANCE GATES**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ACCEPTANCE GATE AUDIT SUMMARY TABLE                          │
├─────┬──────────────────────┬────────────────────────┬────────────────────────┬─────────────────┤
│ Arm │ Experimental Arm     │ Level A (Feasibility)  │ Level B (Target)       │ Overall Verdict │
├─────┼──────────────────────┼────────────────────────┼────────────────────────┼─────────────────┤
│ C0  │ Matched Control (CE) │ FAILED (Prec 7.55%)    │ FAILED (Prec 7.55%)    │ REJECTED        │
│ T1  │ Objective (Focal)    │ FAILED (Prec 1.46%)    │ FAILED (Prec 1.46%)    │ REJECTED        │
│ T2  │ Acoustic (Waveform)  │ FAILED (Prec 7.09%)    │ FAILED (Prec 7.09%)    │ REJECTED        │
│ T3  │ Combined (Focal+Feat)│ FAILED (Prec 1.46%)    │ FAILED (Prec 1.46%)    │ REJECTED        │
└─────┴──────────────────────┴────────────────────────┴────────────────────────┴─────────────────┘
```

---

## 2. Granular Evaluation against Prospective Level A Gates

Level A defines the minimal practical screening threshold for pedagogical deployment. Passing requires satisfying **all 6 criteria simultaneously**:

| Level A Criterion Channel | Registered Threshold | Arm C0 | Arm T1 | Arm T2 | Arm T3 | Historical T-OP |
|---|---:|---:|---:|---:|---:|---:|
| **1. Strict Deployed Precision** | $\ge 25.00\%$ | 7.55% (FAIL) | 1.46% (FAIL) | 7.09% (FAIL) | 1.46% (FAIL) | 6.78% (FAIL) |
| **2. All-Gold Omission Recall** | $\ge 30.00\%$ | 60.00% (PASS) | 100.00% (PASS)| 51.03% (PASS) | 100.00% (PASS)| 56.55% (PASS) |
| **3. False Alarms / 1k Phones** | $\le 30.00$ | 107.07 (FAIL) | 985.44 (FAIL) | 97.33 (FAIL) | 985.44 (FAIL) | 113.20 (FAIL) |
| **4. Binary AP Non-Inferiority**| $\ge 0.313619$ | 0.321619 (PASS)| 0.315171 (PASS)| 0.321573 (PASS)| 0.315847 (PASS)| 0.318619 (PASS)|
| **5. Substitution AP Non-Inf.** | $\ge 0.303974$ | 0.310022 (PASS)| 0.311767 (PASS)| 0.312209 (PASS)| 0.312187 (PASS)| 0.308974 (PASS)|
| **6. Greedy CTC PER Non-Inf.**  | $\le 0.246980$ | 0.238943 (PASS)| 0.239045 (PASS)| 0.239045 (PASS)| 0.239045 (PASS)| 0.241980 (PASS)|
| **LEVEL A STATUS** | **All 6 Must Pass** | **FAILED** | **FAILED** | **FAILED** | **FAILED** | **FAILED** |

### Level A Analysis:
- **Upstream Channels Passed Comfortably:** All four models easily satisfied non-inferiority on Binary MDD AP ($\ge 0.3136$), Substitution AP ($\ge 0.3039$), and greedy CTC PER ($\le 0.2469$). The backbone representations and acoustic-phonetic alignment remained stable and healthy.
- **Strict Omission Precision Failed Disastrously:** No model approached the required $25.0\%$ precision threshold. The best model ($C_0$) achieved only $7.55\%$, while focal supervision ($T_1, T_3$) collapsed to $1.46\%$.
- **False Alarm Budget Breached:** False alarms far exceeded the 30.0 / 1k threshold ($97.3 - 985.4$ / 1k phones).

---

## 3. Granular Evaluation against Prospective Level B Gates

Level B defines the research target for claiming scientific advancement and superiority:

| Level B Criterion Channel | Registered Target | Arm C0 | Arm T1 | Arm T2 | Arm T3 |
|---|---:|---:|---:|---:|---:|
| **1. Strict Deployed Precision** | $\ge 30.00\%$ | 7.55% (FAIL) | 1.46% (FAIL) | 7.09% (FAIL) | 1.46% (FAIL) |
| **2. All-Gold Omission Recall** | $\ge 40.00\%$ | 60.00% (PASS) | 100.00% (PASS)| 51.03% (PASS) | 100.00% (PASS)|
| **3. False Alarms / 1k Phones** | $\le 15.00$ | 107.07 (FAIL) | 985.44 (FAIL) | 97.33 (FAIL) | 985.44 (FAIL) |
| **4. Binary MDD Non-Inferiority**| $\ge 0.313619$ | 0.321619 (PASS)| 0.315171 (PASS)| 0.321573 (PASS)| 0.315847 (PASS)|
| **5. Substitution Non-Inferiority**| $\ge 0.303974$ | 0.310022 (PASS)| 0.311767 (PASS)| 0.312209 (PASS)| 0.312187 (PASS)|
| **6. Greedy PER Non-Inferiority** | $\le 0.246980$ | 0.238943 (PASS)| 0.239045 (PASS)| 0.239045 (PASS)| 0.239045 (PASS)|
| **7. Acoustic Reliance Gap**     | $\ge +0.01000$ | +0.0199 (PASS) | +0.0141 (PASS) | +0.0225 (PASS) | +0.0172 (PASS) |
| **8. Inter-Speaker Robustness**  | Gap $\le 0.150$ | 0.1444 (PASS) | 0.1531 (FAIL) | 0.1546 (FAIL) | 0.1522 (FAIL) |
| **LEVEL B STATUS** | **All 8 Must Pass** | **FAILED** | **FAILED** | **FAILED** | **FAILED** |

---

## 4. Evaluation against Matched Internal Control C0

To eliminate any potential confounding from differences between historical P-SUP and the common factorial architecture, we evaluate non-inferiority directly against **newly trained matched control $C_0$** using margins of $-0.005$ on AP and $+0.005$ on PER:

| Contrast | Deployed Prec Delta | Omission AP Delta | Binary AP Delta | PER Delta | Non-Inferiority Verdict |
|---|---:|---:|---:|---:|---|
| **T1 vs C0 (Objective)** | -6.09 pp | -0.085759 | -0.006448 (exceeds -0.005) | +0.000102 | **INFERIOR (Severe Regression)** |
| **T2 vs C0 (Acoustic)**  | -0.46 pp | -0.017038 | -0.000046 (within margin) | +0.000102 | **INFERIOR on Omission AP** |
| **T3 vs C0 (Combined)**  | -6.09 pp | -0.085720 | -0.005772 (exceeds -0.005) | +0.000102 | **INFERIOR (Severe Regression)** |

### Scientific Verdict on Architecture Advancement:
- Neither $T_1$, $T_2$, nor $T_3$ demonstrated non-inferiority or superiority over $C_0$ on the primary omission metrics.
- Neither representation intervention justifies promotion to independent validation or test partition evaluation.
