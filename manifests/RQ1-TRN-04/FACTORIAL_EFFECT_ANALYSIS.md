# LANGFENS RQ1-TRN-04 — 2×2 Factorial Effect Analysis & Scientific Hypothesis Testing

**Document ID:** `LANGFENS-RQ1-TRN-04-FACTORIAL-EFFECT-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist & Reproducibility Auditor  
**Date:** 2026-11-04  
**Design:** Balanced 2×2 Factorial (Factor A: Objective, Factor B: Acoustic Features)  
**Endpoint:** Fixed-Budget 3,600 Updates on Fold-1 VAL (Protocol Z)  

---

## 1. Primary Factorial Effects Decomposition

The factorial design allows unambiguous mathematical decomposition of the **Main Effect of Supervision Objective ($\text{Factor } A$)**, the **Main Effect of Waveform Acoustic Features ($\text{Factor } B$)**, and their **Factorial Interaction ($A \times B$)**:

$$\text{Main Effect of Objective } (A) = \frac{(T_1 - C_0) + (T_3 - T_2)}{2}$$
$$\text{Main Effect of Acoustic Features } (B) = \frac{(T_2 - C_0) + (T_3 - T_1)}{2}$$
$$\text{Interaction } (A \times B) = T_3 - T_2 - T_1 + C_0$$

### Complete Factorial Decomposition Table

| Metric Channel | Arm C0 (Control) | Arm T1 (Objective) | Arm T2 (Acoustic) | Arm T3 (Combined) | Main Effect: Objective ($A$) | Main Effect: Acoustic ($B$) | Factorial Interaction ($A \times B$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Strict Deployed Precision** | 7.55% | 1.46% | 7.09% | 1.46% | **-5.86 pp** | **-0.23 pp** | **+0.45 pp** |
| **All-Gold Omission Recall** | 60.00% | 100.00% | 51.03% | 100.00% | **+44.48 pp** | **-4.48 pp** | **+8.97 pp** |
| **Dedicated Omission AP** | 0.113331 | 0.027572 | 0.096293 | 0.027611 | **-0.077221** | **-0.008500** | **+0.017077** |
| **False Alarms per 1k Phones**| 107.07 | 985.44 | 97.33 | 985.44 | **+883.24** | **-4.87** | **+9.74** |
| **Binary MDD AP** | 0.321619 | 0.315171 | 0.321573 | 0.315847 | **-0.006087** | **+0.000315** | **+0.000722** |
| **Legacy Substitution AP** | 0.310022 | 0.311767 | 0.312209 | 0.312187 | **+0.000862** | **+0.001304** | **-0.001767** |
| **Greedy CTC PER** | 0.238943 | 0.239045 | 0.239045 | 0.239045 | **+0.000051** | **+0.000051** | **-0.000051** |
| **Acoustic Reliance Gap** | 0.019901 | 0.014102 | 0.022486 | 0.017230 | **-0.005528** | **+0.002857** | **+0.000543** |

---

## 2. Hypothesis Testing Verdicts

### Hypothesis $H_1$ (Objective Effect) — REJECTED
- **Claim:** Class-conditioned focal supervision ($\alpha=0.25$ on coronal/rhotic deletions $\{D, T, R\}$) can improve discrimination between true omissions and misclassified correct phonemes without sacrificing substitution detection.
- **Empirical Measured Evidence:**
  - Dedicated Omission AP dropped by **$-0.077221$** (from $0.1133$ to $0.0276$).
  - Deployed precision degraded by **$-5.86\text{ pp}$** (from $7.55\%$ to $1.46\%$).
  - False alarm rate exploded by **$+883.24$ false alarms per 1,000 canonical phones**, triggering alarms on $98.5\%$ of all tokens.
- **Verdict:** **$H_1$ is conclusively REJECTED.** The null hypothesis cannot be rejected; the intervention produced severe catastrophic logit displacement.

### Hypothesis $H_2$ (Acoustic Feature Effect) — REJECTED
- **Claim:** Additional waveform-derived frame-level acoustic evidence can help distinguish true omissions from reduced or unreleased realizations.
- **Empirical Measured Evidence:**
  - Acoustic features achieved a modest reduction in false alarms ($-4.87$ false alarms/1k, $97.33$ vs $107.07$) and increased acoustic reliance ($+0.002857$ gap improvement).
  - However, deployed precision slightly declined ($-0.23\text{ pp}$, $7.09\%$ vs $7.55\%$).
  - Dedicated Omission AP declined by $-0.008500$ ($0.0963$ vs $0.1133$).
  - All-gold omission recall dropped by $-4.48\text{ pp}$ ($51.03\%$ vs $60.00\%$).
- **Verdict:** **$H_2$ is conclusively REJECTED.** The physical acoustic features did not resolve the omission precision problem and exhibited an adverse trade-off against recall.

### Hypothesis $H_3$ (Interaction Effect) — REJECTED
- **Claim:** Combining the focal objective and additional acoustic features produces a benefit beyond their independent contributions.
- **Empirical Measured Evidence:**
  - Arm T3 performed virtually identically to Arm T1 (Deployed Precision $1.46\%$ vs $1.46\%$; Dedicated Omission AP $0.027611$ vs $0.027572$).
  - The algebraic interaction ($+0.017077$ on Omission AP) is a floor effect artifact caused by both focal models collapsing to near-zero discrimination, not a constructive super-additive synergy.
- **Verdict:** **$H_3$ is conclusively REJECTED.** Zero beneficial interaction was observed.

---

## 3. Trade-Off Analysis: What Was Gained vs What Was Lost

Every architectural and supervision intervention incurred substantial costs:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            INTERVENTION TRADEOFF AUDIT                                         │
├──────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────┤
│ Arm / Intervention   │ Measured Improvement / Gain              │ Accompanying Degradation / Cost              │
├──────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Arm T1 (Focal Loss)  │ Perfect all-gold recall (100.00%, 145/145│ Precision collapsed to 1.46%; 9,811 false    │
│                      │ detections); PER invariant (0.2390).     │ alarms; Omission AP destroyed (-75.7% rel).  │
├──────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Arm T2 (Acoustic)    │ False alarms reduced (-9.1% rel, 97.3 vs │ Omission recall dropped 8.97 pp (51% vs 60%);│
│                      │ 107.1/1k); Acoustic gap increased (+13%).│ Omission AP dropped (-15.0% rel, .096 vs .113│
├──────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Arm T3 (Combined)    │ ASR PER invariant (0.239045);            │ Catastrophic precision collapse (1.46%);     │
│                      │ Substitution AP intact (0.3122).         │ 985 false alarms / 1k; zero synergy.         │
└──────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────┘
```
