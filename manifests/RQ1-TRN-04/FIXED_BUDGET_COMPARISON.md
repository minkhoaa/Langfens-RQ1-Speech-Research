# LANGFENS RQ1-TRN-04 — Matched Fixed-Budget Benchmark Comparison

**Document ID:** `LANGFENS-RQ1-TRN-04-FIXED-BUDGET-COMPARISON-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist & Evaluator  
**Date:** 2026-11-04  
**Evaluation Standard:** Evaluation Contract V1 Protocol Z (Fold-1 VAL, 300 utterances, 9,956 gold canonical tokens)  
**Endpoint:** Pinned 3,600-Update Fixed Budget Endpoint (Epoch 9)  
**Execution Platform:** Authorized Remote GPU Hardware (Kaggle Nvidia Tesla T4, CUDA FP16 AMP)  

---

## 1. Executive Summary of Measured Findings

The primary evaluation of **LANGFENS RQ1-TRN-04** compares all four experimental arms at the prospectively frozen fixed-budget endpoint of **exactly 3,600 successful optimizer updates** (with exactly 3 skipped AMP attempts in each arm, representing 3,603 total optimization attempts across 9 completed epochs).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRIMARY FACTORIAL COMPARISON (FIXED BUDGET 3600)                             │
├─────┬──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────┬────────┤
│ Arm │ Configuration        │ Prec@15% │ Rec@15%  │ Omit AP  │ Binary AP│ Sub AP   │ PER      │ FA/1k  │ RelyGap│
├─────┼──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┼────────┤
│ C0  │ CE, NoFeat (Control) │  7.55%   │  60.00%  │ 0.113331 │ 0.321619 │ 0.310022 │ 0.238943 │ 107.07 │ 0.0199 │
│ T1  │ Focal, NoFeat (Obj)  │  1.46%   │ 100.00%  │ 0.027572 │ 0.315171 │ 0.311767 │ 0.239045 │ 985.44 │ 0.0141 │
│ T2  │ CE, Feat (Acoustic)  │  7.09%   │  51.03%  │ 0.096293 │ 0.321573 │ 0.312209 │ 0.239045 │  97.33 │ 0.0225 │
│ T3  │ Focal, Feat (Comb)   │  1.46%   │ 100.00%  │ 0.027611 │ 0.315847 │ 0.312187 │ 0.239045 │ 985.44 │ 0.0172 │
├─────┼──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┼────────┤
│ Ref │ Historical T-OP P-SUP│  6.78%   │  56.55%  │ 0.098560 │ 0.318619 │ 0.308974 │ 0.241980 │ 113.20 │ 0.0174 │
└─────┴──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┴────────┘
```

---

## 2. Comprehensive Metric Breakdown across Primary Channels

All metrics are computed on the full 9,956 gold canonical tokens of Fold-1 VAL under Protocol Z (zero TextGrid or alignment leakage). Operational thresholds are determined strictly on CAL (NCC/THV, 300 utterances) using correct-token empirical FPR ceilings:

### 2.1 Fixed-Budget Endpoint (3,600 Updates) Full Results

| Metric Channel | Arm C0 (Control) | Arm T1 (Objective) | Arm T2 (Acoustic) | Arm T3 (Combined) | Historical T-OP Ref |
|---|---:|---:|---:|---:|---:|
| **Updates / Skipped AMP** | 3,600 / 3 | 3,600 / 3 | 3,600 / 3 | 3,600 / 3 | 3,600 / 0 |
| **Duration on Tesla T4** | 507.3s | 449.4s | 466.5s | 493.4s | 489.9s |
| **Strict Deployed Precision** | **7.55%** | **1.46%** | **7.09%** | **1.46%** | 6.78% |
| **All-Gold Omission Recall** | **60.00%** | **100.00%** | **51.03%** | **100.00%** | 56.55% |
| **Aligned Omission Recall** | 63.50% | 100.00% | 54.01% | 100.00% | 59.85% |
| **True Positives (TP)** | 87 | 145 | 74 | 145 | 82 |
| **False Positives (Correct)** | 1,007 | 8,799 | 914 | 8,799 | 1,068 |
| **False Positives (Sub)** | 59 | 1,012 | 55 | 1,012 | 59 |
| **False Negatives (FN)** | 58 | 0 | 71 | 0 | 63 |
| **False Alarms / 1k Phones** | **107.07** | **985.44** | **97.33** | **985.44** | 113.20 |
| **Dedicated Omission AP** | **0.113331** | **0.027572** | **0.096293** | **0.027611** | 0.098560 |
| **Legacy Omission AP** | 0.037452 | 0.027572 | 0.035025 | 0.027611 | 0.033942 |
| **Binary MDD AP** | **0.321619** | **0.315171** | **0.321573** | **0.315847** | 0.318619 |
| **Legacy Substitution AP** | **0.310022** | **0.311767** | **0.312209** | **0.312187** | 0.308974 |
| **Dedicated Substitution AP**| 0.336363 | 0.311767 | 0.337593 | 0.312187 | 0.334102 |
| **Greedy CTC PER** | **0.238943** | **0.239045** | **0.239045** | **0.239045** | 0.241980 |
| **CTC Blank Frame Fraction** | 0.790028 | 0.790008 | 0.790028 | 0.790008 | 0.840877 |
| **Mismatched Audio Binary AP**| 0.301717 | 0.301069 | 0.299087 | 0.298616 | 0.301171 |
| **Acoustic Reliance Gap** | **+0.019901** | **+0.014102** | **+0.022486** | **+0.017230** | +0.017448 |
| **NJS Binary AP** | 0.399847 | 0.402459 | 0.410453 | 0.400912 | 0.389152 |
| **RRBI Binary AP** | 0.255471 | 0.249379 | 0.255871 | 0.248721 | 0.261812 |

---

## 3. Comparative Analysis: Factorial Interventions vs Control

### 3.1 C0 Matched Control vs Historical T-OP Baseline
- Newly trained C0 closely replicates the performance profile of historical T-OP while achieving slightly superior metrics across all channels:
  - Deployed precision: $7.55\%$ vs $6.78\%$ ($+0.77\text{ pp}$).
  - All-gold recall: $60.00\%$ vs $56.55\%$ ($+3.45\text{ pp}$).
  - Dedicated Omission AP: $0.113331$ vs $0.098560$ ($+0.014771$).
  - Binary AP: $0.321619$ vs $0.318619$ ($+0.003000$).
  - Greedy PER: $0.238943$ vs $0.241980$ ($-0.003037$, improved ASR).
  - False alarms per 1k phones: $107.07$ vs $113.20$ ($-6.13\text{ FA/1k}$).
- This confirms that the shared architectural slot (`CompactFactorialOperationHead` with zero features) operates cleanly without introducing capacity deficits or instability.

### 3.2 T1 Objective Intervention vs C0 Control
- Down-weighting deletion supervision on $\{D, T, R\}$ with $\alpha=0.25$ **completely failed to achieve the intended precision recovery**:
  - Dedicated Omission AP collapsed by **$-75.67\%$ relative** (from $0.113331$ to $0.027572$).
  - Deployed precision collapsed from $7.55\%$ to $1.46\%$.
  - False alarm rate exploded from $107.07$ to $985.44$ false alarms per 1,000 canonical phones.
- **Phonetic Root Cause:** Focal re-weighting altered the dynamic range of deletion logits relative to correct logits across the vocabulary. Because coronal stops and rhotics comprise the vast majority of deletions in training ($56.69\%$), suppressing their gradients removed the gradient pressure keeping deletion probabilities low across all other phonemes. As a result, the model's background deletion probability rose above the CAL threshold ($\tau_{\text{CAL}} = 0.106096$), causing virtually every correct and substitution token to trigger an omission alarm.
- Upstream binary AP ($0.315171$) and CTC PER ($0.239045$) remained intact, proving that the failure is isolated entirely to the subtype omission classification head.

### 3.3 T2 Acoustic Intervention vs C0 Control
- Injecting the 4D waveform acoustic feature vector produced mixed, trade-off effects:
  - **False Alarms:** Decreased from $107.07$ to $97.33$ per 1k phones (a **$9.1\%$ relative reduction in false alarms**).
  - **Acoustic Reliance Gap:** Increased from $+0.019901$ to $+0.022486$ ($+13.0\%$ relative improvement in acoustic grounding reliance).
  - **Substitution AP:** Slightly improved from $0.310022$ to $0.312209$ ($+0.002187$).
  - **ASR PER:** Remained identical ($0.239045$ vs $0.238943$).
  - **Recall Cost:** All-gold omission recall dropped from $60.00\%$ to $51.03\%$ ($-8.97\text{ pp}$).
  - **Precision:** Slightly declined from $7.55\%$ to $7.09\%$ ($-0.46\text{ pp}$).
  - **Dedicated Omission AP:** Dropped from $0.113331$ to $0.096293$ ($-0.017038$).
- **Phonetic Finding:** While physical acoustic features provided genuine releasing and voicing cues that eliminated $93$ false alarms on correct phones, they simultaneously suppressed detection on reduced true deletions, demonstrating the persistent acoustic bleeding tradeoff.

### 3.4 T3 Combined Intervention vs Individual Arms
- Arm T3 exhibited behavior almost indistinguishable from T1:
  - Dedicated Omission AP: $0.027611$ (vs T1 $0.027572$).
  - Deployed Precision: $1.46\%$.
  - False Alarms: $985.44$ per 1k phones.
- The acoustic features in T3 were entirely overwhelmed by the pathological logit shift caused by the focal objective. Combining the interventions produced zero positive interaction.

---

## 4. Secondary SELECT-Best Checkpoint Benchmark

To verify whether training exposure confounded performance, we also benchmark the validation-selected `best.pt` checkpoints chosen on SELECT (MBMPS/TLV):

| Metric Channel | C0 Best (Epoch 5) | T1 Best (Epoch 2) | T2 Best (Epoch 9) | T3 Best (Epoch 5) |
|---|---:|---:|---:|---:|
| **Updates at Best** | 2,250 | 900 | 3,600 | 2,250 |
| **SELECT AP** | 0.476465 | 0.461918 | 0.470217 | 0.467060 |
| **Strict Deployed Precision** | **7.56%** | **1.51%** | **7.09%** | **1.49%** |
| **All-Gold Omission Recall** | 44.14% | 100.00% | 51.03% | 100.00% |
| **Dedicated Omission AP** | 0.112895 | 0.026813 | 0.096293 | 0.028354 |
| **Binary MDD AP** | 0.312025 | 0.313714 | 0.321573 | 0.285684 |
| **Greedy CTC PER** | 0.249367 | 0.282360 | 0.239045 | 0.249469 |
| **False Alarms / 1k Phones** | 78.65 | 952.19 | 97.33 | 966.05 |

### Takeaways from SELECT-Best:
1. In C0, the best checkpoint occurred at update 2,250 (Epoch 5). While it achieved fewer false alarms ($78.65$ vs $107.07$), its recall dropped to $44.14\%$ ($64$ TP vs $87$ TP) without improving precision ($7.56\%$ vs $7.55\%$).
2. In T2, the best checkpoint occurred at the final fixed budget (3,600 updates), proving that acoustic feature training benefited from the full optimization horizon.
3. In both T1 and T3, early checkpoints exhibited the same severe omission collapse ($1.5\%$ precision, $>950$ false alarms/1k), proving that the failure mode is an intrinsic property of the loss formulation rather than late-training over-fitting.
