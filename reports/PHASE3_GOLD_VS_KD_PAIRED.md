# Phase 3 student results — evaluator output

Prediction arms/seeds evaluated: 14; rows: 98. DEV_SEARCH is exploratory and repeatedly development-exposed.

Protocol `student-eval-v2-upper-tail` selected every policy on DEV_CAL only: 21 coarse quantiles, eight predefined upper-tail quantiles (0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9999), plus `nextafter(max,+inf)` for a no-warning option. This was added to resolve coarse-grid sparsity at the requested risk budgets; the 32×32 candidate grid is fixed and was not tuned on DEV_SEARCH. Selection maximizes macro-3 error F1 = mean(SUB, OMISSION, exact-gap INSERTION F1); macro-4 is tie-break only. Canonical manifest labels and event order are checked before scoring. DEV_SEARCH realized FWR is reported independently; no student policy breached its requested budget. Frozen historical control thresholds remain separate and were not recalibrated.

| Budget | Seed | Gold sub/omis/ins recall | KD-All sub/omis/ins recall | KD−Gold macro-3 error F1 | Realized warning counts Gold/KD |
|---:|---:|---|---|---:|---:|
| 0.20 | 101 | 0.001/0.014/0.232 | 0.002/0.010/0.256 | +0.0014 | 63/73 |
| 1.00 | 101 | 0.065/0.201/0.328 | 0.083/0.163/0.320 | -0.0015 | 377/376 |
| 0.20 | 202 | 0.016/0.024/0.208 | 0.018/0.014/0.208 | -0.0049 | 69/73 |
| 1.00 | 202 | 0.118/0.110/0.208 | 0.129/0.096/0.208 | +0.0000 | 308/317 |
| 0.20 | 303 | 0.000/0.019/0.152 | 0.001/0.010/0.184 | +0.0043 | 47/56 |
| 1.00 | 303 | 0.048/0.230/0.280 | 0.055/0.220/0.272 | +0.0013 | 355/345 |

## Interpretation and limitations

The outputs preserve per-class TP/FP/FN/TN, PRF1, tie-correct AUPRC, ECE/Brier, correct/error confusion, selectivity, speaker/family breakdowns, warning ledgers, and always-CORRECT controls. Macro-3 error F1 includes exact-gap insertion; target-only macro-3 including CORRECT is reported separately. Pareto membership is recomputed only across the unique student threshold operating points using realized DEV_SEARCH FWR and macro-3 error F1. Four-speaker paired bootstrap intervals are provided in `PHASE3_SPEAKER_CLUSTER_BOOTSTRAP.json`; with only four clusters they are descriptive/exploratory, not strong inferential evidence.

Complete frozen DEV_SEARCH ledgers for UPR03, CPL02 F5-RISK-B020, and standalone S/O/I teachers were separately rescored with the same event-confusion/FWR implementation; their frozen historic decisions were not retuned on DEV_CAL. See PHASE3_FROZEN_CONTROLS.json and PHASE3_FROZEN_CONTROL_METRICS.csv. The broader historical Pareto table lacks complete ledgers and is not independently reclassified. No oracle ceiling is computed. Speaker/family uncertainty is limited by four exploratory speakers; seed-level paired evidence is available where Gold-Only and KD-All predictions both exist.

## Full-ledger frozen controls at realized DEV_SEARCH risk <= 0.20

| Frozen control | False warnings / 599 | FWR | Sub recall | Omission recall | Exact-gap recall | Macro-4 F1 |
|---|---:|---:|---:|---:|---:|---:|
| UPR03_SYNTHESIS_CHAMPION | 113 | 0.1886 | 0.059 | 0.043 | 0.056 | 0.3069 |
| CPL02_BEST_FROZEN_POLICY | 53 | 0.0885 | 0.012 | 0.043 | 0.032 | 0.2758 |
| TEACHER_I_CHAMPION | 40 | 0.0668 | 0.000 | 0.000 | 0.088 | 0.2681 |
| TEACHER_I_EXP01 | 73 | 0.1219 | 0.000 | 0.000 | 0.112 | 0.2699 |

All frozen controls above are within 0.20 by realized DEV_SEARCH risk. At the DEV_CAL-selected 0.20 budget, the three matched Gold-Only runs realized 63/69/47 warnings and KD-All 73/73/56 (the DEV_SEARCH allowance is 119). Recall remains low: seed 101 Gold/KD SUB/O/I recall is 0.0006/0.0144/0.232/0.0017/0.0096/0.256 respectively. Student operation at lower realized FWR is not evidence of superiority at matched realized risk; compare the full budget tables and frozen-control points.

Across the three matched seeds at budget 0.20, mean paired KD-All−Gold-Only changes (sample SD across seeds): SUB recall / OM recall / I recall / macro-3 error F1 / macro-4 F1 = +0.0013 ± 0.0009 / -0.0080 ± 0.0028 / +0.0187 ± 0.0167 / +0.0003 ± 0.0047 / +0.0002 ± 0.0035.

Across the three matched seeds at budget 1.00, mean paired KD-All−Gold-Only changes (sample SD across seeds): SUB recall / OM recall / I recall / macro-3 error F1 / macro-4 F1 = +0.0121 ± 0.0051 / -0.0207 ± 0.0154 / -0.0053 ± 0.0046 / -0.0001 ± 0.0014 / +0.0001 ± 0.0011.
