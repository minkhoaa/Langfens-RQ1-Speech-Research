# Phase 3 student results — evaluator output

Prediction arms/seeds evaluated: 14; rows: 98. DEV_SEARCH is exploratory and repeatedly development-exposed.

All policy thresholds were selected on DEV_CAL using the bounded grid; the reported DEV_SEARCH FWR is realized independently and may breach the corresponding budget. The historical Phase 2 policy table has its own fixed operating thresholds and is not mixed with recalibrated student policies.

| Budget | Seed | Gold sub/omis/ins recall | KD-All sub/omis/ins recall | KD−Gold macro-3 error F1 | Realized warning counts Gold/KD |
|---:|---:|---|---|---:|---:|
| 0.20 | 101 | 0.000/0.000/0.008 | 0.001/0.000/0.008 | +0.0006 | 1/0 |
| 1.00 | 101 | 0.065/0.201/0.008 | 0.158/0.014/0.008 | -0.0334 | 231/390 |
| 0.20 | 202 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | +0.0000 | 0/0 |
| 1.00 | 202 | 0.118/0.110/0.000 | 0.127/0.053/0.000 | -0.0306 | 262/350 |
| 0.20 | 303 | 0.000/0.000/0.016 | 0.000/0.000/0.000 | +0.0000 | 2/0 |
| 1.00 | 303 | 0.048/0.230/0.016 | 0.154/0.057/0.000 | +0.0046 | 250/376 |

## Interpretation and limitations

The outputs preserve per-class TP/FP/FN/TN, PRF1, AUPRC, ECE/Brier, correct/error confusion, selectivity, speaker breakdowns, warning ledgers, and always-CORRECT controls. Pareto membership is recomputed only across these student operating points using realized DEV_SEARCH FWR and macro-3 error F1.

Complete frozen DEV_SEARCH ledgers for UPR03, CPL02 F5-RISK-B020, and standalone S/O/I teachers were separately rescored with the same event-confusion/FWR implementation; their frozen historic decisions were not retuned on DEV_CAL. See PHASE3_FROZEN_CONTROLS.json and PHASE3_FROZEN_CONTROL_METRICS.csv. The broader historical Pareto table lacks complete ledgers and is not independently reclassified. No oracle ceiling is computed. Speaker/family uncertainty is limited by four exploratory speakers; seed-level paired evidence is available where Gold-Only and KD-All predictions both exist.

## Full-ledger frozen controls at realized DEV_SEARCH risk <= 0.20

| Frozen control | False warnings / 599 | FWR | Sub recall | Omission recall | Exact-gap recall | Macro-4 F1 |
|---|---:|---:|---:|---:|---:|---:|
| UPR03_SYNTHESIS_CHAMPION | 113 | 0.1886 | 0.059 | 0.043 | 0.056 | 0.3069 |
| CPL02_BEST_FROZEN_POLICY | 53 | 0.0885 | 0.012 | 0.043 | 0.032 | 0.2758 |
| TEACHER_I_CHAMPION | 40 | 0.0668 | 0.000 | 0.000 | 0.088 | 0.2681 |
| TEACHER_I_EXP01 | 73 | 0.1219 | 0.000 | 0.000 | 0.112 | 0.2699 |

All frozen controls above are within 0.20 by realized DEV_SEARCH risk. At seed 101 and the DEV_CAL-selected 0.20 policy, students also remain far below budget but have near-zero error recall; their risk point should not be interpreted as outperforming a comparator on error coverage.

Across the three matched seeds at budget 0.20, mean paired KD-All−Gold-Only changes (sample SD across seeds): SUB recall / OM recall / I recall / macro-3 error F1 / macro-4 F1 = +0.0002 ± 0.0003 / +0.0000 ± 0.0000 / -0.0053 ± 0.0092 / +0.0002 ± 0.0003 / -0.0025 ± 0.0046.

Across the three matched seeds at budget 1.00, mean paired KD-All−Gold-Only changes (sample SD across seeds): SUB recall / OM recall / I recall / macro-3 error F1 / macro-4 F1 = +0.0696 ± 0.0524 / -0.1388 ± 0.0708 / -0.0053 ± 0.0092 / -0.0198 ± 0.0212 / -0.0126 ± 0.0063.
