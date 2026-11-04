# Matching historical V2 genuine-Z reference

Same 9,919 aligned gold keys verified by hash for all 8 corrected cells, same frozen prompt/CMUdict/sub_first policy, same score-family/AP populations and CPU FP32, batch size 1. T-OP is an external architecture/training-run reference, NOT the matched factorial control C0. B-Control subtype AP remains unsupported/null. Separate CPU executions can produce small floating-point differences; no bitwise equality to old V2 encoder features is claimed.

| Historical fixed3600 | Binary AP | Legacy omit AP | Dedicated omit AP | Legacy sub AP | Dedicated sub AP |
|---|---|---|---|---|---|
| t_op_fixed_budget_3600 | 0.318618752 | 0.033941636 | 0.098560390 | 0.308973738 | 0.334102193 |
| b_control_fixed_budget_3600 | 0.316194495 | 0.034287724 | NA | 0.306579784 | NA |

| Arm | Dedicated omit AP | Legacy omit AP | Precision | Recall all | Recall aligned | FA/1,000 | Binary AP | Legacy sub AP | Dedicated sub AP | PER |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.104690254 | 0.033765022 | 0.070298770 | 0.551724138 | 0.583941606 | 106.267577340 | 0.316560146 | 0.306524807 | 0.333795147 | 0.238943427 |
| T1 | 0.029927641 | 0.024291675 | 0.027375201 | 0.117241379 | 0.124087591 | 60.666934512 | 0.311004684 | 0.308639164 | 0.341037241 | 0.239044631 |
| T2 | 0.086463470 | 0.031205329 | 0.063870353 | 0.462068966 | 0.489051095 | 98.633989554 | 0.318158944 | 0.310863253 | 0.335316426 | 0.238943427 |
| T3 | 0.031705546 | 0.025011971 | 0.040968343 | 0.151724138 | 0.160583942 | 51.727601446 | 0.313896816 | 0.311610067 | 0.337825388 | 0.239044631 |

Frozen FIT-only canonical prior remains unavailable; no replacement was fitted. This gap prevents a blanket full-control-completion claim.
