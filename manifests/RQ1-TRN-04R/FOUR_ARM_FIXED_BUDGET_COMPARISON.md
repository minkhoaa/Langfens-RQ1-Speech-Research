# Four-arm fixed-budget genuine-Z comparison

Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.

All endpoints are exactly 3,600 successful updates; original bytes unchanged.

| Arm | Dedicated omit AP | Legacy omit AP | Precision | Recall all | Recall aligned | FA/1,000 | Binary AP | Legacy sub AP | Dedicated sub AP | PER |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.104690254 | 0.033765022 | 0.070298770 | 0.551724138 | 0.583941606 | 106.267577340 | 0.316560146 | 0.306524807 | 0.333795147 | 0.238943427 |
| T1 | 0.029927641 | 0.024291675 | 0.027375201 | 0.117241379 | 0.124087591 | 60.666934512 | 0.311004684 | 0.308639164 | 0.341037241 | 0.239044631 |
| T2 | 0.086463470 | 0.031205329 | 0.063870353 | 0.462068966 | 0.489051095 | 98.633989554 | 0.318158944 | 0.310863253 | 0.335316426 | 0.238943427 |
| T3 | 0.031705546 | 0.025011971 | 0.040968343 | 0.151724138 | 0.160583942 | 51.727601446 | 0.313896816 | 0.311610067 | 0.337825388 | 0.239044631 |

| Arm | TP | FP_correct | FP_substitution | FN all | Matched binary AP | Mismatched binary AP | Gap |
|---|---|---|---|---|---|---|---|
| C0 | 80 | 999 | 59 | 65 | 0.316560146 | 0.299746178 | 0.016813967 |
| T1 | 17 | 591 | 13 | 128 | 0.311004684 | 0.299270884 | 0.011733800 |
| T2 | 67 | 922 | 60 | 78 | 0.318158944 | 0.297257792 | 0.020901152 |
| T3 | 22 | 493 | 22 | 123 | 0.313896816 | 0.295757988 | 0.018138828 |

One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.
