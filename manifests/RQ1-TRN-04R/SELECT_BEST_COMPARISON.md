# Existing SELECT-best checkpoints — secondary only

Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.

Successful updates: C0=2247,T1=897,T2=3600,T3=2247. These unequal-exposure snapshots are not a fixed-budget factorial causal contrast. T2 best and fixed have identical model-state fingerprints despite differing serialized training-state bytes. Registered primary gates target fixed-budget, not a replacement checkpoint selected from this table.

| Arm | Dedicated omit AP | Legacy omit AP | Precision | Recall all | Recall aligned | FA/1,000 | Binary AP | Legacy sub AP | Dedicated sub AP | PER |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.110942488 | 0.035882130 | 0.071258907 | 0.413793103 | 0.437956204 | 78.545600643 | 0.309461713 | 0.297625243 | 0.323354328 | 0.249367473 |
| T1 | 0.025631733 | 0.023895989 | 0.031250000 | 0.082758621 | 0.087591241 | 37.364403375 | 0.308140671 | 0.306309436 | 0.339992089 | 0.282360085 |
| T2 | 0.086463470 | 0.031205329 | 0.063870353 | 0.462068966 | 0.489051095 | 98.633989554 | 0.318158944 | 0.310863253 | 0.335316426 | 0.238943427 |
| T3 | 0.024483826 | 0.026971583 | 0.023622047 | 0.103448276 | 0.109489051 | 62.274005625 | 0.281912278 | 0.273950379 | 0.325517970 | 0.249468677 |

One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.
