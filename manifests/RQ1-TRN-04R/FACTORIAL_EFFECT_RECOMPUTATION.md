# Matched factorial effects — fixed budget

Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.

| Endpoint | Favorable direction | T1-C0 | T2-C0 | T3-C0 | T3-T2-T1+C0 |
|---|---|---|---|---|---|
| PER | lower_is_better | 0.000101204 | 0.000000000 | 0.000101204 | 0.000000000 |
| acoustic_gap_binary_ap | higher_is_better | -0.005080167 | 0.004087184 | 0.001324860 | 0.002317843 |
| binary_ap | higher_is_better | -0.005555461 | 0.001598798 | -0.002663330 | 0.001293334 |
| omission_dedicated_ap | higher_is_better | -0.074762613 | -0.018226784 | -0.072984708 | 0.020004689 |
| omission_fa_per_1000_cal15 | lower_is_better | -45.600642828 | -7.633587786 | -54.539975894 | -1.305745279 |
| omission_precision_cal15 | higher_is_better | -0.042923568 | -0.006428417 | -0.029330427 | 0.020021558 |
| omission_recall_all_gold_cal15 | higher_is_better | -0.434482759 | -0.089655172 | -0.400000000 | 0.124137931 |
| substitution_dedicated_ap | higher_is_better | 0.007242094 | 0.001521280 | 0.004030241 | -0.004733133 |
| substitution_legacy_ap | higher_is_better | 0.002114357 | 0.004338446 | 0.005085260 | -0.001367544 |

T3 has favorable descriptive omission AP/precision/recall interaction terms and improves these endpoints over T1, but remains far below C0. Interaction is scale-dependent and does not prove beneficial practical synergy. Substitution-AP interactions are unfavorable. T2 improves binary/substitution AP and lowers false alarms, but worsens omission AP/precision/recall; strict Pareto domination by C0 is unsupported.

One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.
