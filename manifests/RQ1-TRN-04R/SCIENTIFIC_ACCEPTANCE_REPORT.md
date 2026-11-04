# Registered scientific acceptance recovery

All four fixed-budget arms fail both Level A and Level B. No arm reaches 25%/30% strict omission precision at the primary operating point. C0/T2 meet recall gates; T1/T3 do not. All fail primary false-alarm limits. T1 also fails the external binary-AP floor. Fixed-budget PER and legacy substitution floors pass. Every acoustic mismatch gap exceeds.010, a sensitivity control, not causal localization.

| Arm | Level | Metric | Value | Rule | Threshold | Result |
|---|---|---|---|---|---|---|
| C0 | Level_A | PER | 0.238943427 | <= | 0.246980000 | pass |
| C0 | Level_A | binary_ap | 0.316560146 | >= | 0.313619000 | pass |
| C0 | Level_A | omission_fa_per_1000_cal15 | 106.267577340 | <= | 30 | fail |
| C0 | Level_A | omission_precision_cal15 | 0.070298770 | >= | 0.250000000 | fail |
| C0 | Level_A | omission_recall_all_gold_cal15 | 0.551724138 | >= | 0.300000000 | pass |
| C0 | Level_A | substitution_legacy_ap | 0.306524807 | >= | 0.303974000 | pass |
| C0 | Level_B | PER | 0.238943427 | <= | 0.246980000 | pass |
| C0 | Level_B | acoustic_gap_binary_ap | 0.016813967 | >= | 0.010000000 | pass |
| C0 | Level_B | binary_ap | 0.316560146 | >= | 0.313619000 | pass |
| C0 | Level_B | omission_fa_per_1000_cal15 | 106.267577340 | <= | 15 | fail |
| C0 | Level_B | omission_precision_cal15 | 0.070298770 | >= | 0.300000000 | fail |
| C0 | Level_B | omission_recall_all_gold_cal15 | 0.551724138 | >= | 0.400000000 | pass |
| C0 | Level_B | substitution_legacy_ap | 0.306524807 | >= | 0.303974000 | pass |
| T1 | Level_A | PER | 0.239044631 | <= | 0.246980000 | pass |
| T1 | Level_A | binary_ap | 0.311004684 | >= | 0.313619000 | fail |
| T1 | Level_A | omission_fa_per_1000_cal15 | 60.666934512 | <= | 30 | fail |
| T1 | Level_A | omission_precision_cal15 | 0.027375201 | >= | 0.250000000 | fail |
| T1 | Level_A | omission_recall_all_gold_cal15 | 0.117241379 | >= | 0.300000000 | fail |
| T1 | Level_A | substitution_legacy_ap | 0.308639164 | >= | 0.303974000 | pass |
| T1 | Level_B | PER | 0.239044631 | <= | 0.246980000 | pass |
| T1 | Level_B | acoustic_gap_binary_ap | 0.011733800 | >= | 0.010000000 | pass |
| T1 | Level_B | binary_ap | 0.311004684 | >= | 0.313619000 | fail |
| T1 | Level_B | omission_fa_per_1000_cal15 | 60.666934512 | <= | 15 | fail |
| T1 | Level_B | omission_precision_cal15 | 0.027375201 | >= | 0.300000000 | fail |
| T1 | Level_B | omission_recall_all_gold_cal15 | 0.117241379 | >= | 0.400000000 | fail |
| T1 | Level_B | substitution_legacy_ap | 0.308639164 | >= | 0.303974000 | pass |
| T2 | Level_A | PER | 0.238943427 | <= | 0.246980000 | pass |
| T2 | Level_A | binary_ap | 0.318158944 | >= | 0.313619000 | pass |
| T2 | Level_A | omission_fa_per_1000_cal15 | 98.633989554 | <= | 30 | fail |
| T2 | Level_A | omission_precision_cal15 | 0.063870353 | >= | 0.250000000 | fail |
| T2 | Level_A | omission_recall_all_gold_cal15 | 0.462068966 | >= | 0.300000000 | pass |
| T2 | Level_A | substitution_legacy_ap | 0.310863253 | >= | 0.303974000 | pass |
| T2 | Level_B | PER | 0.238943427 | <= | 0.246980000 | pass |
| T2 | Level_B | acoustic_gap_binary_ap | 0.020901152 | >= | 0.010000000 | pass |
| T2 | Level_B | binary_ap | 0.318158944 | >= | 0.313619000 | pass |
| T2 | Level_B | omission_fa_per_1000_cal15 | 98.633989554 | <= | 15 | fail |
| T2 | Level_B | omission_precision_cal15 | 0.063870353 | >= | 0.300000000 | fail |
| T2 | Level_B | omission_recall_all_gold_cal15 | 0.462068966 | >= | 0.400000000 | pass |
| T2 | Level_B | substitution_legacy_ap | 0.310863253 | >= | 0.303974000 | pass |
| T3 | Level_A | PER | 0.239044631 | <= | 0.246980000 | pass |
| T3 | Level_A | binary_ap | 0.313896816 | >= | 0.313619000 | pass |
| T3 | Level_A | omission_fa_per_1000_cal15 | 51.727601446 | <= | 30 | fail |
| T3 | Level_A | omission_precision_cal15 | 0.040968343 | >= | 0.250000000 | fail |
| T3 | Level_A | omission_recall_all_gold_cal15 | 0.151724138 | >= | 0.300000000 | fail |
| T3 | Level_A | substitution_legacy_ap | 0.311610067 | >= | 0.303974000 | pass |
| T3 | Level_B | PER | 0.239044631 | <= | 0.246980000 | pass |
| T3 | Level_B | acoustic_gap_binary_ap | 0.018138828 | >= | 0.010000000 | pass |
| T3 | Level_B | binary_ap | 0.313896816 | >= | 0.313619000 | pass |
| T3 | Level_B | omission_fa_per_1000_cal15 | 51.727601446 | <= | 15 | fail |
| T3 | Level_B | omission_precision_cal15 | 0.040968343 | >= | 0.300000000 | fail |
| T3 | Level_B | omission_recall_all_gold_cal15 | 0.151724138 | >= | 0.400000000 | fail |
| T3 | Level_B | substitution_legacy_ap | 0.311610067 | >= | 0.303974000 | pass |

## Internal C0 descriptive non-inferiority margins

These point comparisons use registered .005 margins; they are not inferential non-inferiority tests. External absolute gates above remain distinct.

| Arm | Metric | Delta vs C0 | Rule | Margin | Point disposition |
|---|---|---|---|---|---|
| T1 | binary_ap | -0.005555461 | >= | -0.005000000 | descriptive margin failed |
| T1 | legacy_substitution_ap | 0.002114357 | >= | -0.005000000 | descriptive margin met |
| T1 | greedy_per | 0.000101204 | <= | 0.005000000 | descriptive margin met |
| T2 | binary_ap | 0.001598798 | >= | -0.005000000 | descriptive margin met |
| T2 | legacy_substitution_ap | 0.004338446 | >= | -0.005000000 | descriptive margin met |
| T2 | greedy_per | 0.000000000 | <= | 0.005000000 | descriptive margin met |
| T3 | binary_ap | -0.002663330 | >= | -0.005000000 | descriptive margin met |
| T3 | legacy_substitution_ap | 0.005085260 | >= | -0.005000000 | descriptive margin met |
| T3 | greedy_per | 0.000101204 | <= | 0.005000000 | descriptive margin met |

One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.
