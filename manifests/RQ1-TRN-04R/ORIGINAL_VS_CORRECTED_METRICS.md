# Original versus corrected values

Original TRN-04 numbers are preserved claims, not valid genuine-Z measurements. Changes combine protocol/input/population and score-family corrections, not new training effects. Focal dedicated channels were formerly missing and legacy AP was relabelled dedicated; old deployed focal s_error was compared with a deletion-score CAL threshold. Such differences are not same-estimand performance gains. WFC old heuristic and adjudicated estimates have no valid direct delta and are not reconstructed.

| Endpoint | Arm | Metric | Original claimed | Corrected | Delta corrected-original |
|---|---|---|---|---|---|
| fixed_budget_3600 | C0 | dedicated_omission_ap | 0.113330859 | 0.104690254 | -0.008640605 |
| fixed_budget_3600 | C0 | legacy_omission_ap | 0.037452411 | 0.033765022 | -0.003687389 |
| fixed_budget_3600 | C0 | strict_deployed_omission_precision | 0.075455334 | 0.070298770 | -0.005156564 |
| fixed_budget_3600 | C0 | all_gold_omission_recall | 0.600000000 | 0.551724138 | -0.048275862 |
| fixed_budget_3600 | C0 | false_alarms_per_1000_canonical | 107.071112897 | 106.267577340 | -0.803535556 |
| fixed_budget_3600 | C0 | binary_ap | 0.321618669 | 0.316560146 | -0.005058523 |
| fixed_budget_3600 | C0 | legacy_substitution_ap | 0.310022364 | 0.306524807 | -0.003497558 |
| fixed_budget_3600 | C0 | dedicated_substitution_ap | 0.336362790 | 0.333795147 | -0.002567643 |
| fixed_budget_3600 | C0 | greedy_per | 0.238943427 | 0.238943427 | 0.000000000 |
| fixed_budget_3600 | C0 | mismatched_audio_binary_ap | 0.301717369 | 0.299746178 | -0.001971191 |
| fixed_budget_3600 | C0 | acoustic_reliance_gap | 0.019901299 | 0.016813967 | -0.003087332 |
| fixed_budget_3600 | T1 | dedicated_omission_ap | 0.027572370 | 0.029927641 | 0.002355271 |
| fixed_budget_3600 | T1 | legacy_omission_ap | 0.027572370 | 0.024291675 | -0.003280694 |
| fixed_budget_3600 | T1 | strict_deployed_omission_precision | 0.014564082 | 0.027375201 | 0.012811119 |
| fixed_budget_3600 | T1 | all_gold_omission_recall | 1.000000000 | 0.117241379 | -0.882758621 |
| fixed_budget_3600 | T1 | false_alarms_per_1000_canonical | 985.435918039 | 60.666934512 | -924.768983528 |
| fixed_budget_3600 | T1 | binary_ap | 0.315171121 | 0.311004684 | -0.004166437 |
| fixed_budget_3600 | T1 | legacy_substitution_ap | 0.311766614 | 0.308639164 | -0.003127450 |
| fixed_budget_3600 | T1 | dedicated_substitution_ap | 0.311766614 | 0.341037241 | 0.029270628 |
| fixed_budget_3600 | T1 | greedy_per | 0.239044631 | 0.239044631 | 0.000000000 |
| fixed_budget_3600 | T1 | mismatched_audio_binary_ap | 0.301069339 | 0.299270884 | -0.001798454 |
| fixed_budget_3600 | T1 | acoustic_reliance_gap | 0.014101783 | 0.011733800 | -0.002367983 |
| fixed_budget_3600 | T2 | dedicated_omission_ap | 0.096292659 | 0.086463470 | -0.009829190 |
| fixed_budget_3600 | T2 | legacy_omission_ap | 0.035025084 | 0.031205329 | -0.003819755 |
| fixed_budget_3600 | T2 | strict_deployed_omission_precision | 0.070949185 | 0.063870353 | -0.007078832 |
| fixed_budget_3600 | T2 | all_gold_omission_recall | 0.510344828 | 0.462068966 | -0.048275862 |
| fixed_budget_3600 | T2 | false_alarms_per_1000_canonical | 97.328244275 | 98.633989554 | 1.305745279 |
| fixed_budget_3600 | T2 | binary_ap | 0.321573059 | 0.318158944 | -0.003414115 |
| fixed_budget_3600 | T2 | legacy_substitution_ap | 0.312209457 | 0.310863253 | -0.001346204 |
| fixed_budget_3600 | T2 | dedicated_substitution_ap | 0.337593421 | 0.335316426 | -0.002276995 |
| fixed_budget_3600 | T2 | greedy_per | 0.239044631 | 0.238943427 | -0.000101204 |
| fixed_budget_3600 | T2 | mismatched_audio_binary_ap | 0.299087404 | 0.297257792 | -0.001829612 |
| fixed_budget_3600 | T2 | acoustic_reliance_gap | 0.022485654 | 0.020901152 | -0.001584503 |
| fixed_budget_3600 | T3 | dedicated_omission_ap | 0.027610947 | 0.031705546 | 0.004094599 |
| fixed_budget_3600 | T3 | legacy_omission_ap | 0.027610947 | 0.025011971 | -0.002598976 |
| fixed_budget_3600 | T3 | strict_deployed_omission_precision | 0.014564082 | 0.040968343 | 0.026404261 |
| fixed_budget_3600 | T3 | all_gold_omission_recall | 1.000000000 | 0.151724138 | -0.848275862 |
| fixed_budget_3600 | T3 | false_alarms_per_1000_canonical | 985.435918039 | 51.727601446 | -933.708316593 |
| fixed_budget_3600 | T3 | binary_ap | 0.315846785 | 0.313896816 | -0.001949969 |
| fixed_budget_3600 | T3 | legacy_substitution_ap | 0.312187177 | 0.311610067 | -0.000577110 |
| fixed_budget_3600 | T3 | dedicated_substitution_ap | 0.312187177 | 0.337825388 | 0.025638211 |
| fixed_budget_3600 | T3 | greedy_per | 0.239044631 | 0.239044631 | 0.000000000 |
| fixed_budget_3600 | T3 | mismatched_audio_binary_ap | 0.298616384 | 0.295757988 | -0.002858395 |
| fixed_budget_3600 | T3 | acoustic_reliance_gap | 0.017230401 | 0.018138828 | 0.000908426 |
| best | C0 | dedicated_omission_ap | 0.112894515 | 0.110942488 | -0.001952027 |
| best | C0 | legacy_omission_ap | 0.038225018 | 0.035882130 | -0.002342888 |
| best | C0 | strict_deployed_omission_precision | 0.075560803 | 0.071258907 | -0.004301895 |
| best | C0 | all_gold_omission_recall | 0.441379310 | 0.413793103 | -0.027586207 |
| best | C0 | false_alarms_per_1000_canonical | 78.646042587 | 78.545600643 | -0.100441945 |
| best | C0 | binary_ap | 0.312025260 | 0.309461713 | -0.002563547 |
| best | C0 | legacy_substitution_ap | 0.299148746 | 0.297625243 | -0.001523502 |
| best | C0 | dedicated_substitution_ap | 0.326214613 | 0.323354328 | -0.002860285 |
| best | C0 | greedy_per | 0.249367473 | 0.249367473 | 0.000000000 |
| best | C0 | mismatched_audio_binary_ap | 0.290906350 | 0.287824830 | -0.003081520 |
| best | C0 | acoustic_reliance_gap | 0.021118910 | 0.021636882 | 0.000517973 |
| best | T1 | dedicated_omission_ap | 0.026813229 | 0.025631733 | -0.001181496 |
| best | T1 | legacy_omission_ap | 0.026813229 | 0.023895989 | -0.002917240 |
| best | T1 | strict_deployed_omission_precision | 0.015064935 | 0.031250000 | 0.016185065 |
| best | T1 | all_gold_omission_recall | 1.000000000 | 0.082758621 | -0.917241379 |
| best | T1 | false_alarms_per_1000_canonical | 952.189634391 | 37.364403375 | -914.825231016 |
| best | T1 | binary_ap | 0.313714249 | 0.308140671 | -0.005573578 |
| best | T1 | legacy_substitution_ap | 0.310913537 | 0.306309436 | -0.004604101 |
| best | T1 | dedicated_substitution_ap | 0.310913537 | 0.339992089 | 0.029078552 |
| best | T1 | greedy_per | 0.282360085 | 0.282360085 | 0.000000000 |
| best | T1 | mismatched_audio_binary_ap | 0.290147394 | 0.286642833 | -0.003504560 |
| best | T1 | acoustic_reliance_gap | 0.023566856 | 0.021497838 | -0.002069018 |
| best | T2 | dedicated_omission_ap | 0.096292659 | 0.086463470 | -0.009829190 |
| best | T2 | legacy_omission_ap | 0.035025084 | 0.031205329 | -0.003819755 |
| best | T2 | strict_deployed_omission_precision | 0.070949185 | 0.063870353 | -0.007078832 |
| best | T2 | all_gold_omission_recall | 0.510344828 | 0.462068966 | -0.048275862 |
| best | T2 | false_alarms_per_1000_canonical | 97.328244275 | 98.633989554 | 1.305745279 |
| best | T2 | binary_ap | 0.321573059 | 0.318158944 | -0.003414115 |
| best | T2 | legacy_substitution_ap | 0.312209457 | 0.310863253 | -0.001346204 |
| best | T2 | dedicated_substitution_ap | 0.337593421 | 0.335316426 | -0.002276995 |
| best | T2 | greedy_per | 0.239044631 | 0.238943427 | -0.000101204 |
| best | T2 | mismatched_audio_binary_ap | 0.299087404 | 0.297257792 | -0.001829612 |
| best | T2 | acoustic_reliance_gap | 0.022485654 | 0.020901152 | -0.001584503 |
| best | T3 | dedicated_omission_ap | 0.028354069 | 0.024483826 | -0.003870243 |
| best | T3 | legacy_omission_ap | 0.028354069 | 0.026971583 | -0.001382486 |
| best | T3 | strict_deployed_omission_precision | 0.014851992 | 0.023622047 | 0.008770055 |
| best | T3 | all_gold_omission_recall | 1.000000000 | 0.103448276 | -0.896551724 |
| best | T3 | false_alarms_per_1000_canonical | 966.050622740 | 62.274005625 | -903.776617115 |
| best | T3 | binary_ap | 0.285684293 | 0.281912278 | -0.003772015 |
| best | T3 | legacy_substitution_ap | 0.277751241 | 0.273950379 | -0.003800862 |
| best | T3 | dedicated_substitution_ap | 0.277751241 | 0.325517970 | 0.047766729 |
| best | T3 | greedy_per | 0.249468677 | 0.249468677 | 0.000000000 |
| best | T3 | mismatched_audio_binary_ap | 0.261268077 | 0.260463840 | -0.000804237 |
| best | T3 | acoustic_reliance_gap | 0.024416216 | 0.021448438 | -0.002967778 |

## Deployed count changes (CAL15)

Historical FN includes the invalid extra-eight correction; retained as a conflicting original claim, not validated accounting. Aligned-only recall had no valid original counterpart and is separately reported.

| Endpoint | Arm | Count | Original claimed | Corrected | Delta |
|---|---|---|---|---|---|
| fixed_budget_3600 | C0 | TP | 87 | 80 | -7 |
| fixed_budget_3600 | C0 | FP_correct | 1007 | 999 | -8 |
| fixed_budget_3600 | C0 | FP_sub | 59 | 59 | 0 |
| fixed_budget_3600 | C0 | FN | 66 | 65 | -1 |
| fixed_budget_3600 | T1 | TP | 145 | 17 | -128 |
| fixed_budget_3600 | T1 | FP_correct | 8799 | 591 | -8208 |
| fixed_budget_3600 | T1 | FP_sub | 1012 | 13 | -999 |
| fixed_budget_3600 | T1 | FN | 8 | 128 | 120 |
| fixed_budget_3600 | T2 | TP | 74 | 67 | -7 |
| fixed_budget_3600 | T2 | FP_correct | 914 | 922 | 8 |
| fixed_budget_3600 | T2 | FP_sub | 55 | 60 | 5 |
| fixed_budget_3600 | T2 | FN | 79 | 78 | -1 |
| fixed_budget_3600 | T3 | TP | 145 | 22 | -123 |
| fixed_budget_3600 | T3 | FP_correct | 8799 | 493 | -8306 |
| fixed_budget_3600 | T3 | FP_sub | 1012 | 22 | -990 |
| fixed_budget_3600 | T3 | FN | 8 | 123 | 115 |
| best | C0 | TP | 64 | 60 | -4 |
| best | C0 | FP_correct | 745 | 745 | 0 |
| best | C0 | FP_sub | 38 | 37 | -1 |
| best | C0 | FN | 89 | 85 | -4 |
| best | T1 | TP | 145 | 12 | -133 |
| best | T1 | FP_correct | 8479 | 362 | -8117 |
| best | T1 | FP_sub | 1001 | 10 | -991 |
| best | T1 | FN | 8 | 133 | 125 |
| best | T2 | TP | 74 | 67 | -7 |
| best | T2 | FP_correct | 914 | 922 | 8 |
| best | T2 | FP_sub | 55 | 60 | 5 |
| best | T2 | FN | 79 | 78 | -1 |
| best | T3 | TP | 145 | 15 | -130 |
| best | T3 | FP_correct | 8609 | 596 | -8013 |
| best | T3 | FP_sub | 1009 | 24 | -985 |
| best | T3 | FN | 8 | 130 | 122 |
