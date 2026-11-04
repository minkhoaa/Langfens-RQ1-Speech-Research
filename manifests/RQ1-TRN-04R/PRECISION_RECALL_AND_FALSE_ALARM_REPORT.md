# CAL-selected precision, recall and false alarms

Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.

CAL5/CAL10 are secondary predeclared operating points, not VAL-selected substitutes for CAL15. None reaches 25% precision at these three points. Abstentions are no-alarm; the eight unmapped omissions remain false negatives. No claim is made about every conceivable threshold.

| Arm | CAL ceiling | Threshold | CAL achieved FPR | TP | FP_correct | FP_sub | FN all | Precision | Recall all | Recall aligned | FA/1,000 | Extra-prompt alarms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.05 | 0.136839062 | 0.049900448 | 27 | 170 | 6 | 118 | 0.133004926 | 0.186206897 | 0.197080292 | 17.677782242 | 1 |
| C0 | 0.1 | 0.092122018 | 0.099925336 | 50 | 494 | 26 | 95 | 0.087719298 | 0.344827586 | 0.364963504 | 52.229811169 | 3 |
| C0 | 0.15 | 0.058004841 | 0.149950224 | 80 | 999 | 59 | 65 | 0.070298770 | 0.551724138 | 0.583941606 | 106.267577340 | 9 |
| T1 | 0.05 | 0.139050454 | 0.049900448 | 5 | 79 | 2 | 140 | 0.058139535 | 0.034482759 | 0.036496350 | 8.135797509 | 1 |
| T1 | 0.1 | 0.122929230 | 0.099925336 | 11 | 289 | 5 | 134 | 0.036065574 | 0.075862069 | 0.080291971 | 29.529931699 | 1 |
| T1 | 0.15 | 0.105885923 | 0.149950224 | 17 | 591 | 13 | 128 | 0.027375201 | 0.117241379 | 0.124087591 | 60.666934512 | 2 |
| T2 | 0.05 | 0.132375732 | 0.049900448 | 18 | 138 | 6 | 127 | 0.111111111 | 0.124137931 | 0.131386861 | 14.463640016 | 0 |
| T2 | 0.1 | 0.092629269 | 0.099925336 | 43 | 455 | 26 | 102 | 0.082061069 | 0.296551724 | 0.313868613 | 48.312575331 | 4 |
| T2 | 0.15 | 0.059713610 | 0.149950224 | 67 | 922 | 60 | 78 | 0.063870353 | 0.462068966 | 0.489051095 | 98.633989554 | 8 |
| T3 | 0.05 | 0.140241504 | 0.049900448 | 2 | 104 | 9 | 143 | 0.017391304 | 0.013793103 | 0.014598540 | 11.349939735 | 0 |
| T3 | 0.1 | 0.122012898 | 0.099925336 | 8 | 250 | 16 | 137 | 0.029197080 | 0.055172414 | 0.058394161 | 26.717557252 | 1 |
| T3 | 0.15 | 0.104364879 | 0.149950224 | 22 | 493 | 22 | 123 | 0.040968343 | 0.151724138 | 0.160583942 | 51.727601446 | 1 |
