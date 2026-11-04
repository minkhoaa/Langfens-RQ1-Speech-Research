# Speaker and per-phone analysis

NJS is Spanish L1/female; RRBI Hindi L1/male. Two speakers are not a Vietnamese-generalization cohort. CAL thresholds are global and unchanged for these strata. Speaker differences are a descriptive heterogeneity/uncertainty summary, not a confidence interval or independent-seed replication.

| Arm | Speaker | Omit AP | Binary AP | Precision | Recall all | TP | FP_correct | FP_sub | PER |
|---|---|---|---|---|---|---|---|---|---|
| C0 | NJS | 0.103100560 | 0.396354215 | 0.079335793 | 0.537500000 | 43 | 471 | 28 | 0.236539625 |
| C0 | RRBI | 0.116921580 | 0.250192507 | 0.062080537 | 0.569230769 | 37 | 528 | 31 | 0.241365299 |
| T1 | NJS | 0.049245122 | 0.397776830 | 0.038732394 | 0.137500000 | 11 | 262 | 11 | 0.236337971 |
| T1 | RRBI | 0.021994022 | 0.244085517 | 0.017804154 | 0.092307692 | 6 | 329 | 2 | 0.241771638 |
| T2 | NJS | 0.117211820 | 0.407244049 | 0.073839662 | 0.437500000 | 35 | 411 | 28 | 0.236539625 |
| T2 | RRBI | 0.071281483 | 0.251594727 | 0.055652174 | 0.492307692 | 32 | 511 | 32 | 0.241365299 |
| T3 | NJS | 0.041964774 | 0.396368550 | 0.061538462 | 0.200000000 | 16 | 228 | 16 | 0.236337971 |
| T3 | RRBI | 0.024331712 | 0.245847672 | 0.021660650 | 0.092307692 | 6 | 265 | 6 | 0.241771638 |

## Per-phone strict omission error analysis

AP population is correct+omission only; deployed FP_sub separately retained. Positive counts are scored omissions. Small support is flagged; no phone-level significance. Full per-phone legacy/dedicated/binary metrics are in private metrics JSON.

| Arm | Gold phone | Full omission cohort | Scored omissions | Omit AP | TP | FP_correct | FP_sub | Support |
|---|---|---|---|---|---|---|---|---|
| C0 | AA | 91 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | AE | 263 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | AH | 803 | 6 | 0.024513418 | 0 | 0 | 0 | sparse |
| C0 | AO | 142 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | AW | 53 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | AY | 185 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | B | 166 | 1 | 0.058823529 | 0 | 0 | 0 | sparse |
| C0 | CH | 67 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | D | 454 | 52 | 0.179452284 | 52 | 400 | 14 | descriptive |
| C0 | DH | 163 | 1 | 0.015384615 | 0 | 0 | 0 | sparse |
| C0 | EH | 244 | 3 | 0.081154401 | 0 | 0 | 0 | sparse |
| C0 | ER | 217 | 1 | 0.333333333 | 1 | 14 | 1 | sparse |
| C0 | EY | 141 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | F | 212 | 1 | 0.028571429 | 0 | 0 | 0 | sparse |
| C0 | G | 94 | 1 | 0.015151515 | 0 | 4 | 0 | sparse |
| C0 | HH | 289 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | IH | 582 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | IY | 396 | 1 | 0.002932551 | 0 | 0 | 0 | sparse |
| C0 | JH | 37 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | K | 261 | 4 | 0.044119136 | 0 | 3 | 0 | sparse |
| C0 | L | 395 | 2 | 0.017458280 | 0 | 32 | 0 | sparse |
| C0 | M | 287 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | N | 600 | 0 | NA | 0 | 2 | 0 | sparse |
| C0 | NG | 100 | 1 | 0.020833333 | 0 | 0 | 0 | sparse |
| C0 | OW | 88 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | OY | 18 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | P | 174 | 3 | 0.036617563 | 0 | 1 | 0 | sparse |
| C0 | R | 389 | 21 | 0.056601611 | 14 | 310 | 30 | descriptive |
| C0 | S | 386 | 0 | NA | 0 | 1 | 0 | sparse |
| C0 | SH | 112 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | T | 585 | 29 | 0.113251649 | 13 | 215 | 5 | descriptive |
| C0 | TH | 50 | 1 | 0.035714286 | 0 | 0 | 0 | sparse |
| C0 | UH | 55 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | UW | 170 | 0 | NA | 0 | 0 | 0 | sparse |
| C0 | V | 169 | 4 | 0.018270322 | 0 | 2 | 1 | sparse |
| C0 | W | 235 | 0 | NA | 0 | 3 | 0 | sparse |
| C0 | Y | 86 | 2 | 0.040151515 | 0 | 1 | 0 | sparse |
| C0 | Z | 176 | 3 | 0.037896433 | 0 | 11 | 8 | sparse |
| C0 | ZH | 9 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | AA | 91 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | AE | 263 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | AH | 803 | 6 | 0.010852941 | 0 | 1 | 0 | sparse |
| T1 | AO | 142 | 0 | NA | 0 | 1 | 0 | sparse |
| T1 | AW | 53 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | AY | 185 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | B | 166 | 1 | 0.250000000 | 0 | 0 | 0 | sparse |
| T1 | CH | 67 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | D | 454 | 52 | 0.145542130 | 14 | 96 | 2 | descriptive |
| T1 | DH | 163 | 1 | 0.008474576 | 0 | 0 | 0 | sparse |
| T1 | EH | 244 | 3 | 0.070374127 | 0 | 0 | 0 | sparse |
| T1 | ER | 217 | 1 | 1.000000000 | 0 | 0 | 0 | sparse |
| T1 | EY | 141 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | F | 212 | 1 | 0.004739336 | 0 | 0 | 0 | sparse |
| T1 | G | 94 | 1 | 0.012820513 | 0 | 13 | 0 | sparse |
| T1 | HH | 289 | 0 | NA | 0 | 128 | 6 | sparse |
| T1 | IH | 582 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | IY | 396 | 1 | 0.003194888 | 0 | 0 | 0 | sparse |
| T1 | JH | 37 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | K | 261 | 4 | 0.019845114 | 0 | 24 | 0 | sparse |
| T1 | L | 395 | 2 | 0.169913420 | 1 | 174 | 1 | sparse |
| T1 | M | 287 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | N | 600 | 0 | NA | 0 | 55 | 0 | sparse |
| T1 | NG | 100 | 1 | 0.025641026 | 0 | 0 | 0 | sparse |
| T1 | OW | 88 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | OY | 18 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | P | 174 | 3 | 0.015734990 | 0 | 3 | 0 | sparse |
| T1 | R | 389 | 21 | 0.050005116 | 0 | 0 | 0 | descriptive |
| T1 | S | 386 | 0 | NA | 0 | 19 | 0 | sparse |
| T1 | SH | 112 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | T | 585 | 29 | 0.060001377 | 0 | 1 | 0 | descriptive |
| T1 | TH | 50 | 1 | 0.035714286 | 0 | 0 | 0 | sparse |
| T1 | UH | 55 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | UW | 170 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | V | 169 | 4 | 0.031292292 | 0 | 1 | 0 | sparse |
| T1 | W | 235 | 0 | NA | 0 | 0 | 0 | sparse |
| T1 | Y | 86 | 2 | 0.125000000 | 2 | 68 | 0 | sparse |
| T1 | Z | 176 | 3 | 0.034329522 | 0 | 7 | 4 | sparse |
| T1 | ZH | 9 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | AA | 91 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | AE | 263 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | AH | 803 | 6 | 0.042214739 | 0 | 0 | 0 | sparse |
| T2 | AO | 142 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | AW | 53 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | AY | 185 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | B | 166 | 1 | 0.142857143 | 0 | 0 | 0 | sparse |
| T2 | CH | 67 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | D | 454 | 52 | 0.159358059 | 43 | 341 | 12 | descriptive |
| T2 | DH | 163 | 1 | 0.007246377 | 0 | 0 | 0 | sparse |
| T2 | EH | 244 | 3 | 0.035136897 | 0 | 0 | 0 | sparse |
| T2 | ER | 217 | 1 | 0.200000000 | 1 | 12 | 1 | sparse |
| T2 | EY | 141 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | F | 212 | 1 | 0.005780347 | 0 | 0 | 0 | sparse |
| T2 | G | 94 | 1 | 0.020833333 | 0 | 6 | 1 | sparse |
| T2 | HH | 289 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | IH | 582 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | IY | 396 | 1 | 0.004065041 | 0 | 0 | 0 | sparse |
| T2 | JH | 37 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | K | 261 | 4 | 0.018354823 | 0 | 9 | 0 | sparse |
| T2 | L | 395 | 2 | 0.020816044 | 0 | 16 | 0 | sparse |
| T2 | M | 287 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | N | 600 | 0 | NA | 0 | 1 | 0 | sparse |
| T2 | NG | 100 | 1 | 0.014705882 | 0 | 0 | 0 | sparse |
| T2 | OW | 88 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | OY | 18 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | P | 174 | 3 | 0.031354623 | 0 | 2 | 0 | sparse |
| T2 | R | 389 | 21 | 0.090416522 | 13 | 262 | 24 | descriptive |
| T2 | S | 386 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | SH | 112 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | T | 585 | 29 | 0.062701555 | 10 | 251 | 5 | descriptive |
| T2 | TH | 50 | 1 | 0.066666667 | 0 | 0 | 0 | sparse |
| T2 | UH | 55 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | UW | 170 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | V | 169 | 4 | 0.024017095 | 0 | 1 | 0 | sparse |
| T2 | W | 235 | 0 | NA | 0 | 0 | 0 | sparse |
| T2 | Y | 86 | 2 | 0.028283228 | 0 | 2 | 0 | sparse |
| T2 | Z | 176 | 3 | 0.035002144 | 0 | 19 | 17 | sparse |
| T2 | ZH | 9 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | AA | 91 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | AE | 263 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | AH | 803 | 6 | 0.041676255 | 0 | 4 | 0 | sparse |
| T3 | AO | 142 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | AW | 53 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | AY | 185 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | B | 166 | 1 | 0.043478261 | 0 | 0 | 0 | sparse |
| T3 | CH | 67 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | D | 454 | 52 | 0.187467164 | 17 | 66 | 1 | descriptive |
| T3 | DH | 163 | 1 | 0.010000000 | 0 | 0 | 0 | sparse |
| T3 | EH | 244 | 3 | 0.085907859 | 0 | 0 | 0 | sparse |
| T3 | ER | 217 | 1 | 0.250000000 | 0 | 0 | 0 | sparse |
| T3 | EY | 141 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | F | 212 | 1 | 0.006711409 | 0 | 0 | 0 | sparse |
| T3 | G | 94 | 1 | 0.043478261 | 0 | 16 | 0 | sparse |
| T3 | HH | 289 | 0 | NA | 0 | 102 | 5 | sparse |
| T3 | IH | 582 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | IY | 396 | 1 | 0.003048780 | 0 | 0 | 0 | sparse |
| T3 | JH | 37 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | K | 261 | 4 | 0.027773740 | 0 | 36 | 2 | sparse |
| T3 | L | 395 | 2 | 0.023989899 | 2 | 128 | 1 | sparse |
| T3 | M | 287 | 0 | NA | 0 | 1 | 0 | sparse |
| T3 | N | 600 | 0 | NA | 0 | 55 | 2 | sparse |
| T3 | NG | 100 | 1 | 0.027777778 | 0 | 3 | 0 | sparse |
| T3 | OW | 88 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | OY | 18 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | P | 174 | 3 | 0.022810337 | 0 | 3 | 0 | sparse |
| T3 | R | 389 | 21 | 0.066398773 | 1 | 6 | 0 | descriptive |
| T3 | S | 386 | 0 | NA | 0 | 15 | 0 | sparse |
| T3 | SH | 112 | 0 | NA | 0 | 1 | 0 | sparse |
| T3 | T | 585 | 29 | 0.101397085 | 1 | 4 | 0 | descriptive |
| T3 | TH | 50 | 1 | 0.052631579 | 0 | 0 | 0 | sparse |
| T3 | UH | 55 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | UW | 170 | 0 | NA | 0 | 0 | 0 | sparse |
| T3 | V | 169 | 4 | 0.018649134 | 0 | 1 | 0 | sparse |
| T3 | W | 235 | 0 | NA | 0 | 15 | 1 | sparse |
| T3 | Y | 86 | 2 | 0.048766157 | 1 | 25 | 0 | sparse |
| T3 | Z | 176 | 3 | 0.048994253 | 0 | 12 | 10 | sparse |
| T3 | ZH | 9 | 0 | NA | 0 | 0 | 0 | sparse |
