# DEV_CAL risk-budget operating points: EXP-36 / Gold-Only blends

Fixed alpha=0.8 (EXP-36 score weight; Gold-Only weight=0.2), evaluated for student seeds 101, 202, and 303. Every seed/budget threshold set was selected on DEV_CAL using canonical `select_cal_policies`; each selected policy was evaluated on DEV_SEARCH once. This is post-discrimination operating-point analysis. Candidates are **not** filtered for DEV_SEARCH observed FWR > 0.20. AP uses continuous blended scores and is separated from threshold-dependent operating metrics.

## Selection configuration

- Selector: `research.evaluation_v3.policies.select_cal_policies`.
- Budgets per seed: `[0.2, 0.5, 1.0, 2.0, None]`. Numeric budgets constrain DEV_CAL false warnings per utterance; `None` is unconstrained maximum Macro3 error F1.
- Runtime (excluding process startup): 8.79 seconds.
- Threshold fitting used only DEV_CAL labels and score arrays. No training or protected data were used.
- Exact target/gap metadata array equality (including order) was verified between EXP-36 and corresponding student arrays for DEV_CAL and DEV_SEARCH.
- DEV_SEARCH is exploratory, not sealed confirmatory data.

## Seed 101

Alignment: DEV_CAL 10043 targets / 10343 gaps; DEV_SEARCH 19755 targets / 20354 gaps; exact order matched.

### Budget 0.2

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.7283813202619552, 'omission': 0.7472028443217277, 'insertion': 0.5634234367901024}`. DEV_CAL Macro3 error F1: 0.2280; DEV_CAL false warnings: 58 (FWR 0.1933). DEV_SEARCH observed FWR: **0.1336** (80 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9015 | 0.9988 | 0.9477 | 17778 |
| SUBSTITUTION | 0.3975 | 0.7143 | 0.0170 | 0.0331 | 1768 |
| OMISSION | 0.2053 | 0.3889 | 0.0335 | 0.0617 | 209 |
| INSERTION | 0.2860 | 0.3736 | 0.2720 | 0.3148 | 125 |

Macro3 error F1: **0.1365**; Macro4 F1: **0.3393**.

### Budget 0.5

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.6378903304457665, 'omission': 0.6957351277112961, 'insertion': 0.5634234367901024}`. DEV_CAL Macro3 error F1: 0.3415; DEV_CAL false warnings: 149 (FWR 0.4967). DEV_SEARCH observed FWR: **0.4140** (248 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9097 | 0.9898 | 0.9481 | 17778 |
| SUBSTITUTION | 0.3975 | 0.6060 | 0.1148 | 0.1931 | 1768 |
| OMISSION | 0.2053 | 0.2436 | 0.0909 | 0.1324 | 209 |
| INSERTION | 0.2860 | 0.3736 | 0.2720 | 0.3148 | 125 |

Macro3 error F1: **0.2134**; Macro4 F1: **0.3971**.

### Budget 1.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5811540599912406, 'omission': 0.6470559991836546, 'insertion': 0.5634234367901024}`. DEV_CAL Macro3 error F1: 0.4242; DEV_CAL false warnings: 300 (FWR 1.0000). DEV_SEARCH observed FWR: **0.9783** (586 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9216 | 0.9714 | 0.9459 | 17778 |
| SUBSTITUTION | 0.3975 | 0.5220 | 0.2551 | 0.3427 | 1768 |
| OMISSION | 0.2053 | 0.2368 | 0.1722 | 0.1994 | 209 |
| INSERTION | 0.2860 | 0.3736 | 0.2720 | 0.3148 | 125 |

Macro3 error F1: **0.2857**; Macro4 F1: **0.4507**.

### Budget 2.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5298651579208673, 'omission': 0.6396784261614086, 'insertion': 0.5634234367901024}`. DEV_CAL Macro3 error F1: 0.4539; DEV_CAL false warnings: 447 (FWR 1.4900). DEV_SEARCH observed FWR: **1.6861** (1010 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9331 | 0.9479 | 0.9404 | 17778 |
| SUBSTITUTION | 0.3975 | 0.4602 | 0.3988 | 0.4273 | 1768 |
| OMISSION | 0.2053 | 0.2317 | 0.1818 | 0.2038 | 209 |
| INSERTION | 0.2860 | 0.3736 | 0.2720 | 0.3148 | 125 |

Macro3 error F1: **0.3153**; Macro4 F1: **0.4716**.

### Budget None (unconstrained)

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.3562157017737627, 'omission': 0.4285639103571885, 'insertion': 0.5634234367901024}`. DEV_CAL Macro3 error F1: 0.5003; DEV_CAL false warnings: 1682 (FWR 5.6067). DEV_SEARCH observed FWR: **6.1302** (3672 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9651 | 0.8021 | 0.8761 | 17778 |
| SUBSTITUTION | 0.3975 | 0.2830 | 0.7064 | 0.4041 | 1768 |
| OMISSION | 0.2053 | 0.2032 | 0.5502 | 0.2968 | 209 |
| INSERTION | 0.2860 | 0.3736 | 0.2720 | 0.3148 | 125 |

Macro3 error F1: **0.3386**; Macro4 F1: **0.4730**.

## Seed 202

Alignment: DEV_CAL 10043 targets / 10343 gaps; DEV_SEARCH 19755 targets / 20354 gaps; exact order matched.

### Budget 0.2

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.7293003612041471, 'omission': 0.7281166540980338, 'insertion': 0.5749771890875394}`. DEV_CAL Macro3 error F1: 0.2276; DEV_CAL false warnings: 55 (FWR 0.1833). DEV_SEARCH observed FWR: **0.1269** (76 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9018 | 0.9986 | 0.9477 | 17778 |
| SUBSTITUTION | 0.3990 | 0.7111 | 0.0181 | 0.0353 | 1768 |
| OMISSION | 0.2004 | 0.4348 | 0.0478 | 0.0862 | 209 |
| INSERTION | 0.2843 | 0.3976 | 0.2640 | 0.3173 | 125 |

Macro3 error F1: **0.1463**; Macro4 F1: **0.3466**.

### Budget 0.5

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.6982650671005249, 'omission': 0.6300931574016808, 'insertion': 0.5749771890875394}`. DEV_CAL Macro3 error F1: 0.3208; DEV_CAL false warnings: 149 (FWR 0.4967). DEV_SEARCH observed FWR: **0.3339** (200 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9044 | 0.9927 | 0.9465 | 17778 |
| SUBSTITUTION | 0.3990 | 0.6552 | 0.0322 | 0.0615 | 1768 |
| OMISSION | 0.2004 | 0.2258 | 0.1675 | 0.1923 | 209 |
| INSERTION | 0.2843 | 0.3976 | 0.2640 | 0.3173 | 125 |

Macro3 error F1: **0.1904**; Macro4 F1: **0.3794**.

### Budget 1.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5797723041474819, 'omission': 0.6218713328242302, 'insertion': 0.7531012885498448}`. DEV_CAL Macro3 error F1: 0.3857; DEV_CAL false warnings: 298 (FWR 0.9933). DEV_SEARCH observed FWR: **0.9399** (563 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9218 | 0.9710 | 0.9458 | 17778 |
| SUBSTITUTION | 0.3990 | 0.5239 | 0.2545 | 0.3426 | 1768 |
| OMISSION | 0.2004 | 0.2235 | 0.1818 | 0.2005 | 209 |
| INSERTION | 0.2843 | 0.5417 | 0.2080 | 0.3006 | 125 |

Macro3 error F1: **0.2812**; Macro4 F1: **0.4474**.

### Budget 2.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5315074508823455, 'omission': 0.6218713328242302, 'insertion': 0.34005081238300816}`. DEV_CAL Macro3 error F1: 0.4471; DEV_CAL false warnings: 500 (FWR 1.6667). DEV_SEARCH observed FWR: **1.7513** (1049 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9330 | 0.9499 | 0.9414 | 17778 |
| SUBSTITUTION | 0.3990 | 0.4707 | 0.3954 | 0.4298 | 1768 |
| OMISSION | 0.2004 | 0.2235 | 0.1818 | 0.2005 | 209 |
| INSERTION | 0.2843 | 0.2640 | 0.3760 | 0.3102 | 125 |

Macro3 error F1: **0.3135**; Macro4 F1: **0.4705**.

### Budget None (unconstrained)

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.3540217557549477, 'omission': 0.4106950159184635, 'insertion': 0.34005081238300816}`. DEV_CAL Macro3 error F1: 0.4982; DEV_CAL false warnings: 1719 (FWR 5.7300). DEV_SEARCH observed FWR: **6.2053** (3717 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9645 | 0.8037 | 0.8768 | 17778 |
| SUBSTITUTION | 0.3990 | 0.2841 | 0.7031 | 0.4047 | 1768 |
| OMISSION | 0.2004 | 0.1979 | 0.5359 | 0.2890 | 209 |
| INSERTION | 0.2843 | 0.2640 | 0.3760 | 0.3102 | 125 |

Macro3 error F1: **0.3347**; Macro4 F1: **0.4702**.

## Seed 303

Alignment: DEV_CAL 10043 targets / 10343 gaps; DEV_SEARCH 19755 targets / 20354 gaps; exact order matched.

### Budget 0.2

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.7226396990776036, 'omission': 0.7478703432559968, 'insertion': 0.579971098665973}`. DEV_CAL Macro3 error F1: 0.2288; DEV_CAL false warnings: 58 (FWR 0.1933). DEV_SEARCH observed FWR: **0.1285** (77 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9016 | 0.9985 | 0.9476 | 17778 |
| SUBSTITUTION | 0.3959 | 0.6739 | 0.0175 | 0.0342 | 1768 |
| OMISSION | 0.2025 | 0.4000 | 0.0383 | 0.0699 | 209 |
| INSERTION | 0.2812 | 0.4048 | 0.2720 | 0.3254 | 125 |

Macro3 error F1: **0.1431**; Macro4 F1: **0.3443**.

### Budget 0.5

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.6939367629766464, 'omission': 0.6386251723766327, 'insertion': 0.579971098665973}`. DEV_CAL Macro3 error F1: 0.3306; DEV_CAL false warnings: 150 (FWR 0.5000). DEV_SEARCH observed FWR: **0.3523** (211 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9044 | 0.9921 | 0.9462 | 17778 |
| SUBSTITUTION | 0.3959 | 0.6296 | 0.0288 | 0.0552 | 1768 |
| OMISSION | 0.2025 | 0.2384 | 0.1962 | 0.2152 | 209 |
| INSERTION | 0.2812 | 0.4048 | 0.2720 | 0.3254 | 125 |

Macro3 error F1: **0.1986**; Macro4 F1: **0.3855**.

### Budget 1.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5745821953937411, 'omission': 0.6386251723766327, 'insertion': 0.7590328189509974}`. DEV_CAL Macro3 error F1: 0.3949; DEV_CAL false warnings: 291 (FWR 0.9700). DEV_SEARCH observed FWR: **0.9316** (558 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9214 | 0.9711 | 0.9456 | 17778 |
| SUBSTITUTION | 0.3959 | 0.5189 | 0.2483 | 0.3359 | 1768 |
| OMISSION | 0.2025 | 0.2384 | 0.1962 | 0.2152 | 209 |
| INSERTION | 0.2812 | 0.5652 | 0.2080 | 0.3041 | 125 |

Macro3 error F1: **0.2851**; Macro4 F1: **0.4502**.

### Budget 2.0

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.5227275414206088, 'omission': 0.6386251723766327, 'insertion': 0.579971098665973}`. DEV_CAL Macro3 error F1: 0.4553; DEV_CAL false warnings: 444 (FWR 1.4800). DEV_SEARCH observed FWR: **1.6912** (1013 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9333 | 0.9474 | 0.9403 | 17778 |
| SUBSTITUTION | 0.3959 | 0.4590 | 0.3993 | 0.4271 | 1768 |
| OMISSION | 0.2025 | 0.2384 | 0.1962 | 0.2152 | 209 |
| INSERTION | 0.2812 | 0.4048 | 0.2720 | 0.3254 | 125 |

Macro3 error F1: **0.3226**; Macro4 F1: **0.4770**.

### Budget None (unconstrained)

Thresholds (SUBSTITUTION / OMISSION / INSERTION): `{'substitution': 0.372175394948572, 'omission': 0.4251449859666173, 'insertion': 0.579971098665973}`. DEV_CAL Macro3 error F1: 0.5012; DEV_CAL false warnings: 1455 (FWR 4.8500). DEV_SEARCH observed FWR: **5.3472** (3203 false warnings / 599 utterances).

| Class | Raw AP | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| CORRECT | 0.9802 | 0.9626 | 0.8279 | 0.8902 | 17778 |
| SUBSTITUTION | 0.3959 | 0.3077 | 0.6770 | 0.4231 | 1768 |
| OMISSION | 0.2025 | 0.1986 | 0.5455 | 0.2912 | 209 |
| INSERTION | 0.2812 | 0.4048 | 0.2720 | 0.3254 | 125 |

Macro3 error F1: **0.3466**; Macro4 F1: **0.4825**.

## Interpretation boundary

AP is threshold-invariant and repeated per operating point only for convenient reference; changes in precision/recall/F1/FWR are due to DEV_CAL-selected thresholds. Budget labels constrain calibration-set false-warning risk, not held-out risk. Observed DEV_SEARCH FWR values above 0.20 are retained and reported; none are removed.

Results and complete selector metadata are in `results.json`.
