# EXP-TSB-01: EXP-36 / Gold-Only convex score blend

Alpha is EXP-36 weight; student weight is 1-alpha. The predeclared grid was 0, 0.2, 0.4, 0.6, 0.8, 1.0. Selection used DEV_CAL Macro-3 Error AP = mean AP(SUBSTITUTION, OMISSION, INSERTION); tie-break: smallest alpha. Target score vectors were blended componentwise (simplex-preserving); insertion scores were blended as scalar probabilities.

Selected alpha: **0.8**. F1 policy: EVAL-V3 CAL_MAX_ERROR_MACRO_F1 fit on DEV_CAL only; thresholds: substitution 0.3562, omission 0.4286, insertion 0.5634.

| Model | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro-3 AP | Macro-3 F1 | Macro-4 F1 | FWR unfiltered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected blend | DEV_CAL | 0.5629 | 0.5780 | 0.2828 | 0.9511 | 0.5390 | 0.5853 | 0.3766 | 0.8504 | 0.4745 | 0.5003 | 0.5878 | 5.6067 |
| Selected blend | DEV_SEARCH | 0.3975 | 0.2053 | 0.2860 | 0.9802 | 0.4041 | 0.2968 | 0.3148 | 0.8761 | 0.2962 | 0.3386 | 0.4730 | 6.1302 |
| EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| Gold-Only seed101 | DEV_CAL | 0.5280 | 0.4789 | 0.1769 | 0.9415 | 0.5084 | 0.5468 | 0.3107 | 0.8545 | 0.3946 | 0.4553 | 0.5551 | 5.2900 |
| Gold-Only seed101 | DEV_SEARCH | 0.3646 | 0.1594 | 0.1867 | 0.9759 | 0.3866 | 0.2510 | 0.2628 | 0.8827 | 0.2369 | 0.3001 | 0.4458 | 5.7379 |

Alignment: exact teacher/student metadata sequence equality verified before combining. DEV_CAL target/gap events: 10043/10343; DEV_SEARCH: 19755/20354.

Caveat: DEV_SEARCH was exposed for the final evaluation; it is exploratory, not a sealed confirmatory estimate. No training or protected data were used.
