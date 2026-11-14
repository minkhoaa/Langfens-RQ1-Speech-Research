# Three-way frozen teacher-score blend

Weights are ordered EXP-36, EXP-43, Gold-Only seed101. Every target probability vector and scalar insertion probability is combined convexly. Candidate weights are the 0.2 simplex; selection is DEV_CAL Macro-3 Error AP only.

Selected weights: {'EXP-36': 0.2, 'EXP-43': 0.6, 'Gold-Only seed101': 0.2}; DEV_CAL Macro3 AP 0.475869.
DEV_CAL-only CAL_MAX_ERROR_MACRO_F1 thresholds: {'substitution': 0.38155930142104627, 'omission': 0.3770177822254627, 'insertion': 0.5853998523547497}

| Model | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | Unfiltered FWR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected blend | DEV_CAL | 0.5613 | 0.5750 | 0.2912 | 0.9516 | 0.5394 | 0.5793 | 0.4156 | 0.8508 | 0.4759 | 0.5115 | 0.5963 | 5.5900 |
| Selected blend | DEV_SEARCH | 0.3902 | 0.2043 | 0.2920 | 0.9802 | 0.4041 | 0.2970 | 0.3039 | 0.8768 | 0.2955 | 0.3350 | 0.4704 | 6.0835 |
| EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |
| Gold-Only seed101 | DEV_CAL | 0.5280 | 0.4789 | 0.1769 | 0.9415 | 0.5084 | 0.5468 | 0.3107 | 0.8545 | 0.3946 | 0.4553 | 0.5551 | 5.2900 |
| Gold-Only seed101 | DEV_SEARCH | 0.3646 | 0.1594 | 0.1867 | 0.9759 | 0.3866 | 0.2510 | 0.2628 | 0.8827 | 0.2369 | 0.3001 | 0.4458 | 5.7379 |

Alignment: target/gap counts DEV_CAL 10043/10343; DEV_SEARCH 19755/20354. Exact IDs and order verified across all three sources before blend.

Runtime: 4.16 seconds.

Limitations: DEV_SEARCH is development-exposed, not a sealed confirmatory estimate. The coarse 0.2 simplex explores only 21 candidate weight vectors. DEV_CAL contains only two speakers; selection and threshold estimates may have high variance. No training or protected data used.
