# Four-way head-wise frozen teacher-score blend

Weights are ordered EXP-24, EXP-36, EXP-43, Gold-Only seed101. Target vectors and scalar insertion probabilities use independently selected 0.2-simplex head weights.

Selected target weights: {'EXP-24': 0.0, 'EXP-36': 0.8, 'EXP-43': 0.0, 'Gold-Only seed101': 0.2}; gap weights: {'EXP-24': 0.0, 'EXP-36': 0.0, 'EXP-43': 0.8, 'Gold-Only seed101': 0.2}.
DEV_CAL-only CAL_MAX_ERROR_MACRO_F1 thresholds: {'substitution': 0.3562157017737627, 'omission': 0.4285639103571885, 'insertion': 0.5899047959676326}

| Model | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | Unfiltered FWR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected blend | DEV_CAL | 0.5629 | 0.5780 | 0.2933 | 0.9511 | 0.5390 | 0.5853 | 0.4026 | 0.8504 | 0.4781 | 0.5090 | 0.5943 | 5.6000 |
| Selected blend | DEV_SEARCH | 0.3975 | 0.2053 | 0.2924 | 0.9802 | 0.4041 | 0.2968 | 0.2871 | 0.8761 | 0.2984 | 0.3293 | 0.4660 | 6.1152 |
| EXP-24 | DEV_CAL | 0.5218 | 0.5158 | 0.2527 | 0.9418 | 0.5153 | 0.5596 | 0.3204 | 0.8546 | 0.4301 | 0.4651 | 0.5625 | 5.2200 |
| EXP-24 | DEV_SEARCH | 0.3577 | 0.2251 | 0.2476 | 0.9779 | 0.3988 | 0.2535 | 0.2653 | 0.8836 | 0.2768 | 0.3059 | 0.4503 | 5.7129 |
| EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |
| Gold-Only seed101 | DEV_CAL | 0.5280 | 0.4789 | 0.1769 | 0.9415 | 0.5084 | 0.5468 | 0.3107 | 0.8545 | 0.3946 | 0.4553 | 0.5551 | 5.2900 |
| Gold-Only seed101 | DEV_SEARCH | 0.3646 | 0.1594 | 0.1867 | 0.9759 | 0.3866 | 0.2510 | 0.2628 | 0.8827 | 0.2369 | 0.3001 | 0.4458 | 5.7379 |

Alignment: target/gap counts DEV_CAL 10043/10343; DEV_SEARCH 19755/20354. Exact IDs and order verified across all four sources before blend.

Runtime: 5.20 seconds.

Limitations: DEV_SEARCH is development-exposed, not a sealed confirmatory estimate. Each head explores 56 candidates on the coarse 0.2 simplex. DEV_CAL contains only two speakers; selection and threshold estimates may have high variance. No training or protected data used.
