# Head-wise source-wise frozen score fusion

Target weights are selected independently on the 0.2 simplex by DEV_CAL mean AP(S,O); scalar insertion weights are selected independently by DEV_CAL insertion AP. Whole target probability vectors and insertion probabilities are convexly blended. No event-wise gating or DEV_SEARCH tuning.

Target weights: {'EXP-36': 0.8, 'EXP-43': 0.0, 'Gold-Only seed101': 0.2} (DEV_CAL mean AP(S,O)=0.570429)
Insertion weights: {'EXP-36': 0.0, 'EXP-43': 0.8, 'Gold-Only seed101': 0.2} (DEV_CAL insertion AP=0.293306)

| Model | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | FWR raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Head-wise selected | DEV_CAL | 0.5629 | 0.5780 | 0.2933 | 0.9511 | 0.5390 | 0.5853 | 0.4026 | 0.8504 | 0.4781 | 0.5090 | 0.5943 | 5.6000 |
| Head-wise selected | DEV_SEARCH | 0.3975 | 0.2053 | 0.2924 | 0.9802 | 0.4041 | 0.2968 | 0.2871 | 0.8761 | 0.2984 | 0.3293 | 0.4660 | 6.1152 |
| EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |
| Gold-Only seed101 | DEV_CAL | 0.5280 | 0.4789 | 0.1769 | 0.9415 | 0.5084 | 0.5468 | 0.3107 | 0.8545 | 0.3946 | 0.4553 | 0.5551 | 5.2900 |
| Gold-Only seed101 | DEV_SEARCH | 0.3646 | 0.1594 | 0.1867 | 0.9759 | 0.3866 | 0.2510 | 0.2628 | 0.8827 | 0.2369 | 0.3001 | 0.4458 | 5.7379 |
| Shared two-way alpha=.8 | DEV_CAL | 0.5629 | 0.5780 | 0.2828 | 0.9511 | 0.5390 | 0.5853 | 0.3766 | 0.8504 | 0.4745 | 0.5003 | 0.5878 | 5.6067 |
| Shared two-way alpha=.8 | DEV_SEARCH | 0.3975 | 0.2053 | 0.2860 | 0.9802 | 0.4041 | 0.2968 | 0.3148 | 0.8761 | 0.2962 | 0.3386 | 0.4730 | 6.1302 |
| Shared three-way | DEV_CAL | 0.5613 | 0.5750 | 0.2912 | 0.9516 | 0.5394 | 0.5793 | 0.4156 | 0.8508 | 0.4759 | 0.5115 | 0.5963 | 5.5900 |
| Shared three-way | DEV_SEARCH | 0.3902 | 0.2043 | 0.2920 | 0.9802 | 0.4041 | 0.2970 | 0.3039 | 0.8768 | 0.2955 | 0.3350 | 0.4704 | 6.0835 |

Exact target/gap event order verified across all three sources: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.
Runtime: 5.89 seconds.

Limitations: DEV_SEARCH is development-exposed, not a sealed confirmatory estimate. The coarse 0.2 simplex explores only 21 candidate weight vectors independently per head. DEV_CAL contains only two speakers; selection and threshold estimates may have high variance. No training or protected data used.
