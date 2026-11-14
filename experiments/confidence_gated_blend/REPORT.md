# Confidence-gated frozen EXP-36 / Gold-Only blend

No training. Event-level teacher alpha=sigmoid(logit(.8)+b*(H_student-H_EXP36)/log(K)); K=3 target and K=2 binary gap; convex blend of full event distributions.

Selected b=0.0; DEV_CAL candidates (b, Macro3 Error AP): 0.0=0.474540, 0.5=0.474003, 1.0=0.471674, 2.0=0.470041.
Calibration policy: canonical CAL_MAX_ERROR_MACRO_F1 thresholds fitted separately for each system on DEV_CAL; selected gate evaluated once on DEV_SEARCH.

| Model | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | Unfiltered FWR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Confidence-gated selected | DEV_CAL | 0.5629 | 0.5780 | 0.2828 | 0.9511 | 0.5390 | 0.5853 | 0.3766 | 0.8504 | 0.4745 | 0.5003 | 0.5878 | 5.6067 |
| Confidence-gated selected | DEV_SEARCH | 0.3975 | 0.2053 | 0.2860 | 0.9802 | 0.4041 | 0.2968 | 0.3148 | 0.8761 | 0.2962 | 0.3386 | 0.4730 | 6.1302 |
| Constant alpha=.8 | DEV_CAL | 0.5629 | 0.5780 | 0.2828 | 0.9511 | 0.5390 | 0.5853 | 0.3766 | 0.8504 | 0.4745 | 0.5003 | 0.5878 | 5.6067 |
| Constant alpha=.8 | DEV_SEARCH | 0.3975 | 0.2053 | 0.2860 | 0.9802 | 0.4041 | 0.2968 | 0.3148 | 0.8761 | 0.2962 | 0.3386 | 0.4730 | 6.1302 |
| EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| Gold-Only seed101 | DEV_CAL | 0.5280 | 0.4789 | 0.1769 | 0.9415 | 0.5084 | 0.5468 | 0.3107 | 0.8545 | 0.3946 | 0.4553 | 0.5551 | 5.2900 |
| Gold-Only seed101 | DEV_SEARCH | 0.3646 | 0.1594 | 0.1867 | 0.9759 | 0.3866 | 0.2510 | 0.2628 | 0.8827 | 0.2369 | 0.3001 | 0.4458 | 5.7379 |

Exact event-order checks passed for target and gap IDs across EXP-36 and Gold-Only: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.
Runtime: 3.97s on CPU.

Limitations: DEV_SEARCH is development-exposed and not a sealed confirmatory estimate. DEV_CAL contains only two speakers; gate and threshold selection may have high variance. No training, GPU, or protected data used.
