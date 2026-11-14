# Student-seed replication of the frozen head-wise blend

Fixed weights, with no retuning: target probabilities = 0.8 EXP-36 + 0.2 matched Gold-Only seed; exact-gap insertion probability = 0.8 EXP-43 + 0.2 matched Gold-Only seed. CAL_MAX_ERROR_MACRO_F1 thresholds were fitted separately per seed and system on DEV_CAL. DEV_SEARCH was evaluated once.

| Seed | System | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | FWR unfiltered |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 202 | Head-wise blend | DEV_CAL | 0.5641 | 0.5728 | 0.2943 | 0.9510 | 0.5404 | 0.5853 | 0.4026 | 0.8518 | 0.4771 | 0.5094 | 0.5950 | 5.5800 |
| 202 | Head-wise blend | DEV_SEARCH | 0.3990 | 0.2004 | 0.2898 | 0.9802 | 0.4047 | 0.2890 | 0.2886 | 0.8768 | 0.2964 | 0.3274 | 0.4648 | 6.0651 |
| 202 | Gold-Only | DEV_CAL | 0.5268 | 0.4628 | 0.2016 | 0.9405 | 0.5166 | 0.5303 | 0.3117 | 0.8685 | 0.3971 | 0.4528 | 0.5568 | 4.3233 |
| 202 | Gold-Only | DEV_SEARCH | 0.3708 | 0.1462 | 0.1866 | 0.9753 | 0.4052 | 0.2497 | 0.2640 | 0.8998 | 0.2345 | 0.3063 | 0.4547 | 4.6177 |
| 202 | EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| 202 | EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| 202 | EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| 202 | EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |
| 303 | Head-wise blend | DEV_CAL | 0.5636 | 0.5816 | 0.2994 | 0.9510 | 0.5379 | 0.5890 | 0.4156 | 0.8621 | 0.4815 | 0.5141 | 0.6011 | 4.8400 |
| 303 | Head-wise blend | DEV_SEARCH | 0.3959 | 0.2025 | 0.2868 | 0.9802 | 0.4231 | 0.2912 | 0.2886 | 0.8902 | 0.2951 | 0.3343 | 0.4733 | 5.3422 |
| 303 | Gold-Only | DEV_CAL | 0.5296 | 0.5044 | 0.2020 | 0.9406 | 0.5016 | 0.5376 | 0.3204 | 0.8393 | 0.4120 | 0.4532 | 0.5497 | 6.1200 |
| 303 | Gold-Only | DEV_SEARCH | 0.3577 | 0.1584 | 0.1789 | 0.9755 | 0.3726 | 0.2566 | 0.2622 | 0.8693 | 0.2317 | 0.2971 | 0.4402 | 6.4224 |
| 303 | EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| 303 | EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| 303 | EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| 303 | EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |

## Mean DEV_SEARCH deltas (blend minus baseline)

Each cell is averaged across seeds 202 and 303. Per-class entries show AP/F1 separately.

### Versus Gold-Only

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0332 |
| SUBSTITUTION F1 | +0.0250 |
| OMISSION AP | +0.0491 |
| OMISSION F1 | +0.0370 |
| INSERTION AP | +0.1056 |
| INSERTION F1 | +0.0255 |
| CORRECT AP | +0.0049 |
| CORRECT F1 | -0.0011 |
| Macro3_AP | +0.0626 |
| Macro3_F1 | +0.0291 |
| Macro4_F1 | +0.0216 |
| FWR_unfiltered | +0.1836 |

### Versus EXP-36

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0103 |
| SUBSTITUTION F1 | -0.0075 |
| OMISSION AP | -0.0033 |
| OMISSION F1 | +0.0189 |
| INSERTION AP | +0.0124 |
| INSERTION F1 | -0.0121 |
| CORRECT AP | -0.0001 |
| CORRECT F1 | -0.0053 |
| Macro3_AP | +0.0065 |
| Macro3_F1 | -0.0002 |
| Macro4_F1 | -0.0015 |
| FWR_unfiltered | +0.1728 |

### Versus EXP-43

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0340 |
| SUBSTITUTION F1 | -0.0168 |
| OMISSION AP | -0.0043 |
| OMISSION F1 | +0.0226 |
| INSERTION AP | +0.0033 |
| INSERTION F1 | -0.0100 |
| CORRECT AP | +0.0004 |
| CORRECT F1 | -0.0097 |
| Macro3_AP | +0.0110 |
| Macro3_F1 | -0.0014 |
| Macro4_F1 | -0.0035 |
| FWR_unfiltered | +0.5317 |

Seed 202 exact target/gap metadata ID order verified across all sources: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.
Seed 303 exact target/gap metadata ID order verified across all sources: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.

Limitations: DEV_SEARCH is development-exposed, not a sealed confirmatory estimate. Only two replication seeds were evaluated; mean deltas are descriptive. Weights were fixed from the prior DEV_CAL-selected result without retuning. No protected splits or training data were accessed.
