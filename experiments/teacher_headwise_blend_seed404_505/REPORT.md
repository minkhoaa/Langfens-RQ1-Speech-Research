# Student-seed replication of the frozen head-wise blend

Fixed weights, with no retuning: target probabilities = 0.8 EXP-36 + 0.2 matched Gold-Only seed; exact-gap insertion probability = 0.8 EXP-43 + 0.2 matched Gold-Only seed. CAL_MAX_ERROR_MACRO_F1 thresholds were fitted separately per seed and system on DEV_CAL. DEV_SEARCH was evaluated once.

| Seed | System | Split | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | FWR unfiltered |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 404 | Head-wise blend | DEV_CAL | 0.5614 | 0.5819 | 0.2982 | 0.9510 | 0.5397 | 0.5945 | 0.4156 | 0.8514 | 0.4805 | 0.5166 | 0.6003 | 5.5700 |
| 404 | Head-wise blend | DEV_SEARCH | 0.3979 | 0.1946 | 0.2853 | 0.9802 | 0.4038 | 0.2883 | 0.2929 | 0.8754 | 0.2926 | 0.3283 | 0.4651 | 6.1386 |
| 404 | Gold-Only | DEV_CAL | 0.5159 | 0.4661 | 0.1979 | 0.9387 | 0.5064 | 0.5339 | 0.2718 | 0.8654 | 0.3933 | 0.4374 | 0.5444 | 4.5433 |
| 404 | Gold-Only | DEV_SEARCH | 0.3595 | 0.1365 | 0.1641 | 0.9744 | 0.3956 | 0.2456 | 0.2416 | 0.8962 | 0.2200 | 0.2943 | 0.4448 | 4.9332 |
| 404 | EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| 404 | EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| 404 | EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| 404 | EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |
| 505 | Head-wise blend | DEV_CAL | 0.5634 | 0.5781 | 0.2909 | 0.9511 | 0.5394 | 0.5895 | 0.4156 | 0.8512 | 0.4775 | 0.5148 | 0.5989 | 5.5667 |
| 505 | Head-wise blend | DEV_SEARCH | 0.3986 | 0.1979 | 0.2893 | 0.9803 | 0.4025 | 0.2904 | 0.2929 | 0.8740 | 0.2953 | 0.3286 | 0.4650 | 6.2154 |
| 505 | Gold-Only | DEV_CAL | 0.5308 | 0.4953 | 0.2087 | 0.9414 | 0.5066 | 0.5413 | 0.3398 | 0.8446 | 0.4116 | 0.4626 | 0.5581 | 5.9667 |
| 505 | Gold-Only | DEV_SEARCH | 0.3701 | 0.1507 | 0.1842 | 0.9759 | 0.3763 | 0.2558 | 0.2528 | 0.8685 | 0.2350 | 0.2950 | 0.4383 | 6.5259 |
| 505 | EXP-36 | DEV_CAL | 0.5601 | 0.5744 | 0.2765 | 0.9509 | 0.5426 | 0.5872 | 0.3786 | 0.8636 | 0.4703 | 0.5028 | 0.5930 | 4.9667 |
| 505 | EXP-36 | DEV_SEARCH | 0.3871 | 0.2047 | 0.2759 | 0.9803 | 0.4214 | 0.2712 | 0.3007 | 0.8888 | 0.2892 | 0.3311 | 0.4705 | 5.5309 |
| 505 | EXP-43 | DEV_CAL | 0.5480 | 0.5703 | 0.2857 | 0.9510 | 0.5435 | 0.5807 | 0.3896 | 0.8636 | 0.4680 | 0.5046 | 0.5944 | 4.8033 |
| 505 | EXP-43 | DEV_SEARCH | 0.3635 | 0.2058 | 0.2851 | 0.9798 | 0.4307 | 0.2675 | 0.2985 | 0.8932 | 0.2848 | 0.3322 | 0.4725 | 5.1720 |

## Mean DEV_SEARCH deltas (blend minus baseline)

Each cell is averaged across the two new seeds 404 and 505. Per-class entries show AP/F1 separately.

### Versus Gold-Only

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0335 |
| SUBSTITUTION F1 | +0.0172 |
| OMISSION AP | +0.0526 |
| OMISSION F1 | +0.0387 |
| INSERTION AP | +0.1132 |
| INSERTION F1 | +0.0457 |
| CORRECT AP | +0.0051 |
| CORRECT F1 | -0.0076 |
| Macro3_AP | +0.0664 |
| Macro3_F1 | +0.0339 |
| Macro4_F1 | +0.0235 |
| FWR_unfiltered | +0.4474 |

### Versus EXP-36

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0112 |
| SUBSTITUTION F1 | -0.0182 |
| OMISSION AP | -0.0084 |
| OMISSION F1 | +0.0181 |
| INSERTION AP | +0.0114 |
| INSERTION F1 | -0.0077 |
| CORRECT AP | -0.0001 |
| CORRECT F1 | -0.0141 |
| Macro3_AP | +0.0047 |
| Macro3_F1 | -0.0026 |
| Macro4_F1 | -0.0055 |
| FWR_unfiltered | +0.6461 |

### Versus EXP-43

| Measure | Mean delta |
|---|---:|
| SUBSTITUTION AP | +0.0348 |
| SUBSTITUTION F1 | -0.0275 |
| OMISSION AP | -0.0095 |
| OMISSION F1 | +0.0218 |
| INSERTION AP | +0.0023 |
| INSERTION F1 | -0.0056 |
| CORRECT AP | +0.0004 |
| CORRECT F1 | -0.0185 |
| Macro3_AP | +0.0092 |
| Macro3_F1 | -0.0037 |
| Macro4_F1 | -0.0074 |
| FWR_unfiltered | +1.0050 |

Seed 404 exact target/gap metadata ID order verified across all sources: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.
Seed 505 exact target/gap metadata ID order verified across all sources: DEV_CAL 10043/10343; DEV_SEARCH 19755/20354.

Limitations: DEV_SEARCH is development-exposed, not a sealed confirmatory estimate. Only two replication seeds were evaluated; mean deltas are descriptive. Weights were fixed from the prior DEV_CAL-selected result without retuning. No protected splits or training data were accessed.
