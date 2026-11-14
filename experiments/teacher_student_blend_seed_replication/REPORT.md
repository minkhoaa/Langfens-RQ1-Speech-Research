# EXP-TSB-02: Fixed alpha 0.8 seed replication

Alpha is fixed at 0.8 EXP-36 weight (0.2 Gold-Only score weight), carried forward from the seed101 DEV_CAL selection. No alpha sweep or retuning was performed. Each model’s EVAL-V3 CAL_MAX_ERROR_MACRO_F1 policy was fitted on its own DEV_CAL scores; each DEV_SEARCH set was evaluated once.

| Seed | Model | DEV_SEARCH | S AP | O AP | I AP | Correct AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 error AP | Macro3 error F1 | Macro4 F1 | Unfiltered FWR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 202 | Blend | DEV_SEARCH | 0.3990 | 0.2004 | 0.2843 | 0.9802 | 0.4047 | 0.2890 | 0.3102 | 0.8768 | 0.2946 | 0.3347 | 0.4702 | 6.2053 |
| 202 | Gold-Only | DEV_SEARCH | 0.3708 | 0.1462 | 0.1866 | 0.9753 | 0.4052 | 0.2497 | 0.2640 | 0.8998 | 0.2345 | 0.3063 | 0.4547 | 4.6177 |
| 303 | Blend | DEV_SEARCH | 0.3959 | 0.2025 | 0.2812 | 0.9802 | 0.4231 | 0.2912 | 0.3254 | 0.8902 | 0.2932 | 0.3466 | 0.4825 | 5.3472 |
| 303 | Gold-Only | DEV_SEARCH | 0.3577 | 0.1584 | 0.1789 | 0.9755 | 0.3726 | 0.2566 | 0.2622 | 0.8693 | 0.2317 | 0.2971 | 0.4402 | 6.4224 |

## Blend minus same-seed Gold-Only deltas

| Seed | Metric | AP / value delta | F1 delta |
|---|---|---:|---:|
| 202 | SUBSTITUTION | +0.0282 | -0.0005 |
| 202 | OMISSION | +0.0542 | +0.0393 |
| 202 | INSERTION | +0.0977 | +0.0463 |
| 202 | CORRECT | +0.0049 | -0.0230 |
| 202 | Macro-3 error AP | +0.0600 | — |
| 202 | Macro-3 error F1 | +0.0284 | — |
| 202 | Macro-4 F1 | +0.0155 | — |
| 202 | Unfiltered FWR | +1.5876 | — |
| 303 | SUBSTITUTION | +0.0382 | +0.0505 |
| 303 | OMISSION | +0.0441 | +0.0346 |
| 303 | INSERTION | +0.1023 | +0.0632 |
| 303 | CORRECT | +0.0048 | +0.0209 |
| 303 | Macro-3 error AP | +0.0615 | — |
| 303 | Macro-3 error F1 | +0.0494 | — |
| 303 | Macro-4 F1 | +0.0423 | — |
| 303 | Unfiltered FWR | -1.0751 | — |
| Mean | SUBSTITUTION | +0.0332 | +0.0250 |
| Mean | OMISSION | +0.0491 | +0.0370 |
| Mean | INSERTION | +0.1000 | +0.0547 |
| Mean | CORRECT | +0.0049 | -0.0011 |
| Mean | Macro-3 error AP | +0.0608 | — |
| Mean | Macro-3 error F1 | +0.0389 | — |
| Mean | Macro-4 F1 | +0.0289 | — |
| Mean | Unfiltered FWR | +0.2563 | — |

## Alignment and limitations

- Seed 202 DEV_CAL: teacher/student exact target and gap metadata order matched; 10043 targets, 10343 gaps.
- Seed 202 DEV_SEARCH: teacher/student exact target and gap metadata order matched; 19755 targets, 20354 gaps.
- Seed 303 DEV_CAL: teacher/student exact target and gap metadata order matched; 10043 targets, 10343 gaps.
- Seed 303 DEV_SEARCH: teacher/student exact target and gap metadata order matched; 19755 targets, 20354 gaps.
- Two additional seeds only; descriptive, not inferential.
- DEV_SEARCH results are exploratory, not sealed confirmatory estimates.
- Alpha=0.8 was fixed from the seed101 DEV_CAL result; no retuning.
- Cached scores and metadata alignment were used; no training variability was tested.
