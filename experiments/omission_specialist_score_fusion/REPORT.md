# Omission-specialist fixed-formula score fusion

No weight search. Target = EXP-24 p(omission), with residual non-omission mass redistributed between .8 EXP-36 + .2 matched Gold-Only p(correct/substitution) in its original ratio. Exact-gap insertion = .8 EXP-43 + .2 matched Gold-Only. Thresholds fitted per system/seed on DEV_CAL only; DEV_SEARCH was evaluated once. DEV_SEARCH is exposed.

| Seed | System | S AP | O AP | I AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | FWR unfiltered |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | Omission-specialist composition | 0.3918 | 0.2251 | 0.2924 | 0.4049 | 0.2535 | 0.2871 | 0.8749 | 0.3031 | 0.3151 | 0.4551 | 6.1636 |
| 101 | Fixed head-wise blend | 0.3975 | 0.2053 | 0.2924 | 0.4041 | 0.2968 | 0.2871 | 0.8761 | 0.2984 | 0.3293 | 0.4660 | 6.1152 |
| 101 | EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |
| 202 | Omission-specialist composition | 0.3920 | 0.2251 | 0.2898 | 0.4036 | 0.2535 | 0.2886 | 0.8752 | 0.3023 | 0.3152 | 0.4552 | 6.1469 |
| 202 | Fixed head-wise blend | 0.3990 | 0.2004 | 0.2898 | 0.4047 | 0.2890 | 0.2886 | 0.8768 | 0.2964 | 0.3274 | 0.4648 | 6.0651 |
| 202 | EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |
| 303 | Omission-specialist composition | 0.3902 | 0.2251 | 0.2868 | 0.4060 | 0.2535 | 0.2886 | 0.8755 | 0.3007 | 0.3160 | 0.4559 | 6.1319 |
| 303 | Fixed head-wise blend | 0.3959 | 0.2025 | 0.2868 | 0.4231 | 0.2912 | 0.2886 | 0.8902 | 0.2951 | 0.3343 | 0.4733 | 5.3422 |
| 303 | EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |
| 404 | Omission-specialist composition | 0.3914 | 0.2251 | 0.2853 | 0.4174 | 0.2535 | 0.2929 | 0.8884 | 0.3006 | 0.3213 | 0.4630 | 5.3856 |
| 404 | Fixed head-wise blend | 0.3979 | 0.1946 | 0.2853 | 0.4038 | 0.2883 | 0.2929 | 0.8754 | 0.2926 | 0.3283 | 0.4651 | 6.1386 |
| 404 | EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |
| 505 | Omission-specialist composition | 0.3931 | 0.2251 | 0.2893 | 0.4173 | 0.2535 | 0.2929 | 0.8874 | 0.3025 | 0.3212 | 0.4628 | 5.4374 |
| 505 | Fixed head-wise blend | 0.3986 | 0.1979 | 0.2893 | 0.4025 | 0.2904 | 0.2929 | 0.8740 | 0.2953 | 0.3286 | 0.4650 | 6.2154 |
| 505 | EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |

## Five-seed mean

| System | S AP | O AP | I AP | S F1 | O F1 | I F1 | Correct F1 | Macro3 AP | Macro3 F1 | Macro4 F1 | FWR unfiltered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Omission-specialist composition | 0.3917 | 0.2251 | 0.2887 | 0.4098 | 0.2535 | 0.2900 | 0.8803 | 0.3018 | 0.3178 | 0.4584 | 5.8531 |
| Fixed head-wise blend | 0.3978 | 0.2001 | 0.2887 | 0.4077 | 0.2911 | 0.2900 | 0.8785 | 0.2955 | 0.3296 | 0.4668 | 5.9753 |
| EXP-36 | 0.3871 | 0.2047 | 0.2759 | 0.4214 | 0.2712 | 0.3010 | 0.8888 | 0.2892 | 0.3312 | 0.4706 | 5.3890 |

Exact target and gap event IDs/order were validated across EXP-24, EXP-36, EXP-43, and each matched Gold-Only seed on DEV_CAL and DEV_SEARCH. See `results.json` for per-seed thresholds, metrics, and counts.
