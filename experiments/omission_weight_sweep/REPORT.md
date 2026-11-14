# Omission target-weight sweep

Seed 101; GRU-64 target/gap model; 12 epochs per candidate; AdamW, batch 32. TRAIN-only feature scaling and class frequencies. Only target omission weight changed by factors 1, 2, and 4; gap head/weights unchanged. Each checkpoint selected by minimum unweighted DEV_CAL target+gap CE. Candidate selection used DEV_CAL Macro-3 error AP; DEV_SEARCH was evaluated once for the selected candidate. Thresholds are canonical `CAL_MAX_ERROR_MACRO_F1` fit on DEV_CAL. FWR is unfiltered.

## DEV_CAL candidate selection

| Omission weight factor | Correct AP | S AP | O AP | Macro-3 AP | checkpoint epoch |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9415 | 0.5280 | 0.4789 | 0.6495 | 11 |
| 2 | 0.9385 | 0.5178 | 0.4799 | 0.6454 | 11 |
| 4 | 0.9347 | 0.5088 | 0.4538 | 0.6324 | 10 |

Selected factor: **1**. Selected checkpoint SHA-256: `eacaf68385e5f2ae23c7856a05aabb1dd603391123469a5ddd760e18e84dc716`. Runtime: 187.71s on NVIDIA GeForce GTX 1650.

## Selected-model DEV_SEARCH EVAL-V3

| Metric/class | AP | F1 |
|---|---:|---:|
| SUBSTITUTION | 0.3646 | 0.3866 |
| OMISSION | 0.1594 | 0.2510 |
| INSERTION | 0.1867 | 0.2628 |
| Macro-3 error | 0.2369 | 0.3001 |
| Macro-4 mixed-domain | — | 0.4458 |

Unfiltered joint FWR: **5.7379**. DEV_CAL policy thresholds: `{"insertion": 0.5634348285198211, "omission": 0.6101865023374557, "substitution": 0.3293576631695032}`.

## Baseline comparison

Against the task-supplied prior full-EVAL baseline (Macro-3 AP .2580; omission AP .1452), this run gives Macro-3 AP 0.2369 (Δ -0.0211) and omission AP 0.1594 (Δ +0.0142). These are below/above the supplied baseline respectively; the binary specialist omission AP is from a different model/task and not directly comparable.

All candidate checkpoints, candidate DEV_CAL probabilities, selected DEV_SEARCH predictions, histories, hashes, and full metrics are retained in this namespace.
