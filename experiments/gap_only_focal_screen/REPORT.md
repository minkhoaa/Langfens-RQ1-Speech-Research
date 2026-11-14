# Gap-head-only focal-loss screen (seed 101)

Exploratory single-seed run; DEV_SEARCH was evaluated once. GRU-64, 12 epochs (684 updates), batch 32, AdamW (lr 0.001, weight decay 0.0001), TRAIN-only normalization, and the original square-root inverse TRAIN class weights were retained. Target loss was weighted cross-entropy; gap loss was weighted focal loss with gamma 1.5. The checkpoint was selected by unweighted DEV_CAL target+gap cross-entropy, and the existing EVAL-V3 DEV_CAL policy selection was used. No FWR filtering was applied.

## Runtime and checkpoint

- Device: NVIDIA GeForce GTX 1650 CUDA.
- Runtime: 44.585 seconds; peak allocated VRAM: 44.094 MiB.
- Selected epoch: 11; checkpoint: `gap_only_focal_gamma1p5_seed101.pt`.
- SHA-256: `3d7a16cef6ddcc6de27b0ac324a019be349fb5fc0396de871d712b583e6d9279`.
- Paired reference: existing gold-only seed-101 EVAL-V3 score caches; same DEV_CAL policy-fitting procedure.

## DEV_SEARCH paired results

All deltas are experiment minus baseline. Per-class F1 is the DEV_CAL-policy operating point; AP is the unfiltered ranking metric.

| Class | AP experiment | AP baseline | Δ AP | F1 experiment | F1 baseline | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| CORRECT | 0.976355 | 0.975879 | +0.000477 | 0.884802 | 0.882695 | +0.002107 |
| SUBSTITUTION | 0.365318 | 0.364602 | +0.000715 | 0.392556 | 0.386590 | +0.005966 |
| OMISSION | 0.158597 | 0.159433 | -0.000836 | 0.266491 | 0.251012 | +0.015479 |
| INSERTION (exact gap) | 0.199636 | 0.186715 | +0.012920 | 0.245734 | 0.262821 | -0.017087 |

| Aggregate | Experiment | Baseline | Δ |
|---|---:|---:|---:|
| Macro3 error AP | 0.241183 | 0.236917 | +0.004267 |
| Macro3 error F1 | 0.301594 | 0.300141 | +0.001453 |
| Macro4 mixed-domain F1 | 0.447396 | 0.445779 | +0.001616 |
| FWR unfiltered (warnings/utterance) | 5.659432 | 5.737896 | -0.078464 |

Insertion ranking AP improved, but insertion F1 decreased at the DEV_CAL-selected operating point. This single exploratory seed does not establish a general effect. Unfiltered FWR is far above the 0.20 reference (5.66 warnings/utterance); the policy is not safe/deployable. See `results.json` for evaluator details, full class metrics, and baseline values.
