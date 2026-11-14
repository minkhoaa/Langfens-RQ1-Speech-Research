# Fixed gap-positive-weight=3, seed-202 replication

Fixed weight 3 was carried forward from the prior DEV_CAL sweep; there was no weight retuning on DEV_SEARCH. The intervention was trained on CUDA as a matched GRU-64/seed-202, 12-epoch fit. Normalization and class-weight fitting used TRAIN only; checkpoint selection and each model's EVAL-V3 policy calibration used DEV_CAL. The Gold-Only seed-202 comparator is the existing `waveform_full` EVAL-V3 score cache. Both systems were evaluated with the same EVAL-V3 evaluator and policy-selection procedure.

| Measure | Intervention | Gold-Only baseline | Paired delta |
|---|---:|---:|---:|
| Macro3 error AP | 0.235684 | 0.234517 | +0.001168 |
| Macro3 error F1 | 0.305352 | 0.306291 | -0.000939 |
| Macro4 F1 | 0.450567 | 0.454673 | -0.004106 |
| Unfiltered joint FWR (warnings/utterance) | 5.594324 | 4.617696 | +0.976628 |

| Class | Intervention AP | Intervention F1 | Baseline AP | Baseline F1 | Delta AP | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| Correct | 0.976111 | 0.886212 | 0.975278 | 0.899822 | +0.000833 | -0.013610 |
| Substitution | 0.366202 | 0.393583 | 0.370793 | 0.405216 | -0.004591 | -0.011633 |
| Omission | 0.156706 | 0.267696 | 0.146159 | 0.249697 | +0.010547 | +0.017999 |
| Insertion (exact gap) | 0.184144 | 0.254777 | 0.186597 | 0.263959 | -0.002453 | -0.009182 |

The fixed intervention yields a very small Macro3 AP increase, but Macro3 F1 and Macro4 F1 decrease; insertion AP/F1 do not improve. Thus the seed-101 modest gain does not clearly persist as a general metric gain in this seed-202 replication. Runtime was 182.41 s on NVIDIA GeForce GTX 1650. Selected checkpoint: epoch 12, 684 updates, SHA-256 `4a75a78878796aa12654c945f593b99902198ce4ab64d8888a27c312fc656798`.

**Caveat:** DEV_SEARCH is development-exposed and is not an independent held-out test set. FWR is reported unfiltered; no FWR-based filter was applied.
