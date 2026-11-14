# Gap-head-only focal gamma=1.5 replication (seed 202)

Single exploratory seed replication. The gap-only focal arm was trained with target weighted CE and gap square-root-inverse-weighted focal loss (gamma 1.5), GRU-64, 12 epochs/684 updates, AdamW (lr 0.001, weight decay 0.0001), TRAIN-only normalization and class weights. The minimum unweighted target+gap CE on DEV_CAL selected epoch 10. EVAL-V3 policy selection was performed on DEV_CAL; DEV_SEARCH was evaluated once and not used for tuning. Baseline probabilities are the existing Gold-Only seed-202 EVAL-V3 `waveform_full` caches (event IDs/order checked against canonical split); baseline labels/metadata are joined in canonical order.

## DEV_SEARCH results

F1 is at each arm's DEV_CAL-selected EVAL-V3 policy operating point; AP is unfiltered. Deltas are focal experiment minus matched Gold-Only baseline.

| Class | AP experiment | AP baseline | Δ AP | F1 experiment | F1 baseline | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| CORRECT | 0.975851 | 0.975278 | +0.000572 | 0.898950 | 0.899822 | -0.000872 |
| SUBSTITUTION | 0.373443 | 0.370793 | +0.002649 | 0.406585 | 0.405216 | +0.001369 |
| OMISSION | 0.147694 | 0.146159 | +0.001535 | 0.265570 | 0.249697 | +0.015873 |
| INSERTION (exact gap) | 0.150501 | 0.186597 | -0.036096 | 0.221344 | 0.263959 | -0.042616 |

| Aggregate | Experiment | Baseline | Δ |
|---|---:|---:|---:|
| Macro3 error AP | 0.223879 | 0.234517 | -0.010637 |
| Macro3 error F1 | 0.297833 | 0.306291 | -0.008458 |
| Macro4 mixed-domain F1 | 0.448112 | 0.454673 | -0.006561 |
| FWR (warnings/utterance) | 4.804674 | 4.617696 | +0.186978 |

## Runtime and checkpoints

- Experiment device: NVIDIA GeForce GTX 1650 CUDA; runtime 45.340 seconds; peak allocated VRAM 43.976 MiB.
- Experiment checkpoint: `gap_only_focal_gamma1p5_seed202.pt`; SHA-256 `38e402e4f43be74ba07844b194cb9147bfed6672a8ba1fd34ebad5daca58a48f`.
- Gold-Only seed-202 checkpoint: `models/student/STU-01-GOLD-ONLY_seed202.pt`; SHA-256 `c08c771fba5a6d6295fcd7411688ba19636aef8e987b8098fd5146fd063727a8`; its training receipt records 50.897 seconds and 43.976 MiB peak VRAM.
- Baseline EVAL-V3 score caches: `experiments/evaluation_v3/closure/waveform_full/scores/STU-01-GOLD-ONLY_seed202_dev_cal.npz` and `_dev_search.npz`.

This one-seed replication does not reproduce the seed-101 insertion AP/F1 gain: exact-gap insertion AP and F1, aggregate error AP/F1, macro4 F1, and FWR are worse than the matched seed-202 baseline. No broader conclusion is warranted. Full evaluator output, DEV_CAL policy and run parameters are in `results.json`; epoch-level output is in `run.log`.
