# Agent C — frozen teacher ensemble check (EVAL-V3)

## Result
Used the already-frozen EXP-36 seven-model super-fusion on its `dev_replication` prediction set (300 utterances; 9,863 target phones; 10,163 gaps). This is a real precomputed teacher ensemble result, not newly fitted or tuned. EXP-36 includes EXP-24 JSAC, EXP-28 APED and EXP-34 PNC plus four additional models. The predeclared weights are the ones in EXP-36 `RUN_SPEC.yaml`; no DEV_SEARCH data was accessed in this check.

| Class | AP | F1 |
|---|---:|---:|
| CORRECT | 0.9721 | 0.9357 |
| OMISSION | 0.3742 | 0.3576 |
| SUBSTITUTION | 0.4041 | 0.3596 |
| INSERTION | 0.2519 | 0.3636 |
| Macro (four classes) | 0.5835 | 0.5510 |

The corresponding frozen run records utility 0.4303, 356 false warnings / 300 utterances, and `EXPLORATORY` status. These are historical experiment metrics, not a new independent reproduction. Runtime/checkpoint: no inference or training was run; source experiment declares GTX 1650 and 208 MB allocated VRAM, with no deployable ensemble checkpoint listed (the result is precomputed predictions).

## Gold-only comparison / reference validity
Gold-only EVAL-V3 waveform score caches exist for train, DEV_CAL and DEV_SEARCH, but there is no Gold-only `dev_replication` score/prediction artifact in the available files. Therefore a fair same-split Gold-only comparison and a new paired EVAL-V3 AP/F1 evaluation cannot be reported. No DEV_SEARCH tuning or protected-holdout evaluation was performed. EXP-36's own run config identifies its intended references as the caller-provided canonical prompt/orthographic references; this check consumed only its frozen DEV_REPLICATION outputs, not annotated boundaries or gold labels as inference input. The numeric ensemble metrics above are the run's recorded replication evaluation.

## Config and next experiment
Frozen seven-model weights: target `[.12,.15,.12,.12,.16,.16,.17]`, gap `[.08,.12,.12,.18,.12,.18,.20]`; thresholds S/O `.55`, I `.50`. Next: replay the Gold-only checkpoint on the exact EXP-36 `dev_replication` utterances with EVAL-V3's canonical caller reference, then evaluate Gold-only and frozen ensemble side-by-side with identical masks and policy fixed before evaluation. If an aligned teacher score interface is available, next train a small student using DEV_TRAIN labels/teacher outputs and select policies only on DEV_CAL; never train on DEV_SEARCH or protected holdouts.

Command: `python experiments/teacher_evaluation/agent_c_teacher_ensemble_metrics.py`
