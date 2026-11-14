# Temporal local-context direction: results

## Question and scope

Does a small order-aware local temporal CNN over frozen Wav2Vec2 layer-6 windows improve over fit-only 13-layer SM13 target/gap heads under the same speaker split, 12/5-epoch optimizer schedules, calibration policy, and EVAL-V3.1 scorer?

This is exploratory grouped CV on L2-ARCTIC TRAIN, not a fresh confirmatory test. The comparison is specific to one local-window definition, one CNN, and seeds 101 and 202. It does not establish that temporal models in general are unsuitable.

## Protocol

For fold `f`, speakers `GROUPS[f]` are held once, `GROUPS[(f+1)%4]` are used only for calibration, and the other six speakers are fit-only. Each seed's four held groups form one pooled OOF prediction set: exactly 1,800 utterances, 59,562 targets, and 61,362 gaps. Fit and calibration speakers do not overlap held speakers. No dev, test-sealed, or SpeechOcean data are loaded.

The shared evaluator freezes B=.2 warning cutoffs from calibration labels, then applies them once to the held group. Target warning score is the normalized monotonic transform of `max(P(S),P(O))-P(C)`; warning type is the larger of S/O. SHA-256 event-key ordering resolves ties. Exact N targets and N+1 gaps are evaluated with `UnifiedEvaluatorV3` EVAL-V3.1. Classwise precision, recall, F1 and AP are retained. Held fixed-precision recall summaries at .75/.80 are oracle diagnostics because held labels set the operating point; they are not deployable results.

The score summary below pools the four disjoint held folds per seed and runs EVAL-V3.1 once on that full OOF set. Means and seed ranges are descriptive only: the two seeds reuse the same speakers, so they are not independent samples.

## Results

| Model | Seed 101 Macro-3 AP / F1 / joint FWR | Seed 202 Macro-3 AP / F1 / joint FWR | Seed mean Macro-3 AP / F1 / joint FWR | Mean AP change vs all13 |
|---|---:|---:|---:|---:|
| Shared SM13 all13 target + gap | .24687 / .12495 / .21889 | .25071 / .12987 / .23500 | .24879 / .12741 / .22694 | — |
| Pooled-L6 target + shared SM13 gap | .21919 / .10370 / .18056 | .22425 / .11720 / .22889 | .22172 / .11045 / .20472 | -.02707 |
| **Temporal target + shared SM13 gap (primary)** | **.19194 / .09811 / .17611** | **.19851 / .10277 / .19556** | **.19523 / .10044 / .18583** | **-.05357** |
| Full temporal target + gap | .18255 / .08449 / .16389 | .18607 / .08362 / .18889 | .18431 / .08406 / .17639 | -.06448 |

Primary temporal seed ranges were Macro-3 AP `.19194–.19851`, Macro-3 F1 `.09811–.10277`, and joint FWR `.17611–.19556`. Its classwise pooled OOF AP was S/O/I `.31472/.15567/.10542` at seed 101 and `.32829/.15423/.11303` at seed 202. The full temporal model shares target scores, but its temporal gap branch lowers insertion AP/F1 to `.07725/.10138` (seed 101) and `.07570/.10119` (seed 202).

The primary candidate trails all13 in Macro-3 AP and F1 in both seeds and all eight held-group/seed runs. It also trails the pooled-L6 target control in pooled Macro-3 AP and F1 in both seeds. Full temporal gap training further reduces insertion ranking and F1. This implementation did not extract a useful generalization gain from local frame order under this protocol. It does not isolate whether the cause is the convolution/pooling design, layer-6 choice, local window width, or optimization.

The all13 neural baseline's held realized FWR was `.21889` and `.23500` after pooling by seed. This reflects held-set realization under calibration-frozen thresholds; it is not a post-hoc threshold correction.

Per-seed pooled Macro-3 AP/F1/FWR and classwise S/O/I P/R/F1/AP are in [temporal_pooled_oof_metrics.csv](temporal_pooled_oof_metrics.csv). Exact fold-level outcomes and paired deltas for Macro-3 AP, F1, and FWR are in [temporal_fold_and_pooled_deltas.csv](temporal_fold_and_pooled_deltas.csv). Machine-readable pooled OOF evaluator results and event-level IDs, decisions, and probabilities are in sibling `../shared/` outputs and the `baseline_predictions/`, `l6_pooled_predictions/`, `target_only_predictions/`, and `full_temporal_predictions/` directories.

## Models and compute

The shared all13 baseline trains the original SM13 scalar-mix target and gap head classes from frozen pooled layer caches. Target: 12 epochs, batch 512, AdamW `lr=1e-3`, `weight_decay=.01`; gap: 5 epochs, batch 1024, AdamW `lr=8e-4`, `weight_decay=.0001`. The gap schedule matches the SM13 training regimen. The pooled-L6 target control uses the same target head with layer-6 features repeated across scalar-mix slots and shares the baseline gap branch.

The temporal target uses an ordered 17-frame, 768-channel window; the temporal gap uses an ordered 10-frame window. Each branch uses two width-128 Conv1d layers, GELU, learned attention pooling plus temporal mean pooling, and an MLP event head. The target and gap schedules match their corresponding SM13 branches. Trainable parameter counts and per-epoch losses are recorded in fold JSON; state dictionaries are retained.

Windows come from frozen `facebook/wav2vec2-base-960h` layer 6 and TRAIN audio only, centered on existing frozen CharSiu alignment positions. Extraction receipt and hashes are in `../encoder_adapter/raw_cache_receipt.json`. Hardware: NVIDIA GeForce GTX 1650 (3.91 GB reported CUDA memory), 12th Gen Intel Core i5-12450H (12 CPUs), Linux 7.3.3, Python 3.11.17, PyTorch 2.6.0+cu124. The shared GPU lock is `../gpu.lock`; the runner released it after each fold.

## Reproduction

```bash
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py --smoke
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py --baseline-only
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py
.venv/bin/python experiments/sm13_three_directions_20261113/shared/aggregate_baseline.py
```

The one-fold, one-epoch smoke writes to a separate `temporal/smoke/` directory and is excluded. The main run skips completed fold/seed outputs and supports a scheduled pair with `--seed 101 --fold 0`.
