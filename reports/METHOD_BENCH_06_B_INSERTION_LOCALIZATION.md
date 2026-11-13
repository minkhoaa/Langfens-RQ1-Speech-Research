# METHOD-BENCH-06 Agent B — Exact-gap insertion localization

## Verdict

**A direct temporal experiment was feasible and executed, but it failed to improve insertion ranking.** On seed-101 four-fold speaker-grouped TRAIN OOF, the six-feature temporal-pooling probe achieved pooled exact-gap AP-I **0.02220**, versus **0.03557** for its same-classifier static gap-feature control and the historical direct neural insertion head **0.162125**. The temporal method was better than static only on fold 3 and was below it on folds 0–2. It does not support a claim that frame-localization helps here.

## Recovered historical baseline and folds

The original report incorrectly said the MB03 event-level neural baseline could not be recovered. I verified `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`; it contains four `gap_PLAIN_{0..3}` vectors. `speaker_grouped_oof_receipt.json` confirms all N+1 gap predictions, fold membership, supports and AP. Historical PLAIN neural AP-I is **0.16212489699422541** on 61,362 TRAIN gaps / 560 positives. Fold AP-I/supports:

| Fold | Held speakers | Gaps | I positives | MB03 direct neural AP-I |
|---|---|---:|---:|---:|
| 0 | ABA, ASI, BWC | 15,266 | 134 | .188792 |
| 1 | EBVS, HJK, HKK | 15,298 | 127 | .102708 |
| 2 | HQTV, LXC, NJS | 15,439 | 150 | .109449 |
| 3 | PNV, SKA, SVBI | 15,359 | 149 | .251541 |

These folds partition 12 TRAIN speakers. The MB03 receipt specifies U2's gap inputs as adjacent predicted-center crop edge means plus adjacent intended references. The report's old claim of missing exact-event outputs is withdrawn.

## Input inventory and feasibility

- Raw 16-kHz WAVs are present under `data/derived/l2arctic/wavs_16k/`.
- There are also cached waveform replay archives under `experiments/evaluation_v3/closure/waveform_full/utterances/`; the TRAIN utterance archive includes `ctc_logits` with shape `(frames, 42)`, canonical phones, and static `gap_X`. The archives index predicted frame-level CTC outputs, not gold insertion times.
- `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/train_frame_context.npy` has shape `(59,562,17,768)` and is target-indexed; it does **not** provide an utterance-time sequence. H001 saved `gap_event_ids` and `gap_probabilities` are event-level outputs, not a frame timeline.
- The cached replay frame logits plus the canonical-conditioned predicted alignment provide a valid timeline; therefore a temporal experiment could be run without fabricating features. Raw WAVs were not decoded/retrained in this experiment. No gold gap label/timing was used to form an inference feature.

## Method and protocol

**ID:** `RQ1-MB06-B-GROUPED-TEMPORAL-LOGREG-S101`  
**Implementation:** `experiments/rq1_method_bench_06/agent_b/grouped_temporal_probe.py`  
**Receipt/scores:** `grouped_temporal_receipt.json`, `grouped_temporal_scores.npz`; `temporal_oof_pr.csv`, `static_oof_pr.csv` preserve PR coordinates.

For each reference gap, the frozen frame logits were aligned to the canonical phone sequence using predicted CTC alignment (never gold timing). At each predicted gap center a fixed ±5-frame window was pooled into six summaries: mean and maximum non-silence frame maximum, late-minus-early maximum, mean silence posterior, mean frame entropy, and fraction of frames where non-silence maximum exceeds silence. The leading/trailing gaps are retained according to the predicted alignment's boundary centers. The probe is a balanced `LogisticRegression(C=1, solver=lbfgs, max_iter=500, random_state=101)` fit independently in each outer fold. Its comparison control uses the same classifier and fixed six-dimensional existing static `gap_X` features. The direct neural baseline is historical MB03 rather than retrained here.

All 61,362 canonical gap IDs were emitted in manifest utterance order and then positions `0..N`, with 560 positive labels. Labels were used only for outer-training fit and held-fold scoring. Exact event IDs/speakers/labels/features and predictions are preserved in the NPZ. No checkpoint was produced. CPU only; elapsed time **15.13 s**; peak memory was not measured.

## Results

P@560 reports precision among each fold's top 560 ranked events (pooled positive support is 560); P@5% uses the top five percent of that fold's gaps. Train-fit AP is calculated within each outer-training partition, not on its held fold.

| Fold | Temporal train-fit AP-I | Temporal held AP-I | Temporal P@560 / P@5% | Static held AP-I | Static P@560 / P@5% | MB03 neural held AP-I |
|---|---:|---:|---:|---:|---:|---:|
| 0 | .02414 | .02004 | .0196 / .0209 | .03569 | .0518 / .0432 | .18879 |
| 1 | .02212 | .03068 | .0464 / .0444 | .03913 | .0518 / .0392 | .10271 |
| 2 | .02541 | .01982 | .0214 / .0246 | .05445 | .0804 / .0635 | .10945 |
| 3 | .02285 | .02621 | .0446 / .0352 | .01879 | .0375 / .0312 | .25154 |
| Pooled OOF | **.02220** | **.02220** | **.0304 / .0313** | **.03557** | **.1214 / .0437** | **.162125** |

Pooled temporal train-fit mean AP was .02363. Static P@560 is pooled precision among the highest 560 of all OOF scores; per-fold P@560 uses 560 events within each held fold. Static and temporal pooled precision-at-K values have this respective definition; use AP as the primary ranking metric. Event prevalence overall is 560/61,362 = .00913; the temporal held-fold estimates are only modestly above that prevalence.

The temporal score loses to static in three of four folds and has a pooled deficit of .01337 AP. Both are far below the historical grouped neural head. This is negative evidence for this particular six-statistic local pooling mechanism, not for all acoustic temporal models or learned frame-sequence scoring.

## Prior work and controls reviewed

The original external-method review is in [METHOD-BENCH-06-C](METHOD_BENCH_06_C_EXTERNAL_RESEARCH.md); its MixGoP density experiment is target-event scoring, not a gap-localization method. MB05 reports the already-tested T2 score: maximum candidate-phone minus blank log posterior in a ±2-frame window around a predicted Viterbi boundary; its AP-I was .004263 on the 10,139-gap HJK/HKK diagnostic subset. T2 is not the same feature/objective as this experiment's six temporal-summary logistic classifier, though both are local predicted-alignment temporal probes. The temporal result here is not evidence against all temporal localization.

The external review considered the original ICASSP 2022 *Phoneme Mispronunciation Detection By Jointly Learning To Align* (alignment jointly optimized with detection), 2025 work on alignment error and pronunciation scoring, and phonological edit alignment (sequence-level rather than acoustic gap evidence). Those establish motivation/measurement concerns, but do not establish that our local pooled score should improve exact-gap AP. No external paper result is presented as a replication or matched benchmark. Existing MB05/MB06-C external sources are retained; no new source claim is needed for this diagnostic.

## Reproduction and limits

From repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_06/agent_b/grouped_temporal_probe.py
```

The first attempt exposed a diagnostic aggregation bug in the train-fit summary; it failed before writing result outputs. The corrected run above completed in 15.13 s and its receipt contains the final metrics. The extraction uses the existing waveform replay cache and canonical-conditioned predicted alignments; no waveform model inference was run. CPU timing is wall time; memory profiling and a separately trained checkpoint are not applicable/not measured.

**Conclusion:** insertion localization remains unresolved. The direct temporal test is a measured negative against both its same-family static control and the grouped direct neural reference. It does not change the historical baseline or MB05 artifacts, and does not claim to improve the existing system.
