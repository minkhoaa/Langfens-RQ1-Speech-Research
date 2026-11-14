# Independent cross-branch audit

## Scope and method

This audit re-evaluated saved held-fold predictions with `UnifiedEvaluatorV3`; it did not train models or use any new data. The protocol has four disjoint held-speaker groups, with 6 speakers for fit and 3 for calibration in each fold. Per-seed pooled OOF results concatenate the four held folds (1,800 utterances, 59,562 target events, 61,362 gap events) and evaluate once. The 8 fold-seed results repeat speaker groups across seeds, so they are descriptive rather than independent replicates.

## Data-expansion audit

- The 39-phone one-hot reference map from the sorted non-special CharSiu tokenizer inventory exactly reproduces all 59,562 `DatasetManager.target_refs` rows. Four unsupported reference symbols (`D_`, `ER)`, `SPN`, `Z_`) remain zero vectors, consistent with the historical reference encoding; the candidate filters unsupported prompted phones before extraction.
- An initial pseudo-feature cache path used `scipy.resample_poly` and bypassed the Wav2Vec2 `AutoFeatureExtractor` normalization. That path was rejected and its predictions are superseded. The corrected path uses the repository's torchaudio resampler and PCM16 roundtrip, then `AutoFeatureExtractor` normalization for Wav2Vec2.
- On known TRAIN utterance `ABA_arctic_a0003` (44.1 kHz source), the corrected resampled waveform is sample-identical to the repository's 16 kHz waveform. Recomputed pooled layer-6 features match the existing cache (RMSE `4.54e-7`, max absolute difference `4.89e-6`). The prior scipy/no-normalization route had RMSE `0.03575`, max difference `1.0782`; those cache/results are invalidated.
- Re-evaluation of all 16 corrected data-expansion and labelled-control fold files exactly matches the branch's pooled `POOLED_OOF_RESULTS.json` for both seeds and both systems (zero metric/count delta). The labelled-only control exactly matches shared SM13, as expected.

## Adapter audit

- All 8 `matched_pair_seed{seed}_fold{fold}.json` receipts and all 16 eligible control/candidate NPZs are present. I excluded earlier float32-cache controls and used only `encoder_adapter_control_prefix_*` and `encoder_adapter_lora_*` predictions.
- The amended frozen control recomputes layers 5–9 from the same fp16 prefix through frozen blocks 4–8 as the candidate, with the documented same warm start/scaler/schedule; gap scores are the same shared all-13-layer SM13 gap model. The amendment and pair receipts identify this matched input path.
- Direct EVAL-V3.1 on each of the 16 fold files exactly matches its receipt for Macro-3 error AP/F1, joint FWR, warning count, and event counts. Event IDs are unique within each domain, speaker IDs match the receipt's held speakers, all probabilities are finite, and target probability rows sum to one.
- Metric names matter: `macro3_error_ap` and `macro3_error_f1` are the primary S/O/I error metrics. `macro3_target_f1` includes CORRECT; `macro4_mixed_domain_f1` combines target and gap domains and is not a four-way softmax.

## Pooled metrics

Values are seed 101 / seed 202; parentheses show the unweighted mean of the two per-seed pooled results. AP and F1 are Macro-3 S/O/I error metrics. FWR is realized held joint utterance FWR after each fold's calibration-frozen thresholds.

| System | Macro-3 AP | Macro-3 F1 | Held joint FWR |
|---|---:|---:|---:|
| Shared SM13 all-13 baseline | .2469 / .2507 (.2488) | .1249 / .1299 (.1274) | .2189 / .2350 (.2269) |
| Data-expansion labelled control | .2469 / .2507 (.2488) | .1249 / .1299 (.1274) | .2189 / .2350 (.2269) |
| Data-expansion pseudo-target candidate | .2463 / .2481 (.2472) | .1239 / .1406 (.1323) | .2128 / .2556 (.2342) |
| Adapter matched frozen control | .2325 / .2352 (.2338) | .1225 / .1323 (.1274) | .2189 / .2106 (.2147) |
| Adapter LoRA candidate | .2332 / .2346 (.2339) | .1211 / .1316 (.1264) | .2144 / .2117 (.2131) |
| L6 pooled target + shared SM13 gap | .2192 / .2243 (.2217) | .1037 / .1172 (.1105) | .1806 / .2289 (.2047) |
| Temporal target + shared SM13 gap | .1919 / .1985 (.1952) | .0981 / .1028 (.1004) | .1761 / .1956 (.1858) |
| Full temporal target and gap | .1825 / .1861 (.1843) | .0845 / .0836 (.0841) | .1639 / .1889 (.1764) |

Calibration fixes the reference false-warning budget at B=0.2, but does not guarantee realized held FWR is at most 0.2. The pooled realized FWR exceeds 0.2 for both seeds of SM13, the data-expansion candidate, the data labelled control, the adapter frozen control, and the adapter LoRA candidate. These are held diagnostics, not recalibrated operating points.

The data candidate has mixed results: lower Macro-3 AP for both seeds, higher F1 for seed 202 only, and a higher mean FWR than its labelled control. The adapter candidate is close to its matched frozen control, with a small AP increase, a small F1 decrease, and similar FWR. The temporal candidates score below the all-13 baseline in these runs. None of these 2-seed comparisons establishes a general causal or architectural claim.

## Held oracle ranking diagnostics

`pooled_held_oracle_precision_recall.csv` contains per-seed pooled classwise recall at requested precision minima 0.75 and 0.80 for S/O/I across all complete systems. It is explicitly marked `HELD_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE`: it selects thresholds using held labels to describe score ranking. Equal-score groups are kept intact, so ties are never split. This cannot be used as a deployable policy estimate.

## Reproducibility artifacts

- Pooled metrics, fold receipts, classwise metrics and per-seed summaries: [`shared/baseline_pooled_oof.json`](../shared/baseline_pooled_oof.json), [`shared/baseline_pooled_oof.csv`](../shared/baseline_pooled_oof.csv), [`shared/baseline_deltas.csv`](../shared/baseline_deltas.csv), [`shared/baseline_seed_summary.json`](../shared/baseline_seed_summary.json).
- Held oracle diagnostic: [`pooled_held_oracle_precision_recall.csv`](pooled_held_oracle_precision_recall.csv).
- Data-expansion detailed protocol and report: [`data_expansion/BENCHMARK_REPORT.md`](../data_expansion/BENCHMARK_REPORT.md).
- Adapter matched-control amendment and pair receipts: [`encoder_adapter/AMENDMENT_01_MATCHED_CONTROL.md`](../encoder_adapter/AMENDMENT_01_MATCHED_CONTROL.md) and `encoder_adapter/folds/matched_pair_seed{101,202}_fold{0..3}.json`.
