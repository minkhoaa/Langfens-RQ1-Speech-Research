# RQ1-METHOD-BENCH-02 — Agent C comparative analysis

## Scope and protocol

This ledger reuses existing baseline predictions; it does not fit or select on DEV_SEARCH. DEV_SEARCH figures below are descriptive and development-exposed. Exact target event keys were reconstructed from the official split manifests and joined against the candidate JSONL predictions. Target scores and exact-gap scores are separate event domains. The two phone-recognition baselines and discrete-unit surprisal artifact have no insertion-gap predictions, so their exact-gap insertion AP and Macro-3 Error AP are not estimable and they are not complete EVAL-V3.1 system comparisons.

## DEV_CAL manifest discrepancy

The official `data/derived/l2arctic/manifests/dev_cal.jsonl` contains **300 unique utterances, 2 speakers, 10,043 target events and 10,343 exact gaps**. All 300 utterances satisfy `num_gaps = num_target_phones + 1`, and every declared per-utterance count matches the actual event arrays. The exact target and gap event keys are unique. The manifest SHA-256 is `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6`.

Therefore the `9,744` DEV_CAL target count in `reports/RQ1_UNIFIED_02_BAO_CAO_TONG_HOP_CHI_TIET.md` was a report typo, not a population discrepancy. Its gap count, insertion-positive count (102) and rate were already consistent. The report now says **10,043**. For context, official TRAIN has 59,562 targets / 61,362 gaps; DEV_SEARCH has 19,755 / 20,354. There are zero N+1 mismatches in all three splits. Audit counts and hashes are in `experiments/rq1_method_bench_02/agent_c/manifest_audit.csv`.

## Existing baseline comparison

| Artifact | DEV_SEARCH target coverage | AP S | AP O | Binary error AP | Exact-gap I AP | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Soft phone recognition, hard-target arm | 19,755 / 19,755 | 0.36356 | 0.15360 | — | Not estimable | Target-only; no insertion-gap output |
| Soft phone recognition, similarity-soft arm | 19,755 / 19,755 | 0.36214 | 0.14672 | — | Not estimable | Target-only; no insertion-gap output |
| Discrete-unit surprisal | 19,755 / 19,755 | — | — | 0.14318 | Not estimable | One binary target anomaly score; no class diagnosis or gap output |
| EVAL-V3 seed-101 GOLD-ONLY reference (existing leaderboard) | 19,755 targets; 20,354 gaps | 0.36460 | 0.15943 | — | 0.18672 | Complete target+gap score reference; Macro-3 Error AP 0.23692 |

The first three rows are not head-to-head full-task results. Hard phone recognition is the closest class-score baseline: it is near the existing GOLD-ONLY S AP, below it on O AP, and cannot be credited on I. The similarity-soft arm does not improve either target-class AP over its hard-target sibling in this artifact. The discrete-unit artifact supports only binary target-error ranking, and its AP is not directly interchangeable with S/O diagnosis AP.

## Post-hoc error complementarity

For a descriptive complementarity check, each target scorer ranked its own frozen score over the same 19,755 target event IDs. Phone models used `max(score_substitution, score_omission)`; discrete surprisal used its `binary_md_score`. Labels were read only after ranking to count error events in top-K. No threshold, model, weight or hyperparameter was fitted from DEV_SEARCH labels.

| Candidate score set | Error events in top 100 | Union size | Error events in union |
|---|---:|---:|---:|
| Hard phone | 57 | 100 | 57 |
| Similarity-soft phone | 58 | 100 | 58 |
| Discrete-unit surprisal | 30 | 100 | 30 |
| Hard phone + surprisal | — | 199 | 86 |
| Similarity-soft phone + surprisal | — | 199 | 87 |
| All three | — | 229 | 99 |

At top 200, hard phone, similarity-soft phone and surprisal capture 111, 107 and 58 error targets, respectively; the three-way union covers 178 errors among 435 unique alerts. This indicates non-identical rankings, with surprisal adding some distinct flagged errors. It is a candidate-generation observation only: union sizes exceed each fixed K, and this does not show that fusion improves precision, AP, exact-gap insertion detection or a deployable operating point. Any later fusion rule must be specified and selected on permitted training/calibration data, frozen, and then assessed once.

## Artifact identity and reproduction

- Official manifests: `data/derived/l2arctic/manifests/{train,dev_cal,dev_search}.jsonl`; hashes and N+1 audit in `experiments/rq1_method_bench_02/agent_c/manifest_audit.csv`.
- Baseline predictions: `experiments/rq1_method_bench_01/agent_b/soft_phone_recognition/{hard,similarity_soft}_dev_search_target_predictions.jsonl` and `experiments/rq1_method_bench_01/agent_b/discrete_unit_surprisal/dev_search_target_predictions.jsonl`.
- Machine-readable scores and top-K overlap counts: `experiments/rq1_method_bench_02/agent_c/benchmark_ledger.csv`.
- Reproduce with `.venv/bin/python experiments/rq1_method_bench_02/agent_c/build_ledger.py` from the repository root. This only reads official manifests and existing predictions and writes the two ledger CSVs.

## DEV_CAL-selected operating policies for H05

The three H05 optional-path prediction bundles have complete target and exact-gap probabilities on DEV_CAL and DEV_SEARCH. I applied the closure EVAL-V3.1 `CAL_MAX_ERROR_MACRO_F1` implementation unchanged (source hash matches `experiments/evaluation_v3/closure/adapter_sources/67aee329ed6aee8626ed31f44f328b60876033a3ae3988aee04f1e6209a473b3/policies.py`). Each arm selected its three thresholds on DEV_CAL using the predeclared finite grid and actual post-arbitration error Macro-3 F1; those thresholds were frozen before DEV_SEARCH scoring. The thresholds and CAL selection receipts are in `experiments/rq1_method_bench_02/agent_c/h05_operating_policy_metrics.csv` and `h05_cal_policy_receipt.json`.

| Frozen arm | S threshold | O threshold | I threshold | SEARCH S P/R/F1 | SEARCH O P/R/F1 | SEARCH I P/R/F1 | Macro-3 error F1 | Macro-4 F1 | False warnings | Joint FWR |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| H05-A forced Viterbi | 0.272736 | 0.054039 | 0.563435 | .2435 / .6917 / .3602 | .1586 / .2201 / .1844 | .2193 / .3280 / .2628 | .2691 | .4162 | 4,190 | 6.9950 |
| H05-B optional phone | 0.298707 | 0.160842 | 0.563435 | .2614 / .6635 / .3750 | .1405 / .2871 / .1887 | .2193 / .3280 / .2628 | .2755 | .4241 | 3,828 | 6.3907 |
| H05-C single-gap optional | 0.329358 | 0.165736 | 0.563435 | .2796 / .6261 / .3866 | .1165 / .2967 / .1673 | .2193 / .3280 / .2628 | .2723 | .4243 | 3,468 | 5.7896 |
| H05-A targets + frame-decoded I | 0.272736 | 0.054039 | count > 0 | .2435 / .6917 / .3602 | .1586 / .2201 / .1844 | .0501 / .2080 / .0807 | .2084 | .3707 | 4,537 | 7.5743 |

For A/B/C, SEARCH values are the result of DEV_CAL-fitted policy selection; FWR is false warnings per utterance, not a rate bounded to one. H05-C's selected operating point is the one in this table; any other historical fixed-policy value, including an O F1 around .009, belongs to a different policy/threshold. It should not be compared as the result of this DEV_CAL selection.

The frame-decoded insertion artifact contains only DEV_SEARCH raw extra-phone counts, with no DEV_CAL count vector. Its count is not a probability, so it was not passed to the probability policy or given a learned CAL threshold. The displayed hybrid reuses H05-A's CAL-selected S/O thresholds and applies the frame decoder's predeclared `count > 0` insertion decision. The frame decoder records counts only for unambiguous extra-phone mappings; the resulting I operating point is exploratory and its insertion threshold is **not CAL-selected**. It has I precision .0501, recall .2080, F1 .0807 and increases joint FWR to 7.5743.

## Class-wise precision-recall curves

Event-level DEV_SEARCH precision-recall points are exported in `experiments/rq1_method_bench_02/agent_c/classwise_pr_curves.csv` (376,191 rows plus header). It includes H05 A/B/C S/O target and I exact-gap curves; H05 frame-decoded I using raw gap counts; H01 hard and similarity-soft target scores; the OTTC adaptation's available target scores; and discrete-unit unigram/bigram binary target-error scores. The curves use frozen scores and post-hoc DEV_SEARCH labels only for assessment; no threshold or model was fit from them. The `method`, `class`, and `score_semantics` columns distinguish target classes, exact-gap I and binary target-error rankings.

The SylPh-inspired prototype remains unsupported for curve reconstruction because its available artifact is an aggregate AP summary without event-level score vectors. H01, OTTC and discrete-unit artifacts are target-only; they have no exact-gap score and cannot support I AP or complete Macro-3 Error AP. No missing class curves were inferred.

Reproduce the policy receipt and curve export with `.venv/bin/python experiments/rq1_method_bench_02/agent_c/h05_cal_operating_and_pr.py`. The script checks exact event ID and array ordering against official manifests, calls the unmodified closure-matching policy, and writes the metrics, receipt and curves under `experiments/rq1_method_bench_02/agent_c/`.
