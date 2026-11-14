# H-005 calibration — CPU artifact audit

Status: **EXISTING-EVIDENCE AUDIT ONLY; no new calibration experiment**. This audit is not a claim that H-005 passed. No files outside this H-005 namespace and its designated report were changed.

## Data and source facts

- The EVAL-V3 closure artifact `experiments/evaluation_v3/closure/confidence_recovered_v1/calibrator_recipes.json` contains canonical isotonic piecewise-linear mappings. It records fit split DEV_CAL, fit speakers ERMS/THV, and held-out assessment on DEV_SEARCH; it explicitly flags checkpoint-selection reuse of DEV_CAL. DEV_SEARCH is development-exposed, not an independent test.
- The associated `CAL_SEARCH_SHIFT.csv` has raw-native and calibrated-held-out-by-speaker reliability entries. This is existing closure output, not a new fit. In this artifact family, the registered calibrator method is isotonic; the file does not contain temperature- or beta-calibration comparisons.
- Population is 10,043 target events and 10,343 gaps on DEV_CAL (two speakers), and 19,755 targets / 20,354 gaps on DEV_SEARCH (four speakers). SEARCH event count is not four independent speaker replications.
- The published class-conditional isotonic fits are one-vs-rest maps. They are not a jointly normalized target simplex. Insertion is an independent gap task.

## Existing EXP-24 held-out reliability results

These numbers were read from canonical `CAL_SEARCH_SHIFT.csv`, not recomputed. `Brier_base` is the prevalence-constant baseline in that file. All values are DEV_SEARCH and remain development-exposed.

| Head | Brier raw | Brier isotonic | Brier prevalence baseline | ECE raw | ECE isotonic |
|---|---:|---:|---:|---:|---:|
| Substitution | 0.101525 | 0.077436 | 0.081487 | 0.165060 | 0.076090 |
| Omission | 0.016898 | 0.011893 | 0.010468 | 0.020753 | 0.015668 |
| Insertion | 0.009312 | 0.005327 | 0.006104 | 0.010242 | 0.002080 |

Interpretation: isotonic improved Brier and ECE versus raw for these three EXP-24 outputs on the held-out-by-speaker SEARCH population. For omission, the calibrated Brier remained worse than the prevalence-only baseline. This is evidence about reliability for this frozen mapping/checkpoint and exposed development cohort, not evidence of robust calibration on unseen populations. It does not establish an AP gain. A non-strict monotone mapping may create ties, so raw AP remains the discrimination reference unless calibrated-score AP is explicitly recomputed with the same event population and EVAL-V3 scorer.

## Why the requested broader comparison is not currently valid

- No event-level frozen TRAIN out-of-fold predictions with speaker-fold provenance were identified in this audit. In-sample TRAIN predictions from a checkpoint fitted on those speakers cannot stand in for speaker-held-out calibration validation.
- The inspected canonical recipe set does not include temperature, beta, or speaker-held-out TRAIN comparison mappings. The existing EVAL-V3 confidence reports cover isotonic and CAL-selected operating policies; they do not answer all of H-005.
- DEV_CAL has two speakers. Fitting and assessing isotonic on that same split would be in-sample, and choosing among calibrators there then reporting CAL fit metrics would overstate reliability.
- Existing SEARCH calibration measurements are not blind confirmation, and no per-speaker-specific mapper is justified for inference.

## Minimal next experiment

Once speaker-grouped out-of-fold frozen scores are available, compare raw, temperature, isotonic, beta, and regularized class-conditional calibration by outer held-out TRAIN speaker folds. Fit each map using only the other speaker folds, preserve independent one-vs-rest event semantics, and record per-fold score/label/event-ID hashes. Aggregate Brier, ECE10, log loss, reliability counts, precision at fixed predefined confidence thresholds, and risk–coverage across held-out speakers. Treat AP on raw scores as primary discrimination; separately calculate calibrated AP only to document tie effects. Freeze the selected mapping/procedure before DEV_CAL/SEARCH assessment. Do not fit per-speaker mappings using test-time gold speaker identity.

This requires genuine OOF score provenance; availability of those scores was **not established**. Temperature/beta fitting is CPU statistical work, but no corresponding output files were produced in this audit. Neural CUDA was not used.

## Hashes / reproducibility

No new prediction, mapping, or metric artifact was generated. Source artifacts are the EVAL-V3 closure files named above. SHA-256: `CAL_SEARCH_SHIFT.csv` `8787404db4659cc4b1541cce5c838a9d7c0d8d87273c89d328ab8b38951103ef`; `calibrator_recipes.json` `37914fc5cab1351fa7311f4421f29f8b6c997fb00c79007125dae151f043ce31`; evaluator source `76e41f906e5d953ee55cee8156379644bdd96bf96efab69e991b87d7f857d4a2`. Reproduction of this read-only extraction: read `CAL_SEARCH_SHIFT.csv`, select `model_id=EXP-24-FINAL-ROADMAP-EXPANSION`, `split=DEV_SEARCH`, heads `SUBSTITUTION`, `OMISSION`, `INSERTION`, and compare `RAW_NATIVE` with `CALIBRATED_HELDOUT_BY_SPEAKER`; confirm fit provenance in `calibrator_recipes.json`. EVAL-V3 version is 3.1 (`research/evaluation_v3/evaluator.py`).
