# Agent C benchmark preparation

Independent, CPU-only benchmark preparation for RQ1-METHOD-BENCH-01. This directory is separate from RQ1-UNIFIED-02. No CUDA job is started here.

## Comparison contract

- Use `research.evaluation_v3.evaluator.UnifiedEvaluatorV3` (EVAL-V3.1) for comparable event-level predictions.
- Join predictions to gold by canonical event identity, never row order. Targets are C/S/O; insertion is binary at each of the N+1 exact gaps, including leading and trailing gaps.
- Report ranking AP from continuous raw scores (`score_*`/`raw_score_*`), separately from thresholded predictions and any probabilities. Do not infer scores from labels.
- Report AP S, AP O, AP I exact-gap, their Macro-3 Error AP mean; per-class P/R/F1 and supports; Macro-3 Error F1; Macro-4 mixed-domain F1; binary MDD on targets; confusion matrices; class PR curves; Precision@K/Recall@K; realized FWR and false-alert rates.
- External papers' phone error rate, detection accuracy, FAR/FRR, sequence metrics, or binary F1 remain native-task metrics unless a defensible event-score mapping exists. Never manufacture exact-gap insertion outputs.
- DEV_SEARCH is development-exposed and supports exploratory comparisons only. No fitting, threshold choice, calibrator, model selection, or tuning may use its labels. Fit learned models on TRAIN; fit score calibration/policies on DEV_CAL only and freeze before SEARCH assessment.
- A new method is directly comparable only with complete, matched canonical population, event ontology, score semantics, and eligible inference inputs. Record non-comparability explicitly.

## Existing reference context

The EVAL-V3.1 student-only leaderboard reports seed-101 Macro-3 Error AP 0.236917 for GOLD-ONLY, 0.240230 for KD-ALL, and matched seeds 101/202/303. Those artifacts are context only, not substitutes for a matched rerun. The project protocol reports DEV_SEARCH has 599 utterances, 4 speakers, 19,755 targets, and 20,354 gaps; this is not a blind test.

## Intake checklist for each method

1. Record paper/source, version/date, code/checkpoint and license.
2. Label implementation `FAITHFUL_REPRODUCTION`, `ADAPTED_REPRODUCTION`, `INSPIRED_IMPLEMENTATION`, or `NEW_RESEARCH_VARIANT`.
3. Store immutable config, seed, predictions, evaluator JSON, logs, resource cost, hashes, and reproduction command.
4. Verify prediction event coverage and prohibited gold fields before evaluation.
5. Report class-wise deltas against the matched control and whether evidence is acoustic, prior-driven, or indeterminate.

See `../../../../reports/METHOD_BENCH_01_LEADERBOARD.csv` for the row schema.
