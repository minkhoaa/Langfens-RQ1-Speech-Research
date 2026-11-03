# P-SCORE — output exchange, implementation tasks and CPU test plan

Design only. No scorer, test code, emissions, predictions or real-data output created. Shared V1 schema is authoritative; arm fields below extend it without changing populations/evaluator semantics.

## Planned private output bundle

`<approved-private-store>/<run_id>/p-score/` contains `run_manifest.json`, `runtime_predictions.jsonl`, `offline_token_evaluation.jsonl`, `utterance_predictions.jsonl`, `metrics.json`, `metrics_by_speaker.csv`, `failures.jsonl`, and SHA256SUMS. These are future paths, not existing evidence. Git receives lightweight registration/schema/audit hashes only.

Runtime row adds required `scorer_id`, `variant` (binary/greedy_dp/gop_sa/sf_s/sf_sd), `normalization` (raw/occupancy), `graph_revision`, `emissions_sha256`, `canonical_log_likelihood`, `relaxed_log_likelihood`, `raw_gop`, `occupancy_estimate`/floor or null, `s_error`, optional named `s_omit`, `dp_gap`/margin or null, valid-frame count and status. Scores are full precision finite numbers or null+failure reason; no gold fields. Cache records encode phone/blank map and preprocessing/weight fingerprints. Graph position is `prompt_index`, not a phone dictionary key. Repeated phones cannot overwrite each other.

Offline join adds strict label/masks, gold index, prompt mapping and coverage status only after runtime file hash is frozen. Each condition must retain all allowed utterances and gold canonical keys including failures; extras/mapping gaps separately typed. Metrics identify **which score** was used, protocol/population, N/positives, AP/tie convention, operating threshold provenance, TP/FP/TN/FN, achieved correct FPR, precision, recall, PER edits/reference count and coverage. No rounding before serialization. `failures.jsonl` covers every failed utterance/position with stage/reason and artifact identity; not a silent log-only exclusion.

Schema example is intentionally not a prediction: field types are string IDs/hash/version, integer indices/counts, enum variant/status, bool validity and number-or-null scores. Do not insert fake numeric results or gold-derived timestamp into runtime examples.

## Pending implementation sequence (requires new operator approval)

1. Resolve requested missing proposal or explicit accepted limitation; read pinned primary mathematical formulation; clear reuse license or register an independently authored specification. Freeze topology, neighbor restriction, path multiplicity, priors, occupancy and sign conventions.
2. Design isolated scorer/graph/emission-adapter modules and new tests. Do not patch `evaluate_e1_mdd.py`, baseline training or historical metric helpers. If a defect is found, append an erratum and stop for scope review.
3. Implement tiny synthetic graph enumeration oracle and log-space reference first. Prove variant graph paths/terminal semantics before optimized scaled forward recursion.
4. Add S and SD paired scorers, raw and explicitly occupancy-normalized outputs; no learned calibration/fusion. Implement fixed greedy collapse+DP and self-alignment controls without annotation reads.
5. Add exchange-schema/coverage validator and frozen-prediction offline join. Verify bit-for-bit source/weight/cache identity across controls; implement fair denominator reporting.
6. Under a further real-data authorization only, verify private artifact availability and package membership; capture model/source/precision/cache provenance and matched E0 replay. Freeze predictions before offline labels; no TEST.
7. Evaluate registered contrasts, controls, speaker/coverage uncertainty and feedback limitations. Report all failures and inconclusive results; do not promote on a monotone score/threshold change alone.

## Execution Status (2026-11-03)

- **Synthetic CPU Test Matrix**: **PASSED** (269/269 tests passed).
- **Frozen E0 Inference**: **COMPLETED** on 300 Fold-1 VAL utterances (NJS & RRBI) under `AUTH-RQ1-P-SCORE-E0-FOLD1-VAL`.
  - Emission cache: `/home/khoa/Projects/KLTN/Dataset/cache/rq1_p_score_fold1_val_emissions.json` (SHA-256 `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829`).
  - Runtime records generated under `rq1-runtime-v1` schema across all 4 scorers (`sf_s`, `sf_sd`, `greedy_dp`, `gop_sa`), validated via `strict_runtime_validator.py`.
- **Offline Join & Benchmark**: **COMPLETED** in `manifests/RQ1-P-SCORE/evaluation/`.
  - Primary contrast delta (SD - S) on deletion AP: **+0.021615** (0.051026 vs 0.029411).
  - Recall @ FPR15 delta: **+0.182482** (43.07% vs 24.82%).
  - Baseline R CTC PER reproduced: **0.236818** on 9,881 references.
