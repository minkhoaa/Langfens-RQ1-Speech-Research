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

## Synthetic CPU test matrix — PLANNED, NOT RUN

No model construction/download/forward pass, corpus readers, tensors with autograd, optimizer or backward in this test tranche. Use small handcrafted probability matrices and phone lists only.

- Exhaustively enumerate short CTC paths and compare canonical/S/SD forward sums (FP64 reference tolerance abs 1e-10/rel 1e-8); deterministic fixture topology and terminal acceptance.
- Position-specific target deletion, first/last/single target, empty left/right/canonical contexts and zero valid frames; expected explicit failure on unsupported cases, not NaN.
- Adjacent identical labels AA/AA, ABBA and longer runs; repeated label separated/not separated by blank; target equal to next/previous phone; no accidental neighbor deletion or duplicate path counting.
- Impossible length (`T<U+required repeat blanks`), all blank, exact zeros, near-deterministic posteriors, tiny probabilities/long sequences. Log-space vs scaled FP64 equivalence; future FP32 tolerance prospectively registered, never widened after failure.
- Occupancy 0/near0/1 and floor=1; raw output not mislabeled normalized; numerical forward scaling never confused with occupancy division.
- When path-set inclusion holds, canonical≤S≤SD probability; identity and equal-graph cases. Failure is a graph/spec issue, not permission to clip ratios.
- Greedy CTC collapses repeats before removing blanks; DP deterministic ties and repeated-position trace; prompt mapping edits distinct from acoustic edits.
- Schema rejects nonfinite/duplicate/missing records, incorrect mapping/masks, phone/blank mismatch, TEST IDs, gold runtime fields, wrong contract/cache hash and unequal candidate populations. Fixed grouped-tie AP and integer confusion/PER fixtures; no historical semantics edits.
- Coverage includes all failures/abstentions; scorer order/batch/padding cannot change valid outputs; byte-stable serialization/hash on repeated fixtures.

Passing future synthetic tests would establish bounded implementation correctness, NOT acoustic omission accuracy, calibration, runtime cost, feedback quality or authorization to run real data. No test has been executed in this setup.
