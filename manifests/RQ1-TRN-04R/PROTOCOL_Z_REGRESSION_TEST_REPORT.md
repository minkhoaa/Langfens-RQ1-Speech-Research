# RQ1-TRN-04R Protocol Z regression and final saved-row audit — Workstream D

## Test coverage

`tests/trn04/test_protocol_z_regression.py` contains 11 contract-level CPU tests. `tests/trn04/test_protocol_z_integrated.py` contains nine production API integration tests using A's runtime and actual `CompactFactorialOperationHead` with a tiny frozen encoder and synthetic waveforms, and B's metrics API. It verifies prompt-only gold-variant-inert raw logits, no runtime annotation keys, T1/T3 three-way focal semantics, old P-SUP B-Control null semantics, unmapped gold/extras, grouped-tie AP and dynamic FN, dedicated `s_omit` for CAL threshold and deployment without fallback, CAL/VAL runtime hashes before synthetic gold join, mismatch waveform/4D features, speaker disjointness and TEST denylist.

## Frozen run inspected

Actual output root: `/home/khoa/.cache/langfens/trn04r/genuine-z-v1-20261104T083210Z`. Parent source tip at run: `c60a84fb044f049dc30f61046f819ff0ac36dc72`; 4 parallel CPU FP32 arm jobs, two threads each, all exit 0, 396.2823s. Both fixed-budget and best endpoints were present for C0/T1/T2/T3. Frozen V1 contract SHA-256 was the expected `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.

I inspected the runner, source receipts, raw runtime/token/recognition records and freezes. All source hashes listed in run receipts matched current frozen source files. All eight output configurations passed SHA-256 checks; heads/backbone have identical before/after fingerprints; receipts report 0 backward and 0 optimizer updates. CAL allowed ID hash matches ordered NCC/THV 300-item selection; VAL matches NJS/RRBI 300 IDs. Runtime generated/hash-frozen before gold parsing by inspected source and receipts; runtime rows contain no annotation/gold/target fields. All eight cells have identical VAL gold keys/labels: 9,956 positions (8,799 correct / 1,012 sub / 145 deletion), 9,919 aligned, 37 unmapped (25/4/8 correct/sub/deletion), 37 prompt extras. CAL has 9,977 gold positions (8,054 / 1,462 / 461), 9,958 aligned, 19 unmapped and 24 extras. Mismatch mapping is a same-speaker 300-utterance derangement with no fixed points and preserved gold labels.

## Independent recomputation

Ran the read-only standard-library `audit_trn04r_saved_rows.py` against both metric files, saving outputs to `/tmp/d-final-audit-fixed-2.json` and `/tmp/d-final-audit-best-2.json`. It independently agrees on global/speaker/phone AP, CAL thresholds, deployed confusion counts/omission FN, PER, runtime hashes and gold-free row schema. Both endpoint outputs state `INDEPENDENT_RECOMPUTATION_AGREES`; metric SHA-256 values are fixed `9ae6c2f28a32c81484528b320dbd8a91dd5e6201b1a64e9e90ec2b4e398c9012`, best `5d3871683028a19efe66eea3e564d60170edf46cb19c4317fa296ca14317acce`. Per-arm CTC edits: fixed C0/T1/T2/T3 = 2361/2362/2361/2362; best = 2464/2790/2361/2465; each denominator 9,881.

Actual result summary and scientific limits, including the T2 feature-identity issue, are in `INDEPENDENT_SCIENTIFIC_AUDIT.md`. Key fixed-endpoint values: dedicated omission AP C0/T1/T2/T3 `.104690/.029928/.086464/.031706`; CAL15 recall `55.17/11.72/46.21/15.17%`; precision `7.03/2.74/6.39/4.10%`. No architecture winner or generalization claim is made.

## Critical unresolved evidence

Feature receipts: C0/T1/T3 share encoder feature cache hash `49e5ff48fba344f251ce3ba91c136ee6ab063b7e6388886a0eeb1469fd125d3a`; T2 hash differs (`33ef54dd5539aeb4916566875cb066d9963a19192e852d9d1a5cb7d019cd5840`) and encoder feature hashes differ for 464/600 utterances. Waveform hashes, acoustic feature hashes, shapes and frame counts match; backbone source weights/fingerprint match. Cause is unknown; T2 cannot be treated as a strictly matched feature ablation, and no causal acoustic claim is allowed. FIT feature-statistics receipt, FIT-only prior, and original source-training provenance remain unverified/unavailable. HT/event-horizon estimators were not independently validated. G=2, one seed, no population significance.

V1 outputs and independent recomputations remain preserved as genuine artifacts, but scientific acceptance is blocked. V1 logs also show `masked_spec_embed | MISSING` for all separate model loads; its relationship to T2 feature differences is not established.

V2 inference has completed on unchanged source `c60a84fb044f049dc30f61046f819ff0ac36dc72` using one process and a shared immutable encoder feature cache across all eight heads. Pre-metric audit verified source/output/runtime digests, before/after model fingerprints, no backward/optimizer work, freeze-before-gold, and all eight raw CAL/VAL cohorts. Shared cache receipt `a9744f39fc4ae36ab3c688cf6eaa1c1ab37b2dde730a1e6c5207fa718084bc56` has 600 IDs. V2 VAL each has 9,956 canonical rows; CAL each has the actual 9,977 (8,054 correct/1,462 sub/461 deletion), 9,958 aligned, 19 unmapped and 24 extras. V2 metric recomputation and independent saved-row audit remain pending; no V2 metric result or scientific PASS is claimed.

Audit probe correction, preserved: an initial scratch checker incorrectly expected runtime line counts to equal 300 utterances and CAL gold positions to equal VAL's 9,956. Those assertions failed due checker assumptions, not a data/runtime failure. Corrected checks count distinct 300 utterance IDs and actual CAL 9,977 canonical positions; corrected V1 and V2 raw checks passed.

## Commands and scope

- Parent safe test command: `cd /home/khoa/Projects/KLTN/Dataset && .venv/bin/python -m pytest -q tests/trn04/test_protocol_z_integrated.py tests/trn04/test_protocol_z_regression.py tests/trn04r` — **38 passed**.
- Read-only independent recomputation commands used the parent `.venv` and wrote only `/tmp` audit JSON. V2 pre-metric raw/source check: `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python /tmp/d_v2_pre_metric_audit.py` — passed for all eight configs; deliberately did not inspect or recompute V2 metrics while parent recomputation was pending.
- No D inference, training, backward, optimizer update, model mutation, privacy/publication change or TEST payload access. Only authorized frozen CAL/VAL outputs were audited.

Parent checkout was observed with an unstaged edit to `audit_scripts/audit_trn04r_saved_rows.py` after the tests/audits (path compatibility change); D did not edit it. No benchmark PASS is claimed; parent owns synthesis and private persistence.
