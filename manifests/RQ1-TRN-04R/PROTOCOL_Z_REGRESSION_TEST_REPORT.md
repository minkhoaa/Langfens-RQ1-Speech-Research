# RQ1-TRN-04R Protocol Z regression tests (Workstream D)

## Scope and status

Synthetic/no-training tests only; no real-data benchmark. Frozen contract inspected: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Historical runner inspected: `audit_scripts/evaluate_protocol_z.py`. A and B APIs were read-only from sibling commits A `6cd63f6d5da3986119192099a0254323bd9124da`, B `d85f90ca49aeac10364fb3cbb273c0a05e29ae07`; no sibling source was modified. API modules were exposed to tests through a temporary package-path overlay in `/tmp`.

## Regression coverage

`tests/trn04/test_protocol_z_regression.py` contains 11 contract-level CPU tests: prompt-only deterministic stand-in with counterfactual gold; legacy P-SUP B-Control null subtype semantics; gold fields/nonfinite/sealed speaker checks; prompt coverage; deterministic alignment and unmapped positions; 145-deletion denominator arithmetic; grouped-tie AP; CAL-only score-array fixture; CTC-recognition versus learner-omission distinction; mismatch condition holding prompt fixed; and repeated-position ordering.

`tests/trn04/test_protocol_z_integrated.py` adds nine integration checks against A's `langfens_prep.trn04.runtime` and actual `CompactFactorialOperationHead`, and B's `langfens_prep.trn04r.metrics`. A tiny frozen deterministic encoder plus synthetic waveforms are used. Tests cover changed synthetic gold variants with unchanged raw model logits; runtime rows containing no annotation/target key names; T1/T3 three-way focal semantics; legacy B-Control collapsed semantics; unmapped-gold and extra-prompt row accounting; production grouped-tie AP and dynamically computed FN; dedicated focal s_omit for both CAL threshold and deployment counts with no fallback; CAL/VAL runtime JSONL hash before synthetic gold join; waveform/4D-feature mismatch; speaker disjointness and sealed TEST rejection.

Focal assertion targets the new A runtime, not the old P-SUP serializer. Old P-SUP serializer remains only as historical B-Control reference. No training/backward/optimizer/model fitting occurs; the tiny head's eval forward is deterministic CPU inference on synthetic data.

## Validation and remaining integration gates

With temporary outside-repo package overlay, combined command:
`PYTHONPATH=/tmp/trn04r-d-api-overlay:/home/khoa/Projects/KLTN/trn04r-a/src:/home/khoa/Projects/KLTN/trn04r-b/src:/home/khoa/Projects/KLTN/trn04r-d/src /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -c /dev/null -q tests/trn04/test_protocol_z_regression.py tests/trn04/test_protocol_z_integrated.py`
Result: **20 passed** (one pytest cache warning due `-c /dev/null` cache path, no test failures). Temporary overlay and synthetic fixtures do not constitute final integrated parent-worktree test evidence.

Legacy `src/langfens_prep/trn04/evaluator.py` around lines 382/383 falls back from missing dedicated omission/substitution AP to legacy AP while retaining the dedicated metric name. Around line 415 deployment selects `s_omit` if present, else `s_error`, but its CAL omission threshold is based on `p_deletion`/`s_omit` (around lines 165–171). Therefore historical T1/T3 dedicated AP/alarm comparisons can mix score families; do not infer corrected empirical improvement until a corrected frozen benchmark is rerun and independently audited. The synthetic B regression proves only correct score-family routing in the tested metric API, not an empirical effect.

B `calibration_threshold` accepts score arrays and `summarize_cell` currently does not validate calibration speaker membership. The test supplies NCC CAL and NJS VAL but does not claim the production API rejects invalid CAL speakers. Explicit CAL partition identity validation remains an API/coordinator gate. FIT acoustic-feature constants state they derive from 1,799 FIT utterances, but no derivation receipt was found in inspected manifests; provenance remains unverified. Runtime docs prescribe hash-before-gold and the synthetic test writes/hashes both CAL and VAL rows before gold joins, but actual benchmark orchestration order and saved outputs remain coordinator review gates.

No real data/model checkpoint, dataset archive, waveform payload, TextGrid, VAL tuning, or TEST payload was accessed. No publication/privacy change occurred.
