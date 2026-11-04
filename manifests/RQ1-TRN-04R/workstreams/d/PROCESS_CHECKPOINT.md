# RQ1-TRN-04R Workstream D checkpoint

**Status: SYNTHETIC_PROTOCOL_TESTS_PASS; INTEGRATED_BENCHMARK_GATE_PENDING.**

- Base: `d332997e732315292d673a3bd42a03698182f5f6`; lane `review/trn04r-d`.
- Read project `AGENTS.md`, global process checkpoint registry, RQ1-TRN-04 master checkpoint, P-SUP final-v2 checkpoint, and frozen V1 evaluation contract. Contract SHA-256 verified as `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Audited legacy `audit_scripts/evaluate_protocol_z.py`, legacy Protocol Z tests, and recovered P-SUP runtime/dataset/evaluator API in sibling worktree. New A runtime/B metrics were not available in this lane. No source changes made outside owned regression tests and reports.
- Initially added 11 synthetic/no-training tests; that suite passed 11/11. Follow-up coordinator defect adds a 12th test requiring class_focal three-way semantics and binary_collapsed null semantics. It was run against the available serializer and failed as expected: class_focal currently returns null subtype scores. A has been notified; preserve old files. This is a pending integration gate, not a passing regression.
- Acoustic-feature constants' FIT provenance lacks an inspected receipt and is explicitly unverified. Verify frozen/hash-sealed outputs before gold joins for both CAL and VAL.
- Coverage and scientific qualifications are described in `PROTOCOL_Z_REGRESSION_TEST_REPORT.md` and `INDEPENDENT_SCIENTIFIC_AUDIT.md`.
- No real data/model inference, training, backward pass, optimizer/model mutation, CAL/VAL tuning, TEST payload access, privacy change, or publication occurred. No measured benchmark result is reported.
- Safe restart: coordinator integrates A/B and freezes source; rerun `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/trn04/test_protocol_z_regression.py` and extend/run integration only against available frozen APIs. Then independently inspect frozen outputs/manifest for complete prompt+gold accounting, no runtime gold fields, CAL-only thresholds, score semantics, mismatch pairing, order and speaker/test isolation. Do not read sealed TEST payload. Do not call benchmark PASS before frozen source and outputs.
