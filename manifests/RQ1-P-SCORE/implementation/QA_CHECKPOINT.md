# P-SCORE QA checkpoint

- Reviewed branch `prep/rq1-p-score`, source tree tested at `e9a97c620b7cf4e01284583ef5fee2cfdd35dc4e`, frozen merge-base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`.
- Ran the listed P-SCORE CPU/offline suite: 251 passed; P-SCORE CLI `--help` passed; shared JSON-only tests 22 passed; safe inherited synthetic tests 9 passed. All 57 shared source/input/archive fingerprints and frozen V1 hash verified. Shared synchronized artifacts are byte-identical.
- **QA blocker:** shared schema rejects arm emitted key `raw_scores.posterior_target_occupancy`; `target` substring violates strict metadata-key rule. Arm-local tests don't cover shared-validator integration. No arm code was changed by QA.
- Prototype readiness: **HOLD pending schema-compatible field and regression**. Experiment HOLD for missing proposal, author source/license/topology parity, criteria and cache provenance/authorization. No AP/PER evaluator implemented.
- Training/inference NOT_STARTED: 0 successful, 0 skipped, 0 attempts, no artifact. No real data/audio/labels/timestamps/weights/TEST/GPU/remote/API used. Resource estimates are qualitative only.
- Restart: correct output metadata name, add strict shared-validator bridge test, rerun specified CPU/offline commands; capture new source SHA before considering implementation integration complete. See `INDEPENDENT_QA.md` and shared QA report.
