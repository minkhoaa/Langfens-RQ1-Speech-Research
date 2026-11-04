# LANGFENS-RQ1-TRN-04R — corrective campaign checkpoint

## State: IN_PROGRESS — implementation and independent audit lanes launched

- Original HEAD verified: `d332997e732315292d673a3bd42a03698182f5f6`, original branch `review/rq1-trn-04-factorial-training`; initially clean.
- Corrective coordinator branch: `review/rq1-trn-04r-genuine-z-rebenchmark`.
- Frozen contract SHA-256 independently verified: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Read original TRN-04 process/registration and recovered P-SUP final-v2 process checkpoint. Historical TRN-04 conclusions are under review, not accepted as corrected evidence.
- Four native parallel Luna workstreams launched in workflow `4fba5878-50b7-4bbf-9cb9-50247e9cd050`, mission `e218a5ae-c8a7-435f-b27d-4947b7c922eb`.
- Isolated sibling worktrees `/home/khoa/Projects/KLTN/trn04r-{a,b,c,d}` on `review/trn04r-{a,b,c,d}`, starting at original HEAD. A owns new runtime; B independent metrics; C artifact provenance; D independent tests. Parent owns integration, global index, benchmark execution, durable preservation and final scientific acceptance.
- Local fixed-budget and best artifacts for all four arms were observed by filesystem inventory; their bytes/counters and remote identity await C verification. No scientific metrics recomputed yet.

## Execution boundaries

Evaluation-only user authorization. CPU FP32 frozen inference permitted after corrected source freeze. No new training, backward pass, optimizer attempt/update/skipped update, model artifact, TEST payload access, VAL threshold tuning, historical rewrite, publication, push or merge. Existing 3,600-update counters are historical, not new optimization. Last valid model identity remains pending byte verification.

## Integration checkpoint — revisions required before inference

All initial four lanes completed and their owned commits were cherry-picked (coordinator HEAD `eca2cba`). Coordinator review identified runtime mismatch join key inconsistency, absent offline CTC references, null gold-field names in runtime, inconsistent score summation precision, and binary/substitution confusion and speaker PER bugs. Independent tests still targeted an old serializer rather than corrected production code. **Final benchmark NOT_STARTED; integration gate NOT_PASSED.** No failed benchmark outputs exist.

Native retained resumes: A `ee969a99-fa2a-4972-b285-a520c6885814`, B `25364de1-520f-49f3-b1f0-5c2000da700e`, C `bb94c6fd-e0f5-4d2c-bd3e-e4087cb1d49c`, D `3fff2925-8e48-42e7-a163-f91945a80090`. Each remains sole writer in its own worktree; parent owns coordinator checkout. C initial evidence matches eight local checkpoints plus four summaries to pinned private remote revision `af6ac3620fb78873ff213c064dffe71fb8acadd4`; historical training-source commit remains unverified. Exact SELECT-best successful counters are 2247/897/3600/2247 (C0/T1/T2/T3), not rounded historical claims.

## Corrected implementation freeze gate

All four lane follow-ups integrated. Coordinator corrected the raw-recognition phone-string exchange field and recorded before/after parameter fingerprints plus deterministic-algorithm settings. Scoped production integration/regression command executed successfully: `.venv/bin/python -m pytest -q tests/trn04/test_protocol_z_integrated.py tests/trn04/test_protocol_z_regression.py tests/trn04r` — **38 passed**. These are no-training synthetic tests; no benchmark numbers are implied. Runtime/metrics/auditor compilation and `git diff --check` passed. Parent separately rehashed all eight endpoint files, four summaries, authorized archive, prompt JSON and dictionary against their frozen identities.

Safe next step: commit this corrected revision with a clean working tree, then execute four independent parallel CPU FP32 arm processes (2 threads each, fixed+best), each freezing its own CAL/VAL matched/mismatched runtime before annotation access. No source changes during inference. This per-process gold boundary is isolated: one process's offline join cannot influence another process's independent frozen model/runtime. Saved raw predictions and logs go outside Git; model-free B recomputation and independent standard-library audit follow. Final scientific verdict remains pending actual measurements and durability verification.

## Safe restart

Read global index and lane checkpoints. Verify coordinator/lane HEADs and diffs; collect native completion artifacts rather than poll. Integrate only non-overlapping reviewed changes, run safe no-training tests, freeze corrected source in a reviewable commit, then execute benchmark into a new private cache directory. Independently recompute saved rows, verify immutable private SHA-256 readback, and only then finalize conclusions. Never reuse incomplete outputs as complete; do not overwrite original TRN-04 artifacts.
