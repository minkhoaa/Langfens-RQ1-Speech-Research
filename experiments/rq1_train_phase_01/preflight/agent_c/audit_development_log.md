# Agent C audit-run ledger

All Python execution was CPU-only with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`; torch threads were two where imported. No model fits or output-producing legacy mains were invoked.

| Attempt | Command/artifact | Exit | Result |
|---|---|---:|---|
| Initial full-WAV check | `independent_train_audit.py` | 1 | Python `wave` rejected TRAIN float WAV format tag 3. Switched to read-only `soundfile.info`; no WAV/output written by failed attempt. |
| Initial population assertion | `independent_train_audit.py` | 1 | Incorrectly asserted target/gap event IDs disjoint, although same `uid::index` string is valid in separate domains. Removed assertion; domain arrays remain separate. |
| Alias source assertion | `independent_train_audit.py` | 1 | Test expected E4 source to repeat literal `SPN`; E4 takes dynamic class inventory. Replaced with checks for inventory source, `classid[p]`, explicit alias map and identity-preserving normalizer. |
| Policy metric script syntax | `independent_train_audit.py` | 1 | Missing dictionary commas found by compile/shell run; corrected before freezing successful metric run. |
| Freeze guard first retry | `independent_train_audit.py` | 1 | Freeze had a script path in input hashes with pre-amendment hash. Amended freeze with corrected frozen script hash before rerun; no policy/outer metric was emitted by this failed attempt. |
| Frozen audit | `independent_train_audit.py` with log redirect | 0 | Full lineage and policy audit passed after the final audit freeze; see `INDEPENDENT_AUDIT_RECEIPT.json` and `independent_audit_run.log`. |
| A synthetic runner suite | `agent_a/test_runner.py` | 1 | 4 passed, 2 failed due current test fixture/expectation issues described in scientific audit; separate Agent C exact N+1 feature-boundary check passed. |
| B policy tests | `agent_b/test_joint_policy.py` | 0 | 6 passed (wrong-type, speaker/exclusion, exact partial SHA tie, duplicate IDs/NaN, nested union/coverage). |
| Agent C N+1 boundary | `test_gap_boundary.py` | 0 | First/interior/trailing acoustic+phone inputs verified for N=3 → N+1=4. |
