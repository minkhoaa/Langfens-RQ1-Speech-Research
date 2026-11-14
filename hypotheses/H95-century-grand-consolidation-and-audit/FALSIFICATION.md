# Falsification Criteria: Hypothesis H95

Hypothesis H95 will be declared **FALSIFIED** if:
1. The sealed test hash `test_sealed.jsonl` fails to match the pristine checksum `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
2. Any of the 99 prior experiment directories are missing required auditable artifacts (`RUN_SPEC.yaml`, `RUN_STATE.json`, `TRAIN.log`, `REVIEW.md`, `ARTIFACT_HASHES.sha256`).
3. Recomputation from raw predictions reveals a discrepancy $\Delta > 0.001$ against published registry metrics for any champion.
4. Autonomous dispatch fails to pause at EXP-100 without launching EXP-101.
