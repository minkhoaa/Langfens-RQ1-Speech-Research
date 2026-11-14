# Falsification Criteria: Hypothesis H105

Hypothesis H105 will be declared **FALSIFIED** if:
1. The sealed test hash fails to match `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
2. Any of the 109 experiment directories are missing required auditable artifacts (`RUN_SPEC.yaml`, `RUN_STATE.json`, `TRAIN.log`, `REVIEW.md`, `ARTIFACT_HASHES.sha256`).
3. Recomputation from raw predictions reveals a discrepancy $\Delta > 0.001$ against published registry metrics for any champion.
4. Bootstrap 95% confidence intervals reveal overlapping lower bounds that negate verified milestone gains.
