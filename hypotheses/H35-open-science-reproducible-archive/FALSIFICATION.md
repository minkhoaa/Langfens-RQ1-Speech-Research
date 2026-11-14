# Falsification Criteria: Hypothesis H35

## 1. Primary Falsification Gates
Hypothesis H35 will be considered **FALSIFIED** if:
1. **Numerical Drift on Re-run:** Re-executing any of the champion inference pipelines yields a discrepancy $|\Delta \text{AP}| \ge 10^{-4}$ or $|\Delta \text{Utility}| \ge 10^{-4}$.
2. **Missing Artifact Hashes:** Any of the 39 completed experiment directories lacks an `ARTIFACT_HASHES.sha256` manifest.
3. **Leakage Violation:** `test_sealed.jsonl` exhibits a checksum mismatch or file modification.

## 2. Inconclusive Conditions
1. Reproduction succeeds on GPU but shows minor floating-point divergence on CPU due to different BLAS libraries.
