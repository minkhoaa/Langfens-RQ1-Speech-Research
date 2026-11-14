# Falsification Criteria: Hypothesis H32

## 1. Primary Falsification Gates
Hypothesis H32 will be considered **FALSIFIED** if:
1. **Manifest Inconsistency:** Any of the 36 experiment directories fails cryptographic SHA-256 verification or contains corrupted state logs.
2. **Missing Deliverables:** Any mandatory scientific artifact specified in `AGENTS.md` (e.g., Pareto frontier, compute ledger, scorecard, leakage audit) is omitted or mathematically disconnected from raw prediction records.
3. **Sealed Holdout Violation:** `test_sealed.jsonl` exhibits a hash mismatch or unsealing violation.

## 2. Inconclusive Conditions
1. Theoretical derivations fail to explain the observed numerical deltas between APED-Net and PNC-Net.
