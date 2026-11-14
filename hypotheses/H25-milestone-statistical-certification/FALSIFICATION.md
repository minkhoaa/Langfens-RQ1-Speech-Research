# Falsification Criteria: Hypothesis H25

## 1. Primary Falsification Gates
Hypothesis H25 will be considered **FALSIFIED** if:
1. **Bootstrap Instability:** The 95% bootstrap confidence interval of Macro F1 spans a width $> 0.1000$ or falls below 0.5000 at the lower bound.
2. **Precision Milestone Invalidation:** The lower bound of the 95% CI for Macro Precision falls below 0.5800.
3. **Cryptographic Invalidation:** Any of the 29 experimental directories exhibits mismatched or missing `ARTIFACT_HASHES.sha256`.
4. **Leakage Violation:** `test_sealed.jsonl` exhibits a hash mismatch or evidence of access during training.

## 2. Inconclusive Conditions
1. Confidence intervals overlap substantially between Hexad-Pareto and the single-model baseline, failing to reject the null hypothesis of equal performance at $\alpha = 0.05$.
