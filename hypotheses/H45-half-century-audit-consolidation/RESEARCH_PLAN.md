# Research Plan: Experiment EXP-50 (Half-Century Milestone Audit & Clustered Bootstrap)

## 1. Audit Scope & Protocol
Execute a comprehensive milestone audit covering all 49 experiments completed in the laboratory:
1. **Sealed Holdout Integrity Audit:**
   - Verify SHA-256 hash of `data/derived/l2arctic/manifests/test_sealed.jsonl`.
   - Confirm zero exposure, zero tuning, and zero evaluation leakage.
2. **Cryptographic Artifact Manifest Verification:**
   - Scan all directories `experiments/EXP-01-*` through `experiments/EXP-49-*`.
   - Verify presence and valid format of `RUN_SPEC.yaml`, `RUN_STATE.json`, `TRAIN.log`, `REVIEW.md`, and `ARTIFACT_HASHES.sha256`.
3. **10,000-Sample Speaker-Clustered Bootstrap:**
   - Load utterance-level predictions and gold labels for champion models on `DEV_REPLICATION`.
   - Cluster utterances by speaker ID to preserve acoustic correlation structure.
   - Resample with replacement for $B = 10,000$ iterations.
   - Compute mean, standard error, and 95% percentile confidence intervals ($[\text{Q}_{2.5}, \text{Q}_{97.5}]$) for Macro AP, Macro F1, Macro Precision, Omission AP, Substitution AP, and Utility.
4. **Hypothesis Testing:**
   - Paired permutation tests ($B=10,000$) computing two-tailed empirical $p$-values.
5. **Consolidation Receipt:**
   - Output `HALF_CENTURY_AUDIT_RECEIPT.json` sealed with SHA-256 checksums.
