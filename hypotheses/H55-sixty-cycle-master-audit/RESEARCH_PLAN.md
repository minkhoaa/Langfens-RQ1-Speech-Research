# Research Plan: Experiment EXP-60 (Sixty-Cycle Master Audit & Clustered Bootstrap)

## 1. Audit Scope & Methodology
Execute a comprehensive master audit covering all 59 experiments completed across the laboratory's 50+ research cycles:
1. **Sealed Holdout Integrity Audit:**
   - Verify SHA-256 hash of `data/derived/l2arctic/manifests/test_sealed.jsonl`.
   - Confirm zero exposure, zero tuning, and zero evaluation leakage.
2. **Cryptographic Manifest Audit:**
   - Scan all directories `experiments/EXP-01-*` through `experiments/EXP-59-*`.
   - Verify presence and valid format of `RUN_SPEC.yaml`, `RUN_STATE.json`, `TRAIN.log`, `REVIEW.md`, and `ARTIFACT_HASHES.sha256`.
3. **10,000-Sample Speaker-Clustered Bootstrap:**
   - Load utterance-level predictions and gold labels for champion models on `DEV_REPLICATION` (300 utterances across independent speakers).
   - Cluster utterances by speaker ID to preserve acoustic correlation structure.
   - Resample with replacement for $B = 10,000$ iterations (evaluating 1,000 full stratified resamplings).
   - Compute mean, standard error, and 95% percentile confidence intervals ($[\text{Q}_{2.5}, \text{Q}_{97.5}]$) for Macro AP, Macro F1, Macro Precision, Omission AP, Substitution AP, and Utility.
4. **Hypothesis Testing:**
   - Paired permutation tests ($B=10,000$) computing two-tailed empirical $p$-values between Duodecim-Pareto and earlier baselines.
5. **Deterministic Reproduction Check:**
   - Run deterministic reproduction check on `tools/stream_edge_infer.py`.
6. **Consolidation Receipt:**
   - Output `SIXTY_CYCLE_MASTER_AUDIT_RECEIPT.json` sealed with SHA-256 checksums.
