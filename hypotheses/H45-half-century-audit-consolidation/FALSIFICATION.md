# Falsification Criteria: Hypothesis H45

Hypothesis H45 will be declared **FALSIFIED** if any of the following audit failures occur:

1. **Bootstrap Milestone Failure:**
   - The lower bound of the 95% speaker-clustered bootstrap confidence interval for HRTD-Net Macro AP falls below 0.5800.
   - The mean bootstrap Weighted Utility falls below 0.4600.
2. **Statistical Insignificance:**
   - Paired permutation testing between HRTD-Net and baseline models yields $p \ge 0.05$, indicating that observed gains could be attributed to random split noise.
3. **Cryptographic Checksum Failure:**
   - Any checksum mismatch or missing artifact hash across the 49 experiment directories.
4. **Data Leakage Detection:**
   - `test_sealed.jsonl` fails its immutable SHA-256 integrity check (`a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`) or exhibits any file access modification timestamp.
5. **Deterministic Reproduction Discrepancy:**
   - Any metric delta exceeding $10^{-4}$ between freshly executed benchmark predictions and recorded JSON metrics.
