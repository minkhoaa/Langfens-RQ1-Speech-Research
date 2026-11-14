# Research Plan: Experiment EXP-80 (Four-Fifths Century Campaign Consolidation & Statistical Master Audit)

## 1. Objective & Scope
Perform an exhaustive cryptographic, methodological, and statistical certification of the entire research campaign across 80 cycles:
1. **Cryptographic Manifest Verification:**
   Audit and compute SHA-256 digests for all 79 prior experiment directories (`EXP-01` to `EXP-79`).
2. **Sealed Holdout Integrity Verification:**
   Verify byte-exact SHA-256 match for `data/derived/l2arctic/manifests/test_sealed.jsonl` to certify zero exposure and zero data leakage.
3. **10,000-Sample Speaker-Clustered Bootstrap:**
   Resample utterances clustered by speaker ID with replacement ($B = 10,000$) on `DEV_REPLICATION` for Duodeviginti-Pareto (EXP-79) to compute exact empirical 95% Confidence Intervals for:
   - Weighted Utility
   - Macro AP
   - Macro Precision, Recall, F1
   - Class-specific metrics (CORRECT, SUBSTITUTION, OMISSION, INSERTION)
4. **Paired Statistical Significance Testing:**
   Compute exact two-tailed paired bootstrap p-values comparing Duodeviginti-Pareto against:
   - Quindecim-Pareto (EXP-70)
   - Undecim-Pareto (EXP-52)
   - Hexad-Pareto (EXP-36)
   - Baseline BiLSTM (EXP-08)
5. **Cryptographic Receipt Sealing:**
   Output and seal `experiments/EXP-80-CAMPAIGN-FOUR-FIFTHS-CENTURY-CONSOLIDATION-AND-MILESTONE-AUDIT/FOUR_FIFTHS_CENTURY_AUDIT_RECEIPT.json`.
