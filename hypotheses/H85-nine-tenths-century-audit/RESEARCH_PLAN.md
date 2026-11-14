# Research Plan: Experiment EXP-90 (Nine-Tenths Century Campaign Consolidation & Grand Master Audit)

## 1. Objective & Scope
Perform an exhaustive cryptographic, methodological, and statistical grand audit of the entire research campaign across 90 cycles:
1. **Cryptographic Manifest Verification:**
   Audit and compute SHA-256 digests for all 89 prior experiment directories (`EXP-01` to `EXP-89`).
2. **Sealed Holdout Integrity Verification:**
   Verify byte-exact SHA-256 match for `data/derived/l2arctic/manifests/test_sealed.jsonl` to certify zero exposure and zero data leakage.
3. **10,000-Sample Speaker-Clustered Bootstrap:**
   Resample utterances clustered by speaker ID with replacement ($B = 10,000$) on `DEV_REPLICATION` for Viginti-Pareto (EXP-86) and PCR-Engine (EXP-89) to compute exact empirical 95% Confidence Intervals for:
   - Weighted Utility
   - Macro AP
   - Macro Precision, Recall, F1
   - Class-specific metrics (CORRECT, SUBSTITUTION, OMISSION, INSERTION)
4. **Paired Statistical Significance Testing:**
   Compute exact two-tailed paired bootstrap p-values comparing Viginti-Pareto against:
   - Duodeviginti-Pareto (EXP-79)
   - Undecim-Pareto (EXP-52)
   - Hexad-Pareto (EXP-36)
   - Baseline BiLSTM (EXP-08)
5. **Production Efficiency Audit:**
   Certify PCR-Engine (EXP-89) early exit rate ($\ge 85\%$) and CPU speedup ($\ge 3.5\times$).
6. **Cryptographic Receipt Sealing:**
   Output and seal `experiments/EXP-90-CAMPAIGN-NINE-TENTHS-CENTURY-CONSOLIDATION-AND-MILESTONE-AUDIT/NINE_TENTHS_CENTURY_AUDIT_RECEIPT.json`.
