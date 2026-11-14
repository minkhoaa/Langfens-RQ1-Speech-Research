# Research Plan: Hypothesis H30 / Experiment EXP-35

## 1. Objective
Quantify cross-accent generalization across all 6 L1 native language groups (Arabic, Mandarin, Spanish, Hindi, Korean, Vietnamese) for the champion Hexad-Pareto and PNC-Net models across 1,199 unsealed evaluation utterances under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Cohort Partitioning:**
   - Map each speaker in `DEV_SEARCH`, `DEV_CAL`, and `DEV_REPLICATION` to their publisher-annotated native language L1:
     * Arabic: `ABA`, `SKA`
     * Mandarin: `BWC`, `TXHC`
     * Spanish: `EBVS`, `MBXP`
     * Hindi: `HKK`, `TSI`
     * Korean: `HJK`, `YKWK`
     * Vietnamese: `BNV`, `HQTV` (from unsealed partitions)
2. **Inference Execution:**
   - Run frozen Hexad-Pareto (EXP-29) and PNC-Net (EXP-34) on each L1 cohort.
3. **Metric Calculation per L1 Subgroup:**
   - Compute per-L1 Utility, Macro AP, Macro Precision, Macro F1, False Warning Rate, and class-specific metrics for all 4 classes.
   - Compute mean, standard deviation, and range across the 6 L1 groups.
4. **Ledger & Artifact Audit:**
   - Save full subgroup metric tables, predictions, and SHA-256 manifests under `experiments/EXP-35-ENDLESS-SEARCH-CROSS-ACCENT-TRANSFER/`.
