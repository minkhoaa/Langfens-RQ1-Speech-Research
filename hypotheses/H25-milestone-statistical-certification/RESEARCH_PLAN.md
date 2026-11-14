# Research Plan: Hypothesis H25 / Experiment EXP-30

## 1. Objective
Execute a comprehensive 30-experiment portfolio audit, non-parametric paired bootstrap statistical significance test ($B=1,000$), cryptographic hash verification, and publication-ready scientific milestone scorecard.

## 2. Experimental Steps
1. **Cryptographic Integrity Audit:**
   - Verify SHA-256 hashes for all 29 preceding experiment directories (`experiments/EXP-01` through `experiments/EXP-29`).
   - Confirm `data/derived/l2arctic/manifests/test_sealed.jsonl` remains pristine (`a5636249f7...`).
2. **Bootstrap Significance Testing:**
   - Load raw prediction records on `DEV_REPLICATION` for:
     * Hexad-Pareto (EXP-29)
     * APED-Net (EXP-28)
     * Pentad-Pareto (EXP-27)
     * QC-Pareto (EXP-23)
     * SSL-GOP Baseline (EXP-03)
   - Perform $B=1,000$ paired bootstrap resamples of utterances.
   - For each bootstrap resample, compute Macro AP, Macro Precision, Macro F1, Utility, and per-class AP/F1.
   - Compute mean, standard deviation, and 95% empirical bootstrap confidence intervals $[CI_{0.025}, CI_{0.975}]$.
   - Calculate empirical $p$-values for paired comparisons against baseline.
3. **Synthesis & Milestone Report:**
   - Synthesize all Pareto-dominant breakthroughs into `experiments/EXP-30-COMPREHENSIVE-PORTFOLIO-AUDIT/METRICS/PORTFOLIO_AUDIT_REPORT.json`.
   - Update `reports/SCIENTIFIC_SYNTHESIS.md`, `reports/SCORECARD.md`, and `reports/PARETO_FRONTIER.md`.
