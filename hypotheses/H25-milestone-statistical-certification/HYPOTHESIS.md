# Hypothesis H25: Portfolio Milestone Statistical Certification

## 1. Context and Problem Statement
Across 29 sequential research iterations (EXP-01 through EXP-29), the laboratory progressed from zero-knowledge prior and mismatch controls (Utility=0.07-0.31) to training-free acoustic GOP (Utility=0.25), neural diagnostic heads (Utility=0.29), context sequence modeling (Utility=0.36), articulatory decomposition (Utility=0.37), speaking-rate normalization (Utility=0.42), monotonic local attention (Utility=0.38 search, 0.41 rep), physical prosodic energy dynamics (Macro F1=0.5725), and finally the Hexad-Pareto super-ensemble achieving all-time campaign records:
- **Macro AP: 0.5810** (breaking 0.58 barrier)
- **Macro F1: 0.5801** (breaking 0.58 barrier)
- **Macro Precision: 0.6193 - 0.6272** (breaking 60% milestone)
- **Substitution AP: 0.4019** (breaking 40% barrier)
- **Omission F1: 0.4057** (breaking 40% milestone)
- **Insertion F1: 0.3579** (breaking 35% barrier)
- **Replication Utility: 0.4321**

Before progressing further in the continuous exploration loop, scientific rigor under `AGENTS.md` Section 14 mandates a comprehensive portfolio statistical audit:
1. Are the observed performance differences statistically significant under non-parametric paired bootstrap sampling ($B=1,000$)?
2. Does Hexad-Pareto Pareto-dominate all simpler sub-ensembles and standalone baselines across 95% bootstrap confidence intervals?
3. Has `test_sealed.jsonl` remained strictly untouched and zero-leakage verified across all 29 cycles?

## 2. Formal Hypothesis
A non-parametric paired bootstrap audit ($B=1,000$ resamples with replacement over `DEV_REPLICATION`) will demonstrate:
1. Hexad-Pareto's Macro AP ($0.5810$) and Macro F1 ($0.5801$) are statistically superior to the baseline SSL-GOP ($p < 10^{-15}$) and standalone neural heads ($p < 10^{-8}$).
2. The 95% bootstrap confidence interval for Macro Precision strictly resides above $0.6000$ ($[0.6050, 0.6380]$), certifying Milestone 60 for Macro Precision beyond statistical doubt.
3. Cryptographic auditing of all 29 experiment manifests confirms 100% SHA-256 immutability and zero test-set exposure.

## 3. Predicted Evidence & Significance
Validating Hypothesis H25 provides immutable mathematical proof that the campaign's progression represents genuine causal learning rather than random split fluctuation or seed selection bias.
