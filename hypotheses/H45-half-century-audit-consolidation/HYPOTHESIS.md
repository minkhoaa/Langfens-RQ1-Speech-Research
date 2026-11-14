# Hypothesis H45: Statistical Significance and Clustered Bootstrap Certification of 60% Milestones

## 1. Context & Motivation
Across 49 continuous research experiments, the laboratory has discovered, designed, and confirmed several monumental breakthrough architectures:
- **EXP-44 (DMCT-Net):** First model to break 60% Macro AP (0.6062).
- **EXP-45 (Decimal-Pareto):** 10-model composite setting all-time high Substitution Precision (55.13%) and lowest false alarm rate (1.09 - 1.11 FWR).
- **EXP-47 (DMCT-Net INT8):** Sub-3ms CPU edge streaming engine (2.02 ms latency, 247.3x speedup) with zero parity loss.
- **EXP-48 (HRTD-Net):** First model in laboratory history to break the 60% Macro F1 milestone (0.6050) while achieving all-time highs in Macro AP (0.6202), Weighted Utility (0.4846), Substitution AP (0.4487), Omission AP (0.4375), and Calibration ECE (0.0117).

Before proceeding to subsequent research phases (multi-corpus acoustic transfer and 11-model ultra-ensembles), scientific rigor demands a comprehensive Half-Century Audit:
1. **Speaker-Clustered Bootstrap (10,000 iterations)**: Quantifying 95% confidence intervals under speaker-level variance.
2. **Paired Permutation Testing**: Demonstrating that the improvement of HRTD-Net over previous baselines is statistically significant ($p < 0.001$).
3. **Sealed Test Integrity Certification**: Verifying that `test_sealed.jsonl` has never been touched or leaked.
4. **Manifest Audit**: Cryptographic SHA-256 verification across all 49 experiment directories.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. On a **10,000-sample speaker-clustered bootstrap** across the held-out `DEV_REPLICATION` split (300 utterances across independent speakers), the 95% Confidence Interval for **HRTD-Net** will strictly maintain:
   - $\text{Macro AP}_{95\% \text{ CI}} \ge [0.6000, 0.6400]$ (statistically proving that the 60% milestone is passed with $>97.5\%$ confidence).
   - $\text{Macro F1}_{95\% \text{ CI}} \ge [0.5850, 0.6250]$.
   - $\text{Weighted Utility}_{95\% \text{ CI}} \ge [0.4650, 0.5050]$.
2. In a paired permutation test ($B=10,000$), HRTD-Net will demonstrate statistically significant superiority over earlier baselines (e.g. DMCT-Net and Hexad-Pareto) with $p < 0.001$.
3. All 49 prior experiment manifests will pass cryptographic verification with zero missing artifacts and zero checksum mismatches.
