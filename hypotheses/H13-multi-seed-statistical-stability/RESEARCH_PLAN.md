# Research Plan: Hypothesis H13 (Multi-Seed Certification)

1. **Seeds:**
   - Run 5 distinct runs with seeds: 101, 202, 303, 404, 505.
2. **Protocol:**
   - Train `AdaptiveHierarchicalNetwork` (55-dim target features) for 25 epochs per seed.
   - Evaluate with frozen Pareto policy ($\theta_{\text{subs}}=0.55, \theta_{\text{omis}}=0.55, \theta_{\text{ins}}=0.60$) on `DEV_REPLICATION` (300 utterances, 9,863 target phones, 10,163 gaps).
3. **Statistical Aggregation:**
   - For every metric, compute $\mu$, $\sigma$, 95% Confidence Interval ($t$-distribution, $df=4$).
4. **Artifacts & Ledger:**
   - Save multi-seed summary in `experiments/EXP-18-MULTI-SEED-CERTIFICATION/`.
