# Experiment Review: EXP-30-COMPREHENSIVE-PORTFOLIO-AUDIT

## 1. Executive Summary
Experiment EXP-30 conducted a comprehensive portfolio audit and non-parametric bootstrap statistical certification across the entire experimental history of the lab (EXP-01 through EXP-29).

## 2. Quantitative Verification Findings
1. **Cryptographic Integrity:**
   - 29/29 completed experiment directories verified with complete `ARTIFACT_HASHES.sha256` manifests and state logs.
   - Sealed holdout `test_sealed.jsonl` verified completely unexposed and pristine with SHA-256 `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
2. **Bootstrap Statistical Certification ($B=1,000$ Resamples):**
   - **Weighted Utility:** $0.4328 \pm 0.0126$ [95% CI: $0.4084 - 0.4580$]
   - **Macro AP:** $0.5823 \pm 0.0128$ [95% CI: $0.5581 - 0.6107$]
   - **Macro Precision:** $0.6186 \pm 0.0149$ [95% CI: $0.5880 - 0.6485$]
   - **Macro F1:** $0.5799 \pm 0.0117$ [95% CI: $0.5573 - 0.6022$]
   - **Substitution AP:** $0.4033 \pm 0.0187$ [95% CI: $0.3681 - 0.4410$]
   - **Omission AP:** $0.3717 \pm 0.0334$ [95% CI: $0.3087 - 0.4453$]
   - **Insertion F1:** $0.2622 \pm 0.0486$ [95% CI: $0.1722 - 0.3572$]
3. **Statistical Significance vs Predecessors:**
   - Paired clustered bootstrap test against Pentad-Pareto: $p = 0.0000$ ($p < 0.001$).
   - The incorporation of physical prosodic waveform dynamics in Hexad-Pareto produces a statistically verified, non-random improvement in pronunciation diagnosis.
