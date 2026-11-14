# Hypothesis H13: Multi-Seed Statistical Stability & Reproducibility Certification

## 1. Proposition
A scientifically defensible pronunciation assessment breakthrough cannot rely on lucky weight initialization.
The Speaking-Rate Adaptive Duration Network (`SRAD-Net`) will exhibit strict statistical stability across 5 independent random initialization seeds (`seeds = [101, 202, 303, 404, 505]`), satisfying:
1. Macro F1 standard deviation across seeds $\sigma_{\text{seed}} \le 0.015$,
2. OMISSION F1 standard deviation $\sigma_{\text{omis}} \le 0.020$,
3. SUBSTITUTION F1 standard deviation $\sigma_{\text{subs}} \le 0.020$,
4. Consistent replication of the Macro-Precision $> 60\%$ milestone ($\mu_{\text{macro\_P}} \ge 0.60$),
confirming the method's reproducibility before official certification.

## 2. Experimental Plan
- Method ID: `M18-multi-seed-auditor`
- Run ID: `EXP-18-MULTI-SEED-CERTIFICATION`
- Train and evaluate 5 independent runs on `DEV_REPLICATION`.
- Calculate mean, std, min, max, and 95% CI across all 5 seeds.
