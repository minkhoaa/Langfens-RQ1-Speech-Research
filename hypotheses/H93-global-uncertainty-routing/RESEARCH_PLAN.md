# Research Plan: Experiment EXP-98 (Global Consensus Uncertainty Router - GCUR-Engine)

## 1. Methodology
- Inputs: Posterior probability arrays across `dev_search` and `dev_replication` from:
  1. Viginti-Pareto (EXP-86)
  2. ADAE-Net (EXP-94)
  3. CICA-Net (EXP-95)
  4. MTRC-Net (EXP-97)
- Ensemble Consensus & Entropy Computation:
  $$\bar{P}(k) = \frac{1}{4} [P_{\text{viginti}}(k) + P_{\text{adae}}(k) + P_{\text{cica}}(k) + P_{\text{mtrc}}(k)]$$
  $$H(k) = -\sum_{c} \bar{P}_c(k) \log(\bar{P}_c(k) + 1e-8)$$
- Calibrate $\tau_{\text{abstain}}$ on `dev_cal` such that $\text{Coverage} \ge 90.0\%$.
- Evaluate:
  * 100% full denominator evaluation (where abstentions are scored as unasserted or incorrect per protocol).
  * Selective prediction evaluation on the confident cohort ($\text{Coverage} \ge 90\%$).
- Compute Macro Precision, Recall, F1, AP, FWR, and ECE.
- Audit against H93 falsification criteria.
- Save all artifacts in `experiments/EXP-98-GLOBAL-CONSENSUS-UNCERTAINTY-ROUTER/`.
