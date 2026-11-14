# Research Plan: Experiment EXP-107 (Progressive Cascade Viginti-Quintum Router - PCR-25 Engine)

## 1. Methodology
- **Tier 1 Fast Screening Models:**
  * Targets: INT8-ADAE-Net (EXP-96: 3.56 ms latency).
  * Gaps: TDIS-Net (EXP-63: decoupled insertion specialist).
- **Tier 2 Full Ensemble:**
  * Viginti-Quintum-Pareto 25-model mixture (EXP-106).
- **Early Exit Criteria:**
  * Target slot $k$: If $\max(p_c, p_s, p_o) \ge \gamma_{\text{target}}$, exit at Tier 1.
  * Gap slot $g$: If $p_{\text{ins}} < 0.05$ or $p_{\text{ins}} > \gamma_{\text{gap}}$, exit at Tier 1.
- **Threshold Calibration on `dev_cal`:**
  Sweep $\gamma_{\text{target}} \in [0.85, 0.94]$ and $\gamma_{\text{gap}} \in [0.90, 0.98]$ to maximize Pareto utility while enforcing $\ge 85\%$ early exit rate.
- **Evaluation:**
  Evaluate full metrics, early exit rates, CPU latency, and speedup factors on `dev_search` and `dev_replication`.
- Audit against H102 falsification criteria.
- Save all artifacts in `experiments/EXP-107-PROGRESSIVE-CASCADE-VIGINTI-QUINTUM-ROUTER/`.
