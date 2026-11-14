# Research Plan: Experiment EXP-99 (Century Convergence Master Super-Ensemble - Centum-Pareto)

## 1. Ensemble Architecture (Centum-Pareto)
Centum-Pareto fuses:
1. $P_{\text{viginti}}$ (EXP-86: 21 models)
2. $P_{\text{adae}}$ (EXP-94: Articulatory Disentangled Autoencoder Champion)
3. $P_{\text{cica}}$ (EXP-95: Cross-Accent Covariance Aligner)
4. $P_{\text{mtrc}}$ (EXP-97: Multi-Scale Temporal Residual Cascade)
5. $P_{\text{bapm}}$ (EXP-91: Boundary-Aware Posterior Mapper)

## 2. Experimental Execution
1. Load raw prediction posteriors across `dev_search` and `dev_replication`.
2. Optimize convex mixture weights $(w_v, w_a, w_c, w_m, w_b)$ on `dev_cal` via Dirichlet-constrained grid search:
   $$\max_{w, \theta} \text{Utility}_{\text{dev\_cal}}(w, \theta)$$
3. Calibrate class operating thresholds $(\theta_s, \theta_o, \theta_i)$ strictly on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H94 falsification criteria.
6. Save all predictions, metrics, optimal configurations, and hashes in `experiments/EXP-99-CENTURY-CONVERGENCE-MASTER-SUPER-ENSEMBLE/`.
