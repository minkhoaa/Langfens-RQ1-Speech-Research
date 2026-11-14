# Research Plan: Experiment EXP-106 (Extended 25-Model Convergence Super-Ensemble - Viginti-Quintum-Pareto)

## 1. Ensemble Architecture
The 25-model mixture blends:
- $P_{\text{viginti}}$ (EXP-86: 21 models)
- $P_{\text{pear}}$ (EXP-102: LoRA Adapter, High-Recall Substitution)
- $P_{\text{dsga}}$ (EXP-103: Graph Aligner, High-AP Omission)
- $P_{\text{sgan}}$ (EXP-104: Soft Gold, High-Precision Low-FWR)
- $P_{\text{vti}}$ (EXP-101: Physical Vocal Tract Inversion)

## 2. Experimental Execution
1. Load raw prediction posteriors across `dev_search` and `dev_replication`.
2. Optimize convex mixture weights $(w_v, w_p, w_d, w_s, w_t)$ on `dev_cal` via Dirichlet-constrained grid search:
   $$\max_{w, \theta} \text{Utility}_{\text{dev\_cal}}(w, \theta)$$
3. Calibrate class operating thresholds $(\theta_s, \theta_o, \theta_i)$ on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H101 falsification criteria.
6. Save all predictions, metrics, optimal configurations, and hashes in `experiments/EXP-106-DUODEVIGINTI-PLUS-TWO-CONVERGENCE-SUPER-ENSEMBLE/`.
