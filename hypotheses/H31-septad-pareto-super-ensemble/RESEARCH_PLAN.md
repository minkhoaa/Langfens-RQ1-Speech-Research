# Research Plan: Hypothesis H31 / Experiment EXP-36

## 1. Objective
Implement, optimize, and evaluate the 7-Model Septad Super-Ensemble (**Septad-Pareto**) fusing PSC-Net, SRAD-Net, NNA-Adapter, Conformer, JSAC-Net, APED-Net, and PNC-Net on `DEV_CAL` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Checkpoints:**
   - Model 1: PSC-Net (`EXP-22`)
   - Model 2: SRAD-Net (`EXP-17`)
   - Model 3: NNA-Adapter (`EXP-21`)
   - Model 4: Conformer (`EXP-10`)
   - Model 5: JSAC-Net (`EXP-24`)
   - Model 6: APED-Net (`EXP-28`)
   - Model 7: PNC-Net (`EXP-34`)
2. **Master Sequence Construction:**
   - Unified extraction supplying 55-dim, 51-dim, 60-dim, and 63-dim target features, and 6-dim, 9-dim, and 11-dim gap features.
3. **Global Multi-Objective Pareto Optimization on DEV_CAL:**
   - Optimize target mixture weights $(w_1, \dots, w_7)$.
   - Optimize gap mixture weights $(v_1, \dots, v_7)$.
   - Optimize thresholds $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log all metrics, PR curves, ECE, FWR, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save configuration, metrics, predictions, logs, and SHA-256 hashes under `experiments/EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION/`.
