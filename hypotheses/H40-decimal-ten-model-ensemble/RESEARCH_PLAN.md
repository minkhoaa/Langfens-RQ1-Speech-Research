# Research Plan: Hypothesis H40 / Experiment EXP-45

## 1. Objective
Implement, optimize, and evaluate the 10-Model Decimal Super-Ensemble (**Decimal-Pareto**) fusing PSC-Net, SRAD-Net, NNA-Adapter, Conformer, JSAC-Net, APED-Net, PNC-Net, HTC-Net, VTLN-HTC-Net, and DMCT-Net on `DEV_CAL` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Checkpoints:**
   - Model 1: PSC-Net (`EXP-22`)
   - Model 2: SRAD-Net (`EXP-17`)
   - Model 3: NNA-Adapter (`EXP-21`)
   - Model 4: Conformer (`EXP-10`)
   - Model 5: JSAC-Net (`EXP-24`)
   - Model 6: APED-Net (`EXP-28`)
   - Model 7: PNC-Net (`EXP-34`)
   - Model 8: HTC-Net (`EXP-38`)
   - Model 9: VTLN-HTC-Net (`EXP-42`)
   - Model 10: DMCT-Net (`EXP-44`)
2. **Master Input Representation:**
   - Master 66-dim target and 11-dim gap sequences.
3. **Global Multi-Objective Pareto Optimization on DEV_CAL:**
   - Optimize target mixture weights $(w_1, \dots, w_{10})$.
   - Optimize gap mixture weights $(v_1, \dots, v_{10})$.
   - Optimize operating thresholds $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
   - Log all metrics, PR curves, ECE, FWR, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Save configuration, metrics, predictions, logs, and SHA-256 hashes under `experiments/EXP-45-DECIMAL-PARADIGM-SUPER-ENSEMBLE/`.
