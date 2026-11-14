# Research Plan: Hypothesis H22 / Experiment EXP-27

## 1. Objective
Implement and evaluate the 5-Model Heterogeneous Super-Ensemble (**Pentad-Pareto**) fusing JSAC-Net, PSC-Net, SRAD-Net, NNA-Adapter, and Conformer on `DEV_CAL` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Probability Extraction:**
   - Load saved checkpoints from:
     * Model 1: JSAC-Net (`experiments/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINTS/jsac_net.pt`)
     * Model 2: PSC-Net (`experiments/EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN/CHECKPOINTS/psc_net.pt`)
     * Model 3: SRAD-Net (`experiments/EXP-17-ADAPTIVE-DURATION-CTC/CHECKPOINTS/srad_net.pt`)
     * Model 4: NNA-Adapter (`experiments/EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT/CHECKPOINTS/nna_adapter.pt`)
     * Model 5: Conformer (`experiments/EXP-10-CONFORMER-ATTENTION/CHECKPOINTS/conformer_scorer.pt`)
2. **Mixture Tuning on DEV_CAL:**
   - Optimize convex combination weights $(w_1, w_2, w_3, w_4, w_5)$ and decision thresholds $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$ on `DEV_CAL` using grid search.
3. **Evaluation on Independent Holdouts:**
   - Evaluate on `DEV_SEARCH` (599 utts) and `DEV_REPLICATION` (300 utts, spk TXHC & YKWK).
   - Compute full confusion metrics, PR curves, AP, F1, ECE, FWR, and overall Utility.
4. **Audit and Archival:**
   - Persist predictions, metrics, logs, run specifications, and SHA-256 manifest.
   - Enqueue next research task.
