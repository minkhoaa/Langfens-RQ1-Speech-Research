# Research Plan: Experiment EXP-64 (Tredecim-Pareto 13-Model Super-Ensemble)

## 1. Objective & Model Roster
Construct, optimize, and benchmark the **13-Model Tredecim Super-Ensemble ("Tredecim-Pareto")** fusing all 13 landmark architectures:
1. **Model 1:** PSC-Net (EXP-22: Phonological Contrastive Margin)
2. **Model 2:** SRAD-Net (EXP-17: Speaking-Rate Duration Modeling)
3. **Model 3:** NNA-Adapter (EXP-21: Non-Native Acoustic Adapter)
4. **Model 4:** ConformerScorer (EXP-10: Multi-Head Attention)
5. **Model 5:** JSAC-Net (EXP-24: Monotonic Local Gaussian Attention)
6. **Model 6:** APED-Net (EXP-28: Acoustic Prosodic Energy Dynamics)
7. **Model 7:** PNC-Net (EXP-34: CMUDict N-Gram Phonotactic Prior)
8. **Model 8:** HTC-Net (EXP-38: Hierarchical Broad-Category Cascade)
9. **Model 9:** VTLN-HTC-Net (EXP-42: Vocal Tract Length Normalization)
10. **Model 10:** DMCT-Net (EXP-44: Dynamic Sequence Token Masking)
11. **Model 11:** HRTD-Net (EXP-48: 5ms Sub-frame Micro-Dynamics Champion)
12. **Model 12:** DSCA-Net (EXP-55: Dual-Stream Cross-Attention Champion)
13. **Model 13:** TDIS-Net (EXP-63: Triad Decoupled Insertion Specializer Champion)

## 2. Experimental Steps
1. Load all 13 model checkpoints and generate raw prediction tensors across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION` on CPU (4 threads) or GPU.
2. Run convex Bayesian Pareto grid sweep on `DEV_CAL` optimizing mixture weights ($\sum_{m=1}^{13} w_m = 1.0$) and operating thresholds ($\theta_s, \theta_o, \theta_i$).
3. Freeze optimal operating policy and evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Audit against H59 falsification criteria.
5. Save predictions, metrics, configuration, and checksums under `experiments/EXP-64-TREDECIM-GLOBAL-PARETO-ENSEMBLE/`.
