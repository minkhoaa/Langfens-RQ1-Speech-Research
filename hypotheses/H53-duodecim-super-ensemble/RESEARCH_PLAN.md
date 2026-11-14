# Research Plan: Experiment EXP-58 (Duodecim-Pareto 12-Model Super-Ensemble)

## 1. Objective & Model Roster
Construct, optimize, and benchmark the **12-Model Duodecim Super-Ensemble ("Duodecim-Pareto")** fusing all 12 landmark architectures developed in the research campaign:
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
12. **Model 12:** DSCA-Net (EXP-55: Dual-Stream Audio-Prompt Cross-Attention Champion)

## 2. Experimental Steps
1. Load all 12 model checkpoints and generate raw prediction tensors across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION` on GPU.
2. Run convex Bayesian Pareto grid sweep on `DEV_CAL` optimizing mixture weights ($\sum_{m=1}^{12} w_m = 1.0$) and operating thresholds ($\theta_s, \theta_o, \theta_i$).
3. Freeze optimal operating policy and evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Audit against H53 falsification criteria.
5. Save predictions, metrics, configuration, and checksums under `experiments/EXP-58-DUODECIM-SUPER-ENSEMBLE-FUSION/`.
