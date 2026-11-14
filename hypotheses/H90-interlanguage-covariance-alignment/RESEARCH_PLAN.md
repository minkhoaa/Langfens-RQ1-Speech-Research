# Research Plan: Experiment EXP-95 (Cross-Accent Interlanguage Covariance Aligner - CICA-Net)

## 1. Methodology
1. Compute 39x39 empirical phoneme error covariance matrices $\Sigma_{\ell}$ strictly from `data/derived/l2arctic/manifests/train.jsonl` for each L1 cohort.
2. In CICA-Net:
   - Formulate structured covariance projection:
     $$x_{\text{cov}}(k) = \Sigma_{\ell} \cdot x_{\text{one-hot}}(k) \in \mathbb{R}^{39}$$
   - Concatenate with 42-dim canonical prompt stream (total 81 dims).
   - Couple with ADAE-Net 72-dim acoustic stream with orthogonal articulatory latent spaces.
3. Train on GPU for 10 epochs.
4. Calibrate on `dev_cal`.
5. Evaluate on `dev_search` and `dev_replication`.
6. Audit against H90 falsification criteria.
7. Save all artifacts in `experiments/EXP-95-CROSS-ACCENT-INTERLANGUAGE-COVARIANCE-ALIGNER/`.
