# Research Plan: Hypothesis H21 / Experiment EXP-26

## 1. Objective
Design, implement, train, calibrate, and evaluate the **Contrastive Adapter Phonetic Diagnostic Network (CAPD-Net)** on `DEV_SEARCH` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Architecture Definition:**
   - Input: 41-dim logit posterior + frame duration/tempo features + phonological articulatory one-hot embeddings.
   - Bottleneck Adapter: Linear($D \to 64$), LayerNorm, GeLU, Linear($64 \to D$), Residual Addition $x + \text{Adapter}(x)$.
   - Monotonic Local Gaussian Attention: Window radius $R=4$ frames, $\sigma=3.0$, centered on monotonic target phone segment centers.
   - Multi-Task Diagnostic Heads:
     * 3-Class Target Head: Correct, Substitution, Omission (with Focal Loss $\gamma=2.0$).
     * Binary Gap Insertion Head: Binary cross-entropy on inter-phonemic gap representations.
     * Articulatory Auxiliary Heads: Manner (6-class) and Voicing (2-class) auxiliary heads.
     * Normalized Prototype Contrastive Head: Cosine distance margin loss ($m=0.35, \lambda=0.20$).
2. **Training Execution:**
   - Dataset: `TRAIN` split (1800 utterances, 12 speakers).
   - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4, 30 epochs on NVIDIA GTX 1650.
3. **Calibration:**
   - Pareto calibration on `DEV_CAL` (300 utterances) to freeze optimal operating thresholds $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$.
4. **Evaluation & Verification:**
   - Scored on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).
   - Compute full confusion metrics, PR curves, AP, F1, ECE, FWR, and overall Utility.
5. **Ledger & Artifact Audit:**
   - Persist model checkpoints, predictions, metrics, logs, and SHA-256 manifest.
   - Enqueue subsequent research action.
