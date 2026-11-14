# Hypothesis H67: L1-Stratified Discriminative Calibration (LDC-Engine)

## 1. Context & Motivation
Across previous cycles, the laboratory demonstrated two critical findings regarding non-native accent backgrounds:
1. In EXP-53 (DIAA-Net), dialect feature adaptation reduced cross-L1 variance $\sigma_{\text{L1}}$ by 20.0%, proving that non-native mispronunciation detection varies systematically across L1 cohorts.
2. In EXP-71 (CASR-Net), static uncalibrated acoustic thresholds produced 813 false substitutions, as dialect-specific acoustic dispersion triggers false alarms under rigid global thresholds.

Currently, even our champion ensemble (EXP-70 Quindecim-Pareto) uses a single global threshold triplet $(\theta_s=0.52, \theta_o=0.52, \theta_i=0.60)$ across all 6 L1 backgrounds.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **L1-Stratified Discriminative Calibration Engine (LDC-Engine)** by:
   - Estimating dialect-specific temperature scaling $T_{\ell} \in [0.8, 1.4]$ and threshold shifts $\Delta \theta_{\ell} = [\Delta \theta_s, \Delta \theta_o, \Delta \theta_i]_{\ell}$ strictly on `train.jsonl` and `dev_cal.jsonl` for each of the 6 L1 backgrounds (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese).
   - Applying temperature-scaled posterior normalization:
     $$\tilde{p}_{\ell}(y \mid x) = \text{Softmax}\left(\frac{z}{T_{\ell}}\right)$$
     $$\hat{y} = \text{Argmax/Threshold}\left(\tilde{p}_{\ell}, \theta^* + \Delta \theta_{\ell}\right)$$
   - Coupling LDC-Engine with the champion 15-model Quindecim-Pareto ensemble (EXP-70).
2. LDC-Engine will achieve:
   - **Ensemble Macro AP:** $\ge 0.6420$ (64.20%) (Shattering the EXP-70 peak of 0.6390).
   - **Macro Precision:** $\ge 0.6550$ (65.50%) (All-time campaign peak!).
   - **Weighted Utility:** $\ge 0.5010$ (50.10%) (Breaking 50% utility milestone in an ensemble!).
   - **False Warning Rate:** $\le 1.25$ false alarms per utterance (Substantial reduction from 1.413).
   - **Overall ECE:** $\le 0.0250$ (2.50%).

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6420$ (All-time campaign peak!).
2. **DEV_REPLICATION Macro Precision:** $\ge 0.6550$.
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.5010$.
4. **DEV_REPLICATION False Warning Rate:** $\le 1.25$ false alarms/utterance.
