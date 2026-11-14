# Hypothesis H69: VTLN-Warped Cross-Attention Spectral-Residual Network (V-CASR-Net)

## 1. Context & Motivation
In EXP-71 (CASR-Net), spectral cepstral residuals $\Delta_{\text{cep}} = |c_{\text{learner}} - \bar{c}_{\text{canonical}}|$ surged Substitution Recall to an all-time peak of 57.65% and Macro Recall to 63.78%. However, static MFCC centroids conflated harmless speaker vocal tract length differences with true substitutions, driving false alarms to 3.517 per utterance.

In EXP-42 and EXP-62, vocal tract length normalization (VTLN) proved that speaker-specific formant warping factors $\alpha_{\text{spk}} = \bar{F}_{\text{spk}} / \bar{F}_{\text{corpus}}$ normalize acoustic representations across different vocal tract dimensions.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **VTLN-Warped Cross-Attention Spectral-Residual Network (V-CASR-Net)** by:
   - Calculating speaker-specific VTLN warping factors $\alpha_{\text{spk}}$ strictly from unlabelled vowel formant centroid distributions.
   - Normalizing spectral cepstral residuals by speaker vocal tract length:
     $$\Delta_{\text{cep, normalized}} = \frac{|c_{\text{learner}} - \bar{c}_{\text{canonical}}|}{\alpha_{\text{spk}}}$$
   - Augmenting the acoustic stream to 76 dimensions (67 base HRTD + 8 VTLN-normalized cepstral residuals + 1 VTLN $\alpha_{\text{spk}}$).
   - Coupling with the 47-dimensional IPTM prompt stream (canonical phone + CMUDict bigrams + empirical interlanguage transfer priors $\pi(p)$) and 13-dimensional triad gap stream.
   - Aligning both streams via bidirectional Multi-Head Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$) followed by 2-layer BiLSTM and manner/triad-gap gating.
2. V-CASR-Net will achieve:
   - **Standalone Macro AP:** $\ge 0.6220$ (62.20%) (Surpassing EXP-48 HRTD-Net 0.6202 and EXP-69 IPTM-Net 0.6186).
   - **False Warning Rate:** Dropping from 3.517 down to $\le 1.85$ false alarms per utterance.
   - **Standalone Macro Precision:** $\ge 0.6200$ (62.00%).
   - **Standalone Macro F1:** $\ge 0.6150$ (61.50%).
   - **Standalone Weighted Utility:** $\ge 0.4930$ (49.30%).
   - **Substitution AP:** $\ge 0.4500$ with Substitution Precision $\ge 52.00\%$.
   - **Omission AP:** $\ge 0.4400$ with Omission Recall $\ge 53.00\%$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6220$ (All-time standalone peak!).
2. **DEV_REPLICATION False Warning Rate:** $\le 1.85$ (Eliminating over 45% of false alarms from EXP-71).
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.4930$.
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and unlabelled speaker acoustic statistics.
