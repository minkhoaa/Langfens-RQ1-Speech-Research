# Hypothesis H56: Spectral Cepstral Formant Residuals (UCR-Net) for High-Precision Substitution Diagnosis

## 1. Context & Motivation
Across our 60 completed experiments, standalone Substitution AP reached **0.4487** in HRTD-Net (EXP-48) and **0.4320** in DSCA-Net (EXP-55), while Duodecim-Pareto (EXP-58) pushed ensemble Substitution AP to **0.4540**.

However, detecting fine-grained phonetic substitutions (e.g. vowel centralization `/IH/` $\leftrightarrow$ `/IY/`, tense/lax contrasts `/UH/` $\leftrightarrow$ `/UW/`, or fricative place shifts `/TH/` $\leftrightarrow$ `/S/`) faces a fundamental representation bottleneck:
- Pretrained Wav2Vec2 foundation models operate primarily on contextual phonetic classification, abstracting away the physical spectral envelope.
- Sub-frame 5ms energy ratios ($dE/dt$, crest factor) capture plosive bursts, but do not encode the frequency distribution of vowel formants ($F_1, F_2, F_3$) or spectral tilt.

In acoustic phonetics, the physical vocal tract configuration is parameterized by its **Mel-Frequency Cepstral Coefficients (MFCCs)**. Computing the **Cepstral Residual Deviation**:
$$\Delta_{\text{cep}}(k) = C_{\text{realized}}(k) - \bar{C}_{\text{canonical}}(p_k)$$
where $\bar{C}_{\text{canonical}}(p_k)$ is the empirical mean cepstral vector of canonical phone $p_k$ calculated strictly from the training partition, provides the network with a direct physical measure of articulatory deviation.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **UCR-Net (Ultra-High-Resolution Cepstral Residual Network)** by augmenting the 67-dimensional acoustic stream of DSCA-Net with 8 normalized cepstral formant residuals ($\Delta_{\text{cep}, 1-8} \in \mathbb{R}^8$, resulting in a 75-dimensional acoustic stream) coupled with bidirectional prompt cross-attention will:
   - Provide an explicit physical articulatory discrepancy vector for every realized phone slot.
   - Disambiguate subtle non-native vowel formant distortions that are otherwise indistinguishable from coarticulation.
2. UCR-Net will achieve:
   - **Substitution AP** breaking the 46% milestone in a standalone model ($\text{Substitution AP} \ge 0.4600$).
   - **Substitution F1** breaking $\ge 0.4400$ with Substitution Precision $\ge 52.00\%$.
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Weighted Utility** breaking the 0.4950 milestone ($\text{Utility} \ge 0.4950$).
   - **Omission AP** maintaining $\ge 0.4400$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Substitution AP:** $\ge 0.4600$ (All-time standalone peak!).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$.
3. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
4. **DEV_REPLICATION Macro F1:** $\ge 0.6050$.
