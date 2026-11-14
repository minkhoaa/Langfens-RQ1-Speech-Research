# Hypothesis H82: Phonological-Feature Attention Routing Network (FAR-Net)

## 1. Context & Motivation
In English L2 pronunciation assessment, phonetic substitutions are structured along distinctive articulatory axes (SPE phonological theory):
- Voicing shifts (e.g. `/p/` $\leftrightarrow$ `/b/`, `/t/` $\leftrightarrow$ `/d/`, `/s/` $\leftrightarrow$ `/z/`) depend on Voice Onset Time (VOT) and low-frequency harmonic periodicity.
- Place shifts (e.g. `/s/` $\leftrightarrow$ `/θ/`, `/f/` $\leftrightarrow$ `/θ/`, `/n/` $\leftrightarrow$ `/ŋ/`) depend on second/third formant ($F_2, F_3$) locus transitions and spectral peak frequencies.
- Manner shifts (e.g. `/d/` $\leftrightarrow$ `/ð/`, `/b/` $\leftrightarrow$ `/v/`, `/t/` $\leftrightarrow$ `/s/`) depend on plosive burst rise time, silent closure intervals, and frication noise envelope.

In monolithic cross-attention networks (such as DSCA-Net EXP-55 and IPTM-Net EXP-69), attention weights are computed over a single undifferentiated latent representation, causing place and manner cues to compete and dilute each other.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Phonological-Feature Attention Routing Network (FAR-Net)** by:
   - Decomposing the cross-attention layer into 3 specialized articulatory routing streams:
     * **Manner Attention Head:** Focused on energy envelope dynamics, silence contrast, and rise-time gradients.
     * **Place Attention Head:** Focused on formant locus transitions, spectral ratios, and spectral tilt.
     * **Voicing Attention Head:** Focused on zero-crossing rates, pitch periodicity, and voicing indicators.
   - Dynamically recombining the three routed representations via a gated mixture-of-experts (MoE) projection:
     $$h_{\text{fused}} = \text{LayerNorm}(g_m h_{\text{manner}} + g_p h_{\text{place}} + g_v h_{\text{voicing}})$$
     where $[g_m, g_p, g_v] = \text{Softmax}(W_g [X_a, X_p])$.
   - Coupling with 67-dimensional HRTD sub-frame acoustic features, 47-dimensional IPTM prompt features (with empirical interlanguage transfer priors $\pi(p)$), and 13-dimensional triad decoupled gap gating.
2. FAR-Net will achieve:
   - **Standalone Macro F1:** Breaking the 62.5% milestone for the first time in laboratory history ($\text{Macro F1} \ge 0.6250$).
   - **Standalone Macro AP:** $\ge 0.6220$ (62.20%) (Surpassing HRTD-Net 0.6202 and IPTM-Net 0.6186).
   - **Standalone Weighted Utility:** $\ge 0.4950$ (49.50%) (Surpassing DSCA-Net 0.4928).
   - **Substitution AP:** $\ge 0.4500$ with Substitution Precision $\ge 54.00\%$.
   - **Omission Recall:** $\ge 55.00\%$ with Omission F1 $\ge 0.4650$.
   - **Overall ECE:** $\le 0.0120$ (1.20%) with False Warning Rate $\le 1.45$ per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro F1:** $\ge 0.6250$ (All-time standalone peak!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6220$ (All-time standalone peak!).
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$ (All-time standalone peak!).
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and canonical target inputs.
