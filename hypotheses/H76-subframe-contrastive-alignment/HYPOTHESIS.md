# Hypothesis H76: Subframe Temporal Contrastive Aligner Network (STCA-Net)

## 1. Context & Motivation
Across previous standalone breakthroughs, the laboratory achieved:
- **EXP-69 (IPTM-Net):** Standalone Macro F1 peak of 0.6205 and Omission Recall of 55.64% via empirical interlanguage error transfer priors $\pi(p)$.
- **EXP-77 (TPD-Net):** Standalone Macro Recall peak of 66.37% and Macro F1 of 0.6156 via multi-resolution 2.5ms temporal pyramid features.
- **EXP-55 (DSCA-Net):** Standalone Weighted Utility peak of 0.4928 via bidirectional dual-stream cross-attention.

However, all existing alignment heads rely on unconstrained softmax cross-attention or uniform Gaussian local attention ($\sigma = 3.0$). In non-native pronunciation, acoustic transitions at plosive releases and cluster edges are often smeared over 20-30ms, causing adjacent phonemes to blend into each other. This creates attention dispersion, slightly depressing substitution precision.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Subframe Temporal Contrastive Aligner Network (STCA-Net)** by:
   - Introducing an asymmetric temporal contrastive boundary sharpening loss:
     $$\mathcal{L}_{\text{transit}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \text{ReLU}\left(m_{\text{transit}} - \|z_{t+1} - z_t\|_2^2\right) \cdot \mathbb{I}[p_t \neq p_{t+1}]$$
     $$\mathcal{L}_{\text{smooth}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|z_{t+1} - z_t\|_2^2 \cdot \mathbb{I}[p_t = p_{t+1}]$$
   - Coupling the 72-dimensional multi-resolution temporal pyramid acoustic stream (2.5ms micro-burst to 20ms context) with the 47-dimensional IPTM prompt stream (interlanguage error priors $\pi(p)$).
   - Employing sharp monotonic local attention with boundary-gated projection.
2. STCA-Net will achieve:
   - **Standalone Macro F1:** Breaking the 62.5% milestone for the first time in laboratory history ($\text{Macro F1} \ge 0.6250$).
   - **Standalone Macro AP:** $\ge 0.6220$ (62.20%) (Surpassing HRTD-Net 0.6202 and IPTM-Net 0.6186).
   - **Standalone Weighted Utility:** $\ge 0.4950$ (49.50%) (Surpassing DSCA-Net 0.4928).
   - **Substitution AP:** $\ge 0.4550$ with Substitution Precision $\ge 54.00\%$.
   - **Omission Recall:** $\ge 54.00\%$ with Omission F1 $\ge 0.4600$.
   - **False Warning Rate:** $\le 1.80$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro F1:** $\ge 0.6250$ (All-time standalone peak!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6220$ (All-time standalone peak!).
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$ (All-time standalone peak!).
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and strict canonical target inputs.
