# Hypothesis H51: Conformer Depthwise Convolution with Sub-Frame 5ms Micro-Dynamics (CHTD-Net)

## 1. Context & Motivation
Our previous experiments demonstrated two major standalone breakthroughs:
1. **EXP-48 HRTD-Net:** 5ms sub-frame temporal micro-dynamics broke the 60% Macro F1 milestone (0.6050) and achieved Macro AP = 0.6202.
2. **EXP-55 DSCA-Net:** Bidirectional audio-prompt cross-attention broke the standalone Weighted Utility record (0.4928).

However, both models relied on standard BiLSTM encoders. Recurrent cells compress history into hidden state vectors, which can smear localized transient bursts (e.g., plosive stop releases lasting 5-15ms). In contrast, **Conformer depthwise separable convolutions** (kernel size 5) apply shift-invariant filtering directly over local temporal windows, specifically capturing acoustic bursts, formant transitions, and glottal closures.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **CHTD-Net (Conformer High-Resolution Temporal Dynamics Network)** by replacing recurrence with a 2-block Conformer architecture (Macaron-style feedforward, multi-head self-attention, and depthwise convolution) coupled with 5ms sub-frame micro-dynamics and bidirectional prompt cross-attention:
   $$\text{ConvBlock}(X) = \text{Conv1D}_{\text{pointwise}}(\text{GLU}(\text{BatchNorm}(\text{Conv1D}_{\text{depthwise}}(\text{LayerNorm}(X)))))$$
   will yield superior local shift-invariance for transient plosive bursts.
2. CHTD-Net will achieve:
   - **Weighted Utility** breaking the 0.5000 milestone for the first time in a standalone model ($\text{Utility} \ge 0.5000$).
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Macro F1** sustaining $\ge 0.6100$.
   - **Substitution AP** breaking $\ge 0.4500$ with Substitution Precision $\ge 52.00\%$.
   - **Omission AP** maintaining $\ge 0.4450$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (All-time standalone peak).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
3. **DEV_REPLICATION Macro F1:** $\ge 0.6100$.
4. **DEV_REPLICATION Substitution AP:** $\ge 0.4500$.
