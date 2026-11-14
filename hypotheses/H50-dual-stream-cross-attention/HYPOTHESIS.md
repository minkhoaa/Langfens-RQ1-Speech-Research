# Hypothesis H50: Bidirectional Dual-Stream Audio-Prompt Cross-Attention (DSCA-Net)

## 1. Context & Motivation
Across our 54 experiments, the strongest standalone models (EXP-44 DMCT-Net, EXP-48 HRTD-Net) have concatenated target phone identity features (one-hot or posterior projections) directly into the acoustic frame representation prior to self-attention.

However, cognitive phonetics and human speech evaluation demonstrate that pronunciation assessment is fundamentally a **comparative dual-stream process**:
1. **The Canonical Expectation Stream ($P \in \mathbb{R}^{N \times D_p}$):** A sequence of phonological prompt representations encoding target articulatory targets (manner, place, voicing, and CMUDict phonotactic priors).
2. **The Realized Acoustic Stream ($A \in \mathbb{R}^{N \times D_a}$):** Continuous 5ms sub-frame acoustic micro-dynamics (burst onset energy, instantaneous $dE/dt$, crest factor, and spectral tilt).

When these two streams are concatenated early into a single representation, self-attention can suffer from representation collapse, where acoustic frame features dominate and mask subtle phonetic expectation mismatches.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **DSCA-Net (Dual-Stream Cross-Attention Network)** with decoupled acoustic and prompt encoders coupled via **Bidirectional Multi-Head Cross-Attention (MHCA)**:
   $$A_{\text{guided}} = A + \text{MHCA}(Q = \text{LayerNorm}(A), K = P, V = P)$$
   $$P_{\text{verified}} = P + \text{MHCA}(Q = \text{LayerNorm}(P), K = A, V = A)$$
   $$H_{\text{fused}} = \text{LayerNorm}(\text{Linear}([A_{\text{guided}}, P_{\text{verified}}]))$$
   will enable direct articulatory discrepancy comparison.
2. In particular:
   - For **`OMISSION`**, the prompt expectation vector $P$ will query the acoustic stream and detect a near-zero acoustic activation profile, drastically reducing omission misses.
   - For **`SUBSTITUTION`**, discrepancies in manner/place between $A$ and $P$ will create a sharp cross-stream residual vector, improving substitution discrimination.
3. DSCA-Net will achieve:
   - **Substitution AP** breaking the 45.5% milestone ($\text{Substitution AP} \ge 0.4550$).
   - **Omission AP** maintaining $\ge 0.4400$ with Omission Recall $\ge 45.00\%$.
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Weighted Utility** breaking the 0.4850 milestone.
   - **False Warning Rate** dropping to $\le 1.60$ false alarms/utterance on `DEV_REPLICATION`.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
2. **DEV_REPLICATION Macro F1:** $\ge 0.6050$.
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4550$.
4. **DEV_REPLICATION FWR:** $\le 1.60$.
