# Hypothesis H66: Cross-Attention Spectral-Residual Network (CASR-Net)

## 1. Context & Motivation
Across previous cycles, the laboratory isolated distinct mechanisms that attack different aspects of mispronunciation detection:
1. **EXP-55 (DSCA-Net):** Multi-head cross-attention (MHCA) between acoustic and prompt streams achieved an all-time standalone Weighted Utility of 0.4928 and Macro AP = 0.6159.
2. **EXP-61 (UCR-Net):** 8-dimensional spectral cepstral residuals $\Delta_{\text{cep}} = |c_{\text{learner}} - c_{\text{canonical}}|$ provided strong acoustic discrepancy sensitivity (Insertion AP = 0.2945, Target ECE = 0.0110), but lacked prompt cross-attention.
3. **EXP-68 (MRTP-Net):** 2.5ms micro-burst crest features broke the 1.0% ECE barrier (Overall ECE = 0.0083 / 0.83%), dropping false omissions to 89.
4. **EXP-69 (IPTM-Net):** Empirical interlanguage error priors $\pi(p) = [P_c, P_s, P_o]$ achieved standalone Macro F1 = 0.6205 and Omission Recall = 55.64%.

Currently, no single standalone architecture integrates spectral cepstral residuals directly into a cross-attention framework coupled with empirical interlanguage error priors.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Cross-Attention Spectral-Residual Network (CASR-Net)** by:
   - Augmenting the acoustic stream from 67 to 75 dimensions via 8-dimensional spectral cepstral residuals $\Delta_{\text{cep}}$ computed from canonical MFCC phone profiles derived strictly from `train.jsonl`.
   - Conditioning the 47-dimensional prompt stream with empirical interlanguage error transfer priors $\pi(p) = [P(c \mid p), P(s \mid p), P(o \mid p)]$ and CMUDict bigrams.
   - Aligning both modalities via bidirectional Multi-Head Cross-Attention (MHCA, $H=4$, $D=64$) followed by 2-layer BiLSTM and manner/triad-gap gating.
2. CASR-Net will achieve:
   - **Standalone Macro AP:** $\ge 0.6250$ (62.50%) (Surpassing EXP-48 HRTD-Net 0.6202 and EXP-69 IPTM-Net 0.6186).
   - **Standalone Weighted Utility:** $\ge 0.4950$ (Surpassing EXP-55 DSCA-Net 0.4928).
   - **Standalone Substitution AP:** $\ge 0.4500$ with Substitution Precision $\ge 53.00\%$.
   - **Standalone Omission AP:** $\ge 0.4400$ with Omission Recall $\ge 52.00\%$.
   - **Standalone Overall ECE:** $\le 0.0125$ with False Warning Rate $\le 1.85$ per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6250$ (All-time standalone peak!).
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$ (All-time standalone peak!).
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4500$.
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and strict canonical target inputs.
