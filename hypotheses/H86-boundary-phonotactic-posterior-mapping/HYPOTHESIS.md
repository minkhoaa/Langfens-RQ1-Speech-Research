# Hypothesis H86: Boundary-Aware Phonotactic Posterior Mapper Network (BAPM-Net)

## 1. Context & Motivation
Across previous standalone cycles, the laboratory isolated two complementary mechanisms:
1. **EXP-69 (IPTM-Net):** Standalone record for Macro F1 (0.6205) and Omission Recall (55.64%) via empirical interlanguage error transfer priors $\pi(p)$.
2. **EXP-68 (MRTP-Net) & EXP-77 (TPD-Net):** Multi-resolution temporal pyramid features (2.5ms micro-burst crests to 20ms context) achieved all-time record Macro Recall (66.37%) and sub-1% calibration (0.0083 ECE).

However, in all existing models, phonotactic priors $\log P(p_{k+1} \mid p_k)$ and interlanguage error priors $\pi(p)$ are fused statically and uniformly across all phonetic slots, regardless of local acoustic certainty. This causes prior hallucination on clear steady-state vowels while under-utilizing priors at noisy plosive boundaries.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Boundary-Aware Phonotactic Posterior Mapper Network (BAPM-Net)** by:
   - Defining a continuous acoustic boundary certainty gate:
     $$\alpha_{\text{gate}}(k) = \sigma\left(W_c \cdot [r_{\text{dur}}(k), C_{2.5\text{ms}}(k), \Delta e_{\text{onset}}(k)] + b_c\right)$$
     derived from 2.5ms micro-burst dynamics and duration ratios.
   - Dynamically modulating the injection of interlanguage error priors $\pi(p_k)$ and phonotactic bigrams into the prompt conditioning stream:
     $$X_{p, \text{adapted}}(k) = X_{\text{canonical}}(k) + \alpha_{\text{gate}}(k) \cdot \left[W_{\pi} \pi(p_k) + W_g \log P_{\text{trans}}(k)\right]$$
     * In steady-state vowels ($\alpha_{\text{gate}} \to 0.2$), the model relies strictly on acoustic evidence.
     * At rapid boundaries and plosive bursts ($\alpha_{\text{gate}} \to 0.9$), the model engages strong interlanguage priors to resolve ambiguous consonants.
   - Coupling with 72-dimensional multi-resolution temporal pyramid acoustic features and 13-dimensional triad decoupled gap gating.
2. BAPM-Net will achieve:
   - **Standalone Macro F1:** Breaking the 62.5% milestone for the first time in laboratory history ($\text{Macro F1} \ge 0.6250$).
   - **Standalone Macro AP:** $\ge 0.6220$ (62.20%) (Surpassing HRTD-Net 0.6202 and IPTM-Net 0.6186).
   - **Standalone Weighted Utility:** $\ge 0.4950$ (49.50%) (Surpassing DSCA-Net 0.4928).
   - **Substitution AP:** $\ge 0.4550$ with Substitution Precision $\ge 54.00\%$.
   - **Omission Recall:** $\ge 55.50\%$ with Omission F1 $\ge 0.4680$.
   - **Overall ECE:** $\le 0.0115$ (1.15%) with False Warning Rate $\le 1.45$ per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro F1:** $\ge 0.6250$ (All-time standalone peak!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6220$ (All-time standalone peak!).
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$ (All-time standalone peak!).
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and strict canonical target inputs.
