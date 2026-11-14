# Hypothesis H84: Progressive Confidence-Cascaded Viginti Router (PCR-Engine)

## 1. Context & Motivation
Our 21-model super-ensemble (EXP-86 Viginti-Pareto) established the all-time laboratory record for Weighted Utility (**0.5079 / 50.79%**) and Macro F1 (**0.6292 / 62.92%**). However, evaluating all 21 models for every phoneme token imposes substantial latency on edge devices and CPUs.

In natural L2 speech, $\approx 80\%$ of phonemes are produced correctly and unambiguously, where lightweight acoustic models (BiLSTM, DMCT-Net, HRTD-Net) already exhibit $>98\%$ classification certainty with minimal entropy ($H < 0.25$).

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Progressive Confidence-Cascaded Viginti Router (PCR-Engine)** with:
   - **Tier 1 (Fast Linear Gating):** Run lightweight models (BiLSTM, DMCT-Net, HRTD-Net, and TDIS-Net). If confidence $\max_c p_{\text{fast}, c} \ge \tau_{\text{exit}} = 0.82$, exit immediately.
   - **Tier 2 (Heavy Ensemble Consensus):** Only for ambiguous tokens ($\max_c p_{\text{fast}, c} < 0.82$, approximately 20-30% of tokens), invoke the remaining 17 models (diffusion samplers, VTLN cepstral adapters, sonority mappers) to compute full Viginti-Pareto consensus.
2. PCR-Engine will achieve:
   - **Compute Speedup:** Over 70% of tokens exit at Tier 1, reducing CPU latency to $< 5.0$ ms per utterance.
   - **Parity Retention:** Retaining $\ge 99.5\%$ of Viginti-Pareto's Weighted Utility ($\text{Utility} \ge 0.5050$).
   - **Macro F1:** Sustaining $\ge 0.6250$ (62.50%).
   - **Macro Recall:** Sustaining $\ge 61.50\%$ (61.50%).
   - **Overall ECE:** Maintaining $\le 0.0270$ (2.70%).

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5050$ (Retaining champion performance!).
2. **DEV_REPLICATION Macro F1:** $\ge 0.6250$.
3. **Tier 1 Early Exit Rate:** $\ge 70.0\%$.
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and canonical target inputs.
