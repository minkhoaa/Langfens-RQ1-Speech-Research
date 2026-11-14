# Hypothesis H09: L1 Accent Invariance and Subgroup Uncertainty

## 1. Proposition
Phonetic assessment models trained on multi-accent non-native speech often develop latent biases toward specific L1 mother tongues (e.g. over-flagging phonetic substitutions on tonal language speakers like Chinese or Vietnamese, while under-detecting omissions on rhotic Hindi speakers).

The Hierarchical Articulatory model (HM-ADN) abstracts surface acoustics into universal Place, Manner, and Voicing dimensions. We hypothesize that:
1. HM-ADN will maintain invariant performance across all 6 L1 language groups (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese) with Macro F1 variance $\sigma_{\text{L1}} \le 0.08$.
2. The model will not exhibit catastrophic failure ($F1 < 0.25$ on substitutions or omissions) on any single L1 group.
3. Ninety-five percent confidence intervals (95% CI) computed via non-parametric bootstrapping will confirm statistically stable macro metrics.

## 2. Experimental Plan
- Method ID: `M14-l1-group-cross-validation`
- Run ID: `EXP-14-CROSS-VALIDATION-SWEEP`
- Evaluate across all 6 L1 held-out speaker cohorts (1,199 utterances total).
- Report per-L1 confusion matrices, class metrics, calibration, and bootstrap confidence intervals.
