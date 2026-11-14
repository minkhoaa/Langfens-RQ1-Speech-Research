# Hypothesis H60: Phonotactically Guided Acoustic Latent Diffusion (P-ALDS-Net)

## 1. Context & Motivation
Laboratory findings from EXP-59 (ALDS-Net) and EXP-34 (PNC-Net) established two critical mechanisms:
1. **Latent Score-Matching Diffusion (EXP-59):** Achieved an all-time record for standalone Insertion Precision (54.55%) with Gap ECE = 0.0036. However, score matching conditioned solely on canonical prompt targets pulled subtle phonetic substitutions toward canonical centroids, slightly blunting substitution discrimination (Subs AP = 0.3957 vs DSCA-Net 0.4320).
2. **Lexical-Phonotactic Transition Conditioning (EXP-34):** Proved that CMUDict bigram transition probabilities $P(p_k \mid p_{k-1})$ provide strong statistical constraints that differentiate non-native phonotactic errors from natural acoustic coarticulation.

The root cause of over-denoising in ALDS-Net was the *absence of phonotactic transition constraints*: the denoiser had no knowledge of whether a realized acoustic sequence was phonotactically permissible or an illegal mother-tongue transfer.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **P-ALDS-Net (Phonotactically Guided Acoustic Latent Diffusion Sampler)**, which conditions the score-matching denoiser $\epsilon_\theta(z_t, t, c_{\text{prompt}}, \mathcal{T}_{\text{phonotactic}})$ on both:
   - Canonical phoneme prompt embeddings ($c_{\text{prompt}} \in \mathbb{R}^{128}$)
   - Forward and backward lexical bigram log-probabilities ($\log P(p_k \mid p_{k-1})$, $\log P(p_{k+1} \mid p_k)$)
   - Triad decoupled gap routing (EXP-63)
   will prevent over-denoising on genuine phonotactically irregular substitutions while aggressively stripping coarticulatory noise from canonical frames.
2. In particular:
   - For **`SUBSTITUTION`**, when an acoustic realization deviates along a phonotactically illegal direction, the phonotactic prior inhibits canonical projection, preserving substitution divergence ($\text{Substitution AP} \ge 0.4500$).
   - For **`INSERTION`**, triad decoupled gap gating combined with diffusion denoising will maintain $\text{Insertion Precision} \ge 52.00\%$ and $\text{Insertion F1} \ge 0.4200$.
3. P-ALDS-Net will achieve:
   - **Weighted Utility** breaking the 0.5000 milestone for the first time in a single standalone model ($\text{Utility} \ge 0.5000$).
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Macro F1** sustaining $\ge 0.6050$.
   - **Omission AP** maintaining $\ge 0.4400$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (First standalone single model to break 50%!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4500$ (resolving ALDS-Net over-denoising).
4. **DEV_REPLICATION Insertion F1:** $\ge 0.4200$.
