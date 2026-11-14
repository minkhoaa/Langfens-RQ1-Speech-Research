# Hypothesis H48: Dialect-Invariant Feature Modulation (DIAA-Net) for Cross-L1 Generalization

## 1. Context & Motivation
English L2 learners from different mother tongues (L1s) exhibit systematic, predictable phonological transfer patterns:
- **Arabic L1**: Epenthetic vowel insertion in initial consonant clusters and `/p/` $\rightarrow$ `/b/` substitution.
- **Chinese/Mandarin L1**: Final consonant coda deletion (stop consonant omissions) and liquid `/l/`/`/r/` confusion.
- **Hindi L1**: Retroflex stop realizations and aspiration contrasts.
- **Korean L1**: Coda neutralization and vowel epenthesis.
- **Spanish L1**: Pre-consonantal `/e/` epenthesis before initial `/s/` clusters and dental/labiodental fricative confusion (`/b/`/`/v/`).
- **Vietnamese L1**: Coda consonant cluster reductions and tone-induced duration lengthening.

In EXP-35, the baseline system achieved cross-accent consistency with $\sigma_{\text{L1}} = 0.0725$. However, standard models treat all speakers with static LayerNorm parameters, forcing the model to compromise between different L1 acoustic baselines.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Introducing **DIAA-Net (Dialect-Invariant Acoustic Adapter Network)**, which extracts an utterance-level acoustic dialect vector $e_{\text{dialect}} \in \mathbb{R}^{16}$ (derived from speaking tempo, spectral tilt, pitch mean/variance, and high-frequency centroid distribution) and modulates the internal feature space via **Feature-wise Linear Modulation (FiLM)**:
   $$\tilde{h} = \gamma(e_{\text{dialect}}) \odot \text{LayerNorm}(h) + \beta(e_{\text{dialect}})$$
   will dynamically compensate for L1-specific phonological transfer patterns without requiring speaker identity labels at test time.
2. DIAA-Net applied to **HRTD-Net (5ms sub-frame micro-dynamics)** will:
   - Reduce cross-L1 variance on `DEV_REPLICATION` and all 6 L1 cohorts to $\sigma_{\text{L1}} \le 0.0550$.
   - Eliminate L1-specific coda false omission alarms in Chinese/Vietnamese speakers and epenthesis false alarms in Spanish/Arabic speakers.
   - Elevate standalone Macro AP to $\ge 0.6250$ (62.50%).
   - Elevate Weighted Utility to $\ge 0.4850$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
2. **Cross-L1 Variance $\sigma_{\text{L1}}$:** $\le 0.0550$.
3. **DEV_SEARCH Omission AP:** $\ge 0.3250$.
4. **Weighted Utility:** $\ge 0.4850$.
