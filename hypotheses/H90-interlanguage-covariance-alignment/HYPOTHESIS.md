# Hypothesis H90: Cross-Accent Interlanguage Covariance Aligner (CICA-Net)

## 1. Context & Motivation
Non-native phonological transfer is structured by native language (L1) phonemic inventories. Phonetic errors do not occur independently:
- Spanish L1 speakers exhibit coupled vowel system reductions across /IY/-/IH/ and /EY/-/EH/.
- Arabic L1 speakers exhibit coupled obstruent voicing collapses across /P/-/B/ and /F/-/V/.
- Mandarin L1 speakers exhibit coupled coda plosive omissions across final /P/, /T/, /K/.
While IPTM-Net (EXP-69) modeled scalar phoneme priors $\pi(p)$, it ignored pairwise inter-phoneme error covariance.

## 2. Hypothesis Statement
We hypothesize that:
1. Computing empirical phoneme error covariance matrices $\Sigma_{\ell} \in \mathbb{R}^{39 \times 39}$ for each L1 cohort $\ell \in \{\text{Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese}\}$ strictly from `train.jsonl`:
   $$\Sigma_{\ell}[i, j] = \frac{1}{N_{\ell}} \sum_{u \in \text{Train}_{\ell}} (e_{u, i} - \bar{e}_i)(e_{u, j} - \bar{e}_j)$$
   and projecting the prompt representations through $\Sigma_{\ell}$ will:
   - Provide structured phonological co-occurrence priors that disambiguate coupled consonant/vowel substitutions.
   - Elevate standalone **Substitution F1 past 0.5000 (50.00%)**.
   - Sustain standalone **Weighted Utility $\ge 0.5000$** and **Macro F1 $\ge 0.6250$** on `DEV_REPLICATION`.
2. Falsification will occur if covariance projection causes over-smoothing across unrelated phonemes, dropping Macro F1 below 0.6150 or Utility below 0.4850.
