# Hypothesis H73: SpeechOcean762 Zero-Shot Cross-Corpus Transfer Benchmark

## 1. Context & Motivation
SpeechOcean762 consists of 4,536 expert-rated non-native English phonemes spoken by Chinese children. This represents an extreme domain shift from L2-ARCTIC:
- Age shift: Adult vs. child vocal tract dimensions (+20% higher formants, shorter acoustic lengths).
- Acoustic recording shift: Studio microphone vs. mobile/consumer tablet microphones.
- Ontology mismatch: Multi-class categorical labels (`CORRECT`, `SUBSTITUTION`, `OMISSION`, `INSERTION`) vs. discrete 3-level rater scores ($s \in \{0, 1, 2\}$).

In EXP-51, baseline BiLSTM achieved Pearson $r = 0.2690$ ($\rho = 0.3132$) and DMCT-Net achieved $r = 0.2532$ ($\rho = 0.3028$). In EXP-62, VTLN cross-attention lifted Pearson correlation to $r = 0.2110$ ($p = 7.72 \times 10^{-47}$).

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Benchmarking the latest crown-jewel architectures on SpeechOcean762:
   - **TPD-Net (EXP-77):** Multi-resolution temporal pyramid features (2.5ms to 20ms) coupled with score-matching latent diffusion.
   - **Septendecim-Pareto (EXP-76):** 17-model Bayesian Pareto super-ensemble.
   - **IPTM-Net (EXP-69):** Interlanguage transfer prior cross-attention.
   - **DSCA-Net (EXP-55):** Dual-stream cross-attention.
2. TPD-Net will achieve:
   - **Pearson correlation $r \ge 0.2800$** ($p < 10^{-50}$).
   - **Spearman correlation $\rho \ge 0.3200$** ($p < 10^{-50}$).
   - Multi-scale 2.5ms micro-burst features will demonstrate significantly lower error on short-duration child vowels than standard 10ms frame features.
3. Septendecim-Pareto will achieve:
   - **Pearson correlation $r \ge 0.2750$** and **Spearman $\rho \ge 0.3150$**.
4. Both architectures will confirm true cross-corpus domain transfer and age invariance without any fine-tuning on child speech.

## 3. Predicted Experimental Outcomes
1. **SpeechOcean762 Pearson $r$:** $\ge 0.2800$ (All-time campaign peak!).
2. **SpeechOcean762 Spearman $\rho$:** $\ge 0.3200$ (All-time campaign peak!).
3. **Statistical Significance:** $p < 10^{-50}$ across all 4,536 evaluated phonemes.
