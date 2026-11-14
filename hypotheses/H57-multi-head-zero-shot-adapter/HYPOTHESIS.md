# Hypothesis H57: Vocal Tract Length Normalized Dual-Stream Cross-Attention (DAMH-Net) for Domain-Invariant Transfer

## 1. Context & Motivation
Cross-corpus evaluations in EXP-41 and EXP-51 on SpeechOcean762 (4,536 expert-annotated phonemes) exposed a clear domain shift:
- **BiLSTM (EXP-08)** achieved Pearson $r = +0.2690$ ($\rho = +0.3132$) and **DMCT-Net (EXP-44)** achieved $r = +0.2532$ ($\rho = +0.3028$).
- However, models trained exclusively on adult L2-ARCTIC waveforms (such as HRTD-Net, $r = 0.1731$) suffered when exposed to child and adolescent speakers in SpeechOcean762 whose vocal tract lengths are shorter, causing higher fundamental frequencies ($F_0$) and shifted vowel formants ($F_1, F_2$).
- In EXP-42, Vocal Tract Length Normalization (VTLN) proved effective in warping spectral centroids $\alpha_{\text{vtln}} = C_{\text{ref}} / C_{\text{utt}}$.

In EXP-55, we established **DSCA-Net (Dual-Stream Cross-Attention Network)** as the laboratory's peak standalone architecture (Weighted Utility = 0.4928). However, DSCA-Net has not yet been coupled with VTLN formant warping.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **DAMH-Net (Domain-Adaptive Multi-Head Cross-Attention Network)** by integrating VTLN formant warping ($\alpha_{\text{vtln}}, C_{\text{rel}}, \log \alpha_{\text{vtln}}$) into the dual-stream cross-attention framework:
   - Spectral centroids will be dynamically warped toward canonical adult vocal tract scales ($C_{\text{ref}} = 1850$ Hz).
   - The canonical prompt stream will cross-attend to vocal-tract-normalized acoustic embeddings, eliminating age and speaker vocal tract length mismatch.
2. DAMH-Net will achieve:
   - On **SpeechOcean762-Test (4,536 phonemes)**: Zero-shot Pearson correlation breaking the 0.3000 barrier ($r \ge 0.3000$, $p < 10^{-90}$), with Spearman $\rho \ge 0.3300$ and Error AP $\ge 0.1400$.
   - On **L2-ARCTIC DEV_REPLICATION**: Sustaining Macro AP $\ge 0.6150$, Macro F1 $\ge 0.6020$, and Weighted Utility $\ge 0.4900$.

## 3. Predicted Experimental Outcomes
1. **SpeechOcean762 Zero-Shot Pearson $r$:** $\ge 0.3000$ (Breaking previous campaign record of 0.2690).
2. **SpeechOcean762 Spearman $\rho$:** $\ge 0.3300$.
3. **L2-ARCTIC DEV_REPLICATION Macro AP:** $\ge 0.6150$.
4. **L2-ARCTIC DEV_REPLICATION Weighted Utility:** $\ge 0.4900$.
