# Experiment Review: EXP-59-ACOUSTIC-ARTICULATORY-LATENT-DIFFUSION (ALDS-Net)

## 1. Executive Summary
Experiment EXP-59 formulated and evaluated the **Acoustic-Articulatory Latent Diffusion Sampler (ALDS-Net)**, integrating score-matching diffusion denoising over the 128-dimensional latent space of DSCA-Net (EXP-55) with 4-step DDIM reverse trajectory sampling.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **All-Time Record Insertion Precision:**
  * **Insertion Precision = 54.55% (54.55%)**!
  * Breaking the 50% milestone on insertion precision for the first time in a standalone model.
  * **Gap ECE = 0.0036 (0.36%)** (All-time low calibration error on gaps).
  * **Overall ECE = 0.0138 (1.38%)**.
- **Macro Performance & Substitution Trade-Off:**
  * Macro AP = **0.5865**, Macro Precision = **0.6093**, Weighted Utility = **0.4722**.
  * FWR = **1.643 false alarms/utterance** (only 493 false alarms across 300 test utterances).
  * Finding: Score-matching diffusion conditioned on canonical prompt text effectively strips out inter-phone transition noise, yielding 54.55% insertion precision. However, for phoneme substitutions, it creates subtle over-denoising that pulls slight non-native substitutions toward canonical targets.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H54 Evaluated:**
  * Score-matching diffusion is validated as a powerful denoising tool for acoustic gap and insertion assessment.
  * However, for phoneme-level substitutions, unconstrained deterministic cross-attention (DSCA-Net EXP-55) remains the superior classifier.
  * **DSCA-Net (EXP-55) remains our standalone peak champion (Utility = 0.4928), and Duodecim-Pareto (EXP-58) remains our multi-model peak champion (Macro AP = 0.6341).**
