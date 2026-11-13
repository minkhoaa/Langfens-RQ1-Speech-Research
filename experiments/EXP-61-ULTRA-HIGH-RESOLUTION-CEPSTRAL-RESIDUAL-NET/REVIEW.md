# Experiment Review: EXP-61-ULTRA-HIGH-RESOLUTION-CEPSTRAL-RESIDUAL-NET (UCR-Net)

## 1. Executive Summary
Experiment EXP-61 implemented and evaluated the **Ultra-High-Resolution Cepstral Residual Network (UCR-Net)**, augmenting the 67-dimensional acoustic stream of DSCA-Net with 8-dimensional empirical spectral cepstral residual deviations ($\Delta_{\text{cep}} = \text{MFCC}_{1-8} - \bar{C}_{\text{canonical}}$) computed strictly from `TRAIN` partition audio waveforms.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **All-Time Record Standalone Insertion AP:**
  * **Insertion AP = 0.2945 (29.45%)**!
  * (Breaking the previous standalone peak of 0.2825 in EXP-59).
  * Insertion Recall = **35.11%**, Insertion F1 = **0.3511**.
- **Low False Alarm Rate & High Calibration:**
  * Total false alarms dropped to **446** across 300 test utterances (FWR = **1.487 false alarms/utterance**).
  * **Target ECE = 0.0110 (1.10%)** (Exceptional probability calibration on target phones).
  * **Overall ECE = 0.0173 (1.73%)**.
- **Macro Performance & Substitution Analysis:**
  * Macro AP = **0.5953**, Macro Precision = **0.6163**, Weighted Utility = **0.4726**.
  * Finding: Cepstral residuals provide clean boundary and energy transition cues for insertion detection, but unadapted MFCCs introduce subtle speaker-dependent timbre variance that requires speaker-level normalization to prevent blunting vowel substitution precision.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H56 Evaluated:**
  * Cepstral residuals decisively improve insertion detection (Insertion AP = 0.2945) and lower false alarms to 1.487 per utterance.
  * For phoneme substitutions, unconstrained bidirectional cross-attention without raw MFCC residuals (DSCA-Net EXP-55) remains the superior classifier.
  * **DSCA-Net (EXP-55) remains our standalone peak champion (Utility = 0.4928), and Duodecim-Pareto (EXP-58) remains our multi-model peak champion (Macro AP = 0.6341).**
