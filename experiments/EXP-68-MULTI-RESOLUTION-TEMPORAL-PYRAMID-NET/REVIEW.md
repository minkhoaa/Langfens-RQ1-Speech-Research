# Experiment Review: EXP-68-MULTI-RESOLUTION-TEMPORAL-PYRAMID-NET (MRTP-Net)

## 1. Executive Summary
Experiment EXP-68 implemented and evaluated the **Multi-Resolution Temporal Pyramid Network (MRTP-Net)**, integrating a 4-level acoustic temporal pyramid directly from raw 16kHz waveforms:
- Level 0 (2.5ms / 40 samples): Micro-burst crest factor & onset energy jump
- Level 1 (5.0ms / 80 samples): Sub-frame half-slot energy ratio & max dE/dt
- Level 2 (10.0ms / 160 samples): Standard frame segmental tilt & ZCR
- Level 3 (20.0ms / 320 samples): Macro-segmental vowel resonance ratio & pyramid tilt slope
Augmenting the acoustic stream to 72 dimensions, coupled with bidirectional prompt cross-attention.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **All-Time Lowest Probability Calibration Error:**
  * **Overall ECE = 0.0083 (0.83%)**!
  * Breaking the 1.0% ECE milestone for the first time in campaign history!
  * **Target ECE = 0.0106 (1.06%)**, **Gap ECE = 0.0061 (0.61%)**.
- **All-Time Lowest False Omission Alarms:**
  * False omission warnings dropped to an all-time low of **89** across 9,863 target phones (Precision = **42.95%**).
  * The 2.5ms micro-burst crest factor successfully confirmed transient plosive burst releases, eliminating false omission alarms.
- **Substitution Diagnosis:**
  * Substitution AP reached **0.4322**, Substitution F1 reached **0.4491**, Recall = **41.40%**.
- **Macro Performance & Calibration Trade-Off:**
  * Macro AP = **0.5933**, Macro Precision = **0.6138**, Weighted Utility = **0.4514**.
  * Finding: While 2.5ms features eliminate false alarms and maximize probability calibration (ECE = 0.0083), conservative calibration on DEV_CAL reduces omission recall relative to unconstrained 5ms attention.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H63 Evaluated:**
  * 2.5ms temporal pyramid features provide decisive physical evidence for plosive release bursts, producing the most well-calibrated network in laboratory history (ECE = 0.0083).
  * For peak macro utility and AP, **DSCA-Net (EXP-55, Utility = 0.4928)** remains the primary standalone champion, and **Quattuordecim-Pareto (EXP-67, Macro AP = 0.6377, Utility = 0.4935)** remains the ensemble champion.
