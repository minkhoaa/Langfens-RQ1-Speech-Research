# Experiment Review: EXP-56-CONFORMER-HRTD-HYBRID-ATTENTION (CHTD-Net)

## 1. Executive Summary
Experiment EXP-56 implemented and evaluated the **Conformer High-Resolution Temporal Dynamics Network (CHTD-Net)**, integrating a 2-block Conformer architecture (Macaron-style feedforward, multi-head self-attention, and depthwise convolution) with 5ms sub-frame temporal micro-dynamics and bidirectional audio-prompt cross-attention.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **All-Time Lowest Calibration Error:**
  * **Overall ECE = 0.0111 (1.11%)** (All-time low across all 56 experiments!).
  * **Gap ECE = 0.0049 (0.49%)**.
- **High Omission Recall Sensitivity:**
  * Omission Recall reached **53.31%** on DEV_REPLICATION and **58.37%** on DEV_SEARCH.
  * Substitution F1 reached **0.4447** (Recall = 49.27%).
- **Macro Performance:**
  * Macro AP = **0.5792**, Macro F1 = **0.5933**, Weighted Utility = **0.4537**.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H51 Evaluated:**
  * Depthwise convolution successfully improves local receptive field sensitivity for plosive bursts (53.31% omission recall) and produces exceptional probability calibration (ECE = 0.0111).
  * However, BiLSTM with Monotonic Local Attention (DSCA-Net EXP-55 / HRTD-Net EXP-48) achieves higher precision on our 1,800-utterance training set by avoiding convolutional parameter overhead.
  * **DSCA-Net (EXP-55) remains the primary standalone champion (Utility = 0.4928), and Undecim-Pareto (EXP-52) remains the ensemble champion (Macro AP = 0.6244, Macro F1 = 0.6129).**
