# Hypothesis H72: Temporal-Pyramid Diffusion-Coupled Network (TPD-Net)

## 1. Context & Motivation
Across previous cycles, the laboratory established:
1. **EXP-68 (MRTP-Net):** Multi-resolution temporal pyramid features (2.5ms micro-burst crest factor + 5ms onset dynamics) broke the 1.0% ECE barrier for the first time in laboratory history (Overall ECE = 0.0083 / 0.83%), dropping false omissions to an all-time low of 89 across 9,863 target phones.
2. **EXP-65 (P-ALDS-Net):** Phonotactic latent score diffusion achieved an all-time standalone Substitution Precision of 55.83% and Substitution AP = 0.4485.
3. **EXP-69 (IPTM-Net):** Empirical interlanguage transfer priors $\pi(p)$ achieved an all-time standalone Macro F1 of 0.6205 and Omission Recall of 55.64%.

Currently, score-matching diffusion has only been paired with standard 5ms acoustic features (67 dims), lacking the 2.5ms micro-burst temporal pyramid features that prevent false omission errors during rapid plosive releases and flap transitions.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Temporal-Pyramid Diffusion-Coupled Network (TPD-Net)** by:
   - Augmenting the acoustic stream to 71 dimensions by injecting 4-dimensional 2.5ms micro-burst dynamics (energy step, micro-crest factor, onset burst differential) into the 67-dim base HRTD vector.
   - Conditioning the 47-dimensional prompt stream with empirical interlanguage error transfer priors $\pi(p)$ and CMUDict bigrams.
   - Training a score-matching diffusion network with 4-step DDIM deterministic posterior sampling to denoise acoustic representations conditioned on multi-resolution temporal pyramid features.
   - Coupling with 13-dimensional triad decoupled gap gating (pause, cluster, coda).
2. TPD-Net will achieve:
   - **Probability Calibration:** Breaking the 0.80% ECE barrier ($\text{Overall ECE} \le 0.0080$).
   - **False Omission Suppression:** False omission alarms $\le 95$ across 9,863 target phones.
   - **Standalone Substitution AP:** $\ge 0.4550$ (45.50%) with Substitution Precision $\ge 56.00\%$.
   - **Standalone Macro F1:** $\ge 0.6150$ (61.50%).
   - **Standalone Weighted Utility:** $\ge 0.4900$ (49.00%).
   - **False Warning Rate:** $\le 1.45$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Overall ECE:** $\le 0.0080$ (All-time campaign peak calibration!).
2. **DEV_REPLICATION Substitution Precision:** $\ge 56.00\%$.
3. **DEV_REPLICATION Macro F1:** $\ge 0.6150$.
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and strict canonical target inputs.
