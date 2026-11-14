# Research Plan: Hypothesis H27 / Experiment EXP-32

## 1. Objective
Quantify the acoustic noise degradation curve and stress resilience of Hexad-Pareto, APED-Net, and SSL-GOP under controlled additive Gaussian noise levels ($\text{SNR} \in \{20\text{ dB}, 10\text{ dB}, 0\text{ dB}\}$) on `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Stress Formulation:**
   - For each utterance on `DEV_REPLICATION`, simulate additive acoustic white Gaussian noise at:
     * Clean ($\infty\text{ dB}$)
     * $+20\text{ dB}$ SNR (light background noise)
     * $+10\text{ dB}$ SNR (typical classroom ambient noise)
     * $0\text{ dB}$ SNR (severe noise equal in power to speech signal)
   - Add noise directly to the dense acoustic feature vectors according to signal variance:
     $\sigma_{\text{noise}}^2 = \sigma_{\text{signal}}^2 \times 10^{-\text{SNR} / 10}$.
2. **Evaluation Across SNR Regimes:**
   - Evaluate Hexad-Pareto, APED-Net, and baseline under each noise regime.
   - Measure degradation in Macro AP, Macro F1, Utility, and False Warning Rate.
3. **Ledger & Artifact Audit:**
   - Save degradation curves, metrics, and SHA-256 hashes under `experiments/EXP-32-SELF-SUPERVISED-ROBUSTNESS-STRESS/`.
