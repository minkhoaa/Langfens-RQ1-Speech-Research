# Research Plan: Experiment EXP-46 (Continuous Multi-Stress Robustness Benchmark)

## 1. Objective & Scope
Benchmark the robustness of the three top champion architectures under simulated real-world classroom and mobile capture stress perturbations on the 300 held-out utterances of `DEV_REPLICATION`:
1. **EXP-45 Decimal-Pareto** (10-Model Consensus)
2. **EXP-44 DMCT-Net** (Masked Context Invariance)
3. **EXP-38 HTC-Net** (Hierarchical Baseline Control)

## 2. Stress Conditions Evaluated
1. **Condition 1 (Clean Baseline):** Unperturbed 16kHz WAV audio.
2. **Condition 2 (Dynamic Syllabic Tempo Perturbation):** Syllable durations randomly jittered by $\pm 15\%$ via temporal interpolation, shifting speaking rate and duration contrast ratios.
3. **Condition 3 (Pitch Shift & Spectral Warping):** Formant/pitch frequencies perturbed by $\pm 12\%$ simulating child/adolescent vocal tract shifts.
4. **Condition 4 (Combined Severe Stress):** Joint application of tempo perturbation, dynamic range non-linear soft clipping (simulating smartphone mic saturation), and additive background classroom babble (+15dB SNR).

## 3. Metrics & Audit Protocol
- Deploy strictly with locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1` and locked thresholds ($\theta_s=0.55, \theta_o=0.55, \theta_i=0.55$ for Decimal-Pareto; $\theta_s=0.55, \theta_o=0.55, \theta_i=0.60$ for DMCT-Net).
- Measure Retention Ratio: $\rho = \text{Utility}_{\text{stress}} / \text{Utility}_{\text{clean}}$.
- Record Macro AP, Macro Precision, Omission AP, Substitution AP, Insertion F1, and FWR across all conditions.
- Save predictions and metrics under `experiments/EXP-46-CONTINUOUS-ACCENT-ROBUSTNESS-STRESS/`.
