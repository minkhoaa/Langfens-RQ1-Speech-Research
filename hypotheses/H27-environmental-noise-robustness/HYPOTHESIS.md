# Hypothesis H27: Environmental Acoustic Noise Robustness & Stress Resilience

## 1. Context and Problem Statement
In real-world educational deployment, learner speech is captured via varying microphone hardware in non-ideal acoustic environments (e.g. ambient classroom chatter, HVAC hum, fan noise).
Prior experiments established high diagnostic accuracy in clean conditions:
- Hexad-Pareto: Macro AP = 0.5810, Macro F1 = 0.5801, Utility = 0.4321.
- APED-Net: Macro F1 = 0.5725, Substitution F1 = 0.4461, Insertion F1 = 0.3579.

However, single-model acoustic classifiers often experience catastrophic degradation under low Signal-to-Noise Ratios (SNR) because noise perturbs energy floors and posterior logits. Hypothesis H27 investigates whether the multi-model Hexad-Pareto ensemble exhibits graceful degradation, maintaining diagnostic utility under severe noise stress (+20 dB, +10 dB, and 0 dB SNR).

## 2. Formal Hypothesis
Subjecting input representations to controlled additive Gaussian acoustic noise at SNR $\in \{20\text{ dB}, 10\text{ dB}, 0\text{ dB}\}$ will demonstrate:
1. **Graceful Degradation of Hexad-Pareto:** Hexad-Pareto retains $\ge 85\%$ of its clean Utility at $+20\text{ dB}$ SNR (Utility $\ge 0.3650$) and $\ge 70\%$ at $+10\text{ dB}$ SNR (Utility $\ge 0.3000$).
2. **Superiority over Single Models:** Under severe noise ($0\text{ dB}$ SNR), Hexad-Pareto outperforms standalone baseline SSL-GOP and standalone APED-Net by at least $+0.0500$ in Utility, demonstrating ensemble variance reduction under acoustic domain shift.

## 3. Predicted Evidence & Significance
If validated, this confirms that architectural diversity acts as an effective regularizer against environmental acoustic noise, satisfying the robustness mandate of `AGENTS.md` Section 8.
