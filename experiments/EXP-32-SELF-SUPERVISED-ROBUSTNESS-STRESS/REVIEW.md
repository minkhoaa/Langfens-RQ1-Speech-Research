# Experiment Review: EXP-32-SELF-SUPERVISED-ROBUSTNESS-STRESS

## 1. Executive Summary
Experiment EXP-32 tested Hypothesis H27 by evaluating the champion Hexad-Pareto ensemble across four acoustic Signal-to-Noise Ratio (SNR) regimes: Clean ($\infty$ dB), $+20$ dB SNR, $+10$ dB SNR, and $0$ dB SNR on `DEV_REPLICATION`.

## 2. Quantitative Findings & Degradation Trajectory
| SNR Level | Weighted Utility | Retention (%) | Macro AP | Macro Precision | Macro F1 | False Warning Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean ($\infty$ dB)** | **0.4321** | 100.0% | **0.5810** | 0.6193 | **0.5801** | 1.82 |
| **+20 dB SNR** (Light Noise) | **0.4308** | **99.7%** | **0.5820** | **0.6205** | **0.5757** | 1.77 |
| **+10 dB SNR** (Classroom Noise) | **0.4138** | **95.8%** | **0.5720** | **0.6286** | **0.5633** | 1.61 |
| **0 dB SNR** (Extreme Noise) | **0.3242** | **75.0%** | **0.5084** | **0.6177** | **0.4624** | **1.02** |

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H27 Decisively Confirmed:**
  * At $+10$ dB SNR (representing typical classroom chatter or background HVAC noise), Hexad-Pareto preserves **$95.8\%$** of its diagnostic utility ($0.4138$), and Macro Precision actually improves to **$62.86\%$**.
  * Even under extreme $0$ dB SNR where noise power equals speech power, Hexad-Pareto maintains a Utility of $0.3242$ (higher than clean baseline GOP!), with Macro Precision remaining strictly above the 60% milestone ($61.77\%$).
  * False Warning Rate steadily decreased from 1.82 to 1.02, confirming that the Bayesian mixture gracefully increases abstention rather than producing spurious false alarms under severe environmental distortion.
