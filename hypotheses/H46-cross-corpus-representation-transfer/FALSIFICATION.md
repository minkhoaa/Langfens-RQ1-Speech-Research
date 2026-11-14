# Falsification Criteria: Hypothesis H46

Hypothesis H46 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating HRTD-Net, DMCT-Net, and Decimal-Pareto zero-shot on the 4,536 phonemes of SpeechOcean762-Test:

1. **Correlation Collapse:**
   - HRTD-Net Pearson correlation drops below $0.2000$ ($r < 0.2000$) or fails to achieve statistical significance ($p \ge 10^{-10}$).
2. **Error Detection Regression:**
   - Decimal-Pareto Error AP drops below $0.1100$ (inferior to uncalibrated baseline).
   - Mispronunciation Error Recall falls below $50.0\%$.
3. **Severe Cross-Domain Degradation:**
   - Both HRTD-Net and Decimal-Pareto exhibit Spearman rank correlation $\rho < 0.2500$, indicating that feature representations fail under Mandarin-accented adolescent speech.
