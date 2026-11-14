# Falsification Criteria: Hypothesis H100

Hypothesis H100 will be declared **FALSIFIED** if:
1. Negative transfer occurs on held-out L2-ARCTIC `DEV_REPLICATION`:
   - Standalone Macro F1 falls below 0.6050.
   - Standalone Weighted Utility falls below 0.4800.
2. Cross-corpus SpeechOcean762 transfer correlation fails to improve:
   - Pearson $r < 0.2400$ or Spearman $\rho < 0.3000$.
3. In-domain False Warning Rate exceeds 3.30 false alarms per utterance.
