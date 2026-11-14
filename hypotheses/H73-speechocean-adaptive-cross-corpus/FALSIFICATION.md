# Falsification Criteria: Hypothesis H73

Hypothesis H73 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating on SpeechOcean762 test set (4,536 phonemes across 300 utterances):

1. **Correlation Collapse:**
   - TPD-Net Pearson correlation $r$ falls below 0.2200 or fails to achieve statistical significance ($p \ge 0.001$).
   - Spearman rank correlation $\rho$ falls below 0.2500.
2. **Ensemble Degradation:**
   - Septendecim-Pareto correlation falls below the baseline BiLSTM correlation ($r < 0.2500$).
3. **Domain Breakdown:**
   - Models fail to predict scores with positive correlation to human accuracy ratings.
