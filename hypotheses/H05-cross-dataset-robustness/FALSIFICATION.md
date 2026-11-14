# Falsification Criteria for Hypothesis H05

1. **Correlation Falsification:**
   If the Pearson correlation $r$ between model predictions and human expert phoneme scores on SpeechOcean762 is $< 0.40$ or not statistically significant ($p \ge 0.01$), H05 is falsified, indicating dataset overfitting to L2-ARCTIC.

2. **Domain Collapse:**
   If the model's predictions collapse to predicting a single score or variance $\approx 0$ under non-native child speech, H05 is falsified.
