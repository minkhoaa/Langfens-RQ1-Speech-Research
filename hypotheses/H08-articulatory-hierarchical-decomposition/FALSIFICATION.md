# Falsification Criteria for Hypothesis H08

1. **Articulatory Gain Falsification:**
   If HM-ADN fails to achieve higher Average Precision or higher F1 on `SUBSTITUTION` than the baseline single-task models (EXP-08 / EXP-10 / EXP-11) on `DEV_SEARCH`, H08 is rejected.

2. **Negative Multi-Task Transfer:**
   If auxiliary articulatory losses cause optimization conflict and degrade `CORRECT` F1 below 0.90, the multi-task formulation is rejected.
