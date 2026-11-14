# Research Plan: Hypothesis H09 (Subgroup Robustness)

1. **Subgroup Mapping:**
   - Map each held-out speaker to their L1 native language:
     - Arabic: YBAA
     - Chinese: NCC, TXHC
     - Hindi: RRBI
     - Korean: YDCK, YKWK
     - Spanish: ERMS
     - Vietnamese: THV
2. **Model Evaluation:**
   - Run forward inference of HM-ADN on all held-out sequences (1,199 utterances, ~40k tokens).
   - Apply frozen decision policy ($\theta_{\text{subs}}=0.55, \theta_{\text{omis}}=0.55, \theta_{\text{ins}}=0.55$).
3. **Statistical Aggregation:**
   - Evaluate PronunciationEvaluator metrics individually per L1 group.
   - Run 1,000 bootstrap resamples per L1 to compute 95% Confidence Intervals for Macro F1, Substitution F1, and Omission F1.
4. **Artifacts & Ledger:**
   - Save subgroup breakdown tables and review report in `experiments/EXP-14-CROSS-VALIDATION-SWEEP/`.
