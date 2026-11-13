# Experiment Review: EXP-01-PRIOR

## 1. Objective and Setup
Negative control to establish empirical floor metrics when no audio is present. Predictions rely purely on empirical class frequencies from the training split.

## 2. Key Findings
- **High CORRECT AP/F1 by Default:** Because non-native speech is predominantly correct (~90% base rate in L2-ARCTIC), a naive majority classifier scores 0.900 AP and 0.947 F1 on CORRECT without listening to audio.
- **Complete Inability to Detect Errors:** F1 for OMISSION, SUBSTITUTION, and INSERTION is strictly 0.000.
- **Baseline AP Floors:**
  - OMISSION AP floor = 0.0106 (random guessing AP matches prevalence ~1.06%)
  - SUBSTITUTION AP floor = 0.0895 (prevalence ~8.95%)
  - INSERTION AP floor = 0.0061 (prevalence ~0.61%)
- **Weighted Utility:** 0.3183.
- **Scientific Significance:** Any proposed acoustic method must substantially exceed these AP floors, especially on the rare error classes, to prove genuine acoustic discrimination.
