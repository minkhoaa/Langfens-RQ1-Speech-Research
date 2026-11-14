# Track C: Safe Detection & Calibration Research
**Lead Agent:** Agent C  
**Source Repository:** `/home/khoa/Projects/Pronunciation-Scientific-Recovery`  
**Mission:** Develop reliable pronunciation error decisions with low false-warning rates (FWR <= 0.20/utt) and useful error recall.

## Active Status
- Baseline Experiment: `EXP-C101` (`exp_c101_neyman_pearson_calibration.py`)
- Key Findings: Neyman-Pearson grid search on DEV_CAL (300 utts) with safety limit (tau_omis=0.94, tau_subs=0.90) generalizes safely to unexposed DEV_SEARCH (599 utts), achieving FWR = 0.1669 (PASSING <= 0.20 gate) with 58.55% precision on substitutions.
- Next Step: `EXP-C102` — Phone-Family Specific Risk Calibration (separate thresholds for vowels vs consonants vs stops) and Selective Abstention to further expand error recall under FWR <= 0.20.
