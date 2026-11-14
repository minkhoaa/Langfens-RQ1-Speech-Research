# Track A: Acoustic Evidence Research
**Lead Agent:** Agent A  
**Source Repository:** `/home/khoa/Projects/Pronunciation-Research-EXP101`  
**Mission:** Improve detection of phoneme omission and substitution using genuinely discriminative acoustic evidence.

## Active Status
- Baseline Experiment: `EXP-A101` (`exp_a101_energy_gated_l6.py`)
- Key Findings: Layer-6 frame contrast without metric regularization achieves high omission recall (90.9%) but incurs high FWR (11.29 warnings/utt).
- Next Step: `EXP-A102` — Cosine Margin Whitened Prototype Contrast to suppress acoustic false alarms while preserving omission sensitivity.
