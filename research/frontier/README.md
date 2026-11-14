# Track B: Sequence & Insertion Research
**Lead Agent:** Agent B  
**Source Repository:** `/home/khoa/Projects/Pronunciation-Frontier-Research`  
**Mission:** Improve sequence-level error localization, especially genuine exact-gap insertion detection.

## Active Status
- Baseline Experiment: `EXP-B101` (`exp_b101_spectral_flux_insertion.py`)
- Key Findings: Across 20,354 gaps in DEV_SEARCH, acoustic frame duration and spectral flux distinguish insertions (4.12s, 0.080 flux vs 1.27s, 0.013 flux). Initial classifier achieves Precision 10.34%, Recall 7.20%, FWR 0.1302 err/utt.
- Next Step: `EXP-B102` — Coarticulation-filtered boundary flux fusion to suppress vowel transition artifacts and improve exact-gap insertion precision.
