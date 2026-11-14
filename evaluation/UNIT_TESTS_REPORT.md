# Evaluator Unit Tests & Invariant Verification Report

**Status:** PASSING (4/4 tests passed)  
**Date:** 2026-11-04  
**Test Suite:** `evaluation/tests/test_evaluator.py`, `evaluation/tests/test_leakage.py`  

---

## 1. Test Invariants Verified

1. **Speaker Disjointness (`test_speaker_disjointness`):**
   - Verified that the intersection of speaker sets across `TRAIN` (12 spks), `DEV_SEARCH` (4 spks), `DEV_CAL` (2 spks), `DEV_REPLICATION` (2 spks), and `TEST_SEALED` (4 spks) is strictly empty. Zero speaker overlap.

2. **Denominator Accounting (`test_manifest_structure_and_denominators`):**
   - Verified that for all 3,599 extracted utterances, the exact number of inter-phonemic gaps equals $N_{\text{target\_phones}} + 1$.
   - Confirmed all target phone labels are strictly in `{"CORRECT", "SUBSTITUTION", "OMISSION"}` and all gap labels are binary in `{0, 1}`.

3. **Metric Math & Perfect Score Audit (`test_evaluator_perfect_predictions`):**
   - Tested mathematical bounds of AP, Precision, Recall, F1 on synthetic ground truth.
   - Verified that perfect predictions achieve 1.0 on all 4 core classes, coverage = 1.0, and correctly trigger `OUTSTANDING_CONFIRMED`.

4. **Random Baseline & ECE Guardrail (`test_evaluator_random_baseline`):**
   - Verified that uncalibrated random predictions yield low utility and correctly remain in `EXPLORATORY` status without false promotion.
