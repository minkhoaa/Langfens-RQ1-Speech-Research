# LANGFENS — RQ1-DIAG-01 / Phoneme Omission & False-Negative Root Cause Diagnostic Audit

**Status:** `COMPLETED; DIAGNOSTIC_AUDIT_VERIFIED; READY_FOR_EXPERIMENT_DESIGN`

**Recorded at:** 2026-11-01T23:55:00+07:00

**Evidence source:**
- Standalone diagnostic harness: `audit_scripts/diagnose_omissions_and_false_negatives.py`
- Master diagnostic report: `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json` (SHA-256 `65076d510768e0bc17c77983aef8d4118d9e3c40dcd4604f6838a545c1cede9f`)
- Disaggregated CSV tables in `manifests/RQ1-DIAG-01/evaluation/`:
  - `omission_error_breakdown_by_speaker.csv`
  - `omission_error_breakdown_by_phone.csv`
  - `omission_error_breakdown_by_position.csv`
  - `omission_error_breakdown_by_duration.csv`
  - `representative_false_negatives.csv`
- Automated regression test suite:
  - `tests/test_rq1_diag_01_phoneme_omission_fn.py` (8 tests, 100% green)
  - Full repository regression suite: 347 passed, 1 skipped, 1 warning (100% green)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Cross-Attention Temporal Profiling, Full 39,576-Token TextGrid Ingestion Audit, Disaggregated Census, Matched/Mismatched Acoustic Sensitivity Decomposition)

**Branch:** `feat/rq1-diag-01-omission-fn-root-cause`

---

## 1. Executive Summary & Objective

Phase **RQ1-DIAG-01** resolves the core diagnostic questions surrounding high phoneme deletion rates and severe false-negative rates in the Baseline R MDD system prior to designing new training interventions.

Executed under strict CPU-first rules with zero GPU training, zero optimizer steps, and 100% sealed test partitions ($S_{\text{test}}^{(1..4)}$), this audit establishes:
1. **Definitive Conceptual Demarcation:** Distinct operational definitions separating CTC Levenshtein deletions (~4.9%), true gold phoneme omissions (1.59% prevalence, 14.93% of errors), and Binary MDD false negatives (83.38% at $\tau=0.50$; 97.46% on omissions).
2. **Exhaustive 4-Fold Token Census:** Clean partition of 39,576 tokens into 35,363 correct (89.35%), 3,584 substitutions (9.06%), and 629 deletions (1.59%).
3. **Severe Omission Detection Deficit:** Deletion AP collapses to **0.030506** (vs 0.264270 for substitutions). The acoustic sensitivity gap collapses to **+0.001677** (vs +0.010715 for substitutions). At standard $\tau=0.50$, **97.46% of true omissions are missed** (only 16/629 detected). At 15% FPR ceiling, **68.68% are missed**.
4. **Cross-Attention Temporal Collapse:** In-window attention mass is only **2.8%** (near random noise). Peak attention frames have a mean temporal error of **1.415 seconds** from the gold phone center, with individual heads collapsing onto arbitrary global anchor frames.
5. **Phonetic & Positional Concentration:** **55.64% of omissions are word-final** coda reductions; **86.65% occur in intervals under 80 ms**; and top 3 phonemes (**/D/**, **/R/**, **/T/**) account for **58.03%** of all omissions.
6. **False-Negative Forensics:** Discovered that extreme substitution false negatives are dominated by coda nasal neutralizations (/M/ $\to$ /N/ or /NG/), while omission false negatives are dominated by coda stop elisions and unstressed glide/fricative droppings.
7. **Ranked Root-Cause Hypotheses:** Ranked 5 root causes by measured evidence, distinguishing facts from speculation.
8. **Minimal Controlled Interventions & Gates:** Formulated 3 minimal controlled experiments (alignment regularization, positional embeddings, 3-class MDD) with quantitative acceptance gates without executing training.

---

## 2. Checkpoint Ledger

| Step | Checkpoint Title | Status | Primary Artifact | Test Verification |
|:---:|---|:---:|---|:---:|
| 01 | [Phoneme Omission & False-Negative Root Cause](checkpoints/01-phoneme-omission-and-false-negative-root-cause.md) | `COMPLETED` | `OMISSION_AND_FALSE_NEGATIVE_REPORT.json` (SHA-256 `65076d51...`) | `tests/test_rq1_diag_01_phoneme_omission_fn.py` (8 passed) |

---

## 3. Next Authorized Action
Await operator review of Phase RQ1-DIAG-01 diagnostic findings and authorization of next research gate before proceeding to implementation or training.
