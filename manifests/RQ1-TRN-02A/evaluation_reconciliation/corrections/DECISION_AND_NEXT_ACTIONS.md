# Final Decision & Architecture Recommendations — RQ1-TRN-02A-EVAL-R2-CORR

**Acceptance Gate Decision:** **RQ1-TRN-02A-EVAL-R2-CORR: PASS**  
**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Dedicated Branch:** `feat/rq1-trn-02a-eval-r2-corr`  

---

## 1. Audit Acceptance Checklist

| Gate Requirement | Status | Verification Summary |
|---|---|---|
| **Numerical Inconsistencies Resolved** | **PASS** | Reconciled NJS mismatched AP (0.378744) and RRBI (0.234979). Corrected the 0.3% relative tolerance claim to explicitly differentiate exact matches, precision deltas, and +1.52% threshold boundary shifts. |
| **Fair Baseline Comparison** | **PASS** | Evaluated neural model and empirical prompt prior on identical coarse (0.05) and fine-grained (0.01) threshold grids. Labeled post-hoc F1 as exploratory; established AP as primary unbiased metric. |
| **Auditable Prediction Records Exported** | **PASS** | Exported `PAIRED_PREDICTION_RECORDS.json` (300 utterances, 9,956 phone tokens) with complete schema. Fully verified by automated regression test `tests/test_prediction_reconstruction.py`. |
| **Acoustic Controls & Statistical Rigor** | **PASS** | Tested four distinct within-speaker derangements (`shift_1`, `shift_2`, `shift_75`, `reverse`), documenting sensitivity ($\Delta \text{AP}$ from +0.0223 to +0.0030). Explicitly documented two-speaker ecological limits. |
| **Canonical Input Contract Specified** | **PASS** | Formally defined Protocol O (Oracle TextGrid Benchmark) and Protocol Z (Zero-Oracle CMUdict G2P Inference). Documented 72.67% G2P-to-Gold sequence divergence and 4.22% phone edit rate. |
| **Label Independence Proved** | **PASS** | Demonstrated mathematical logit invariance ($\Delta = 0.0$) in `tests/test_label_independence.py`. Refactored evaluator forward call to prediction-only arguments. |
| **Official Test Sealing Preserved** | **PASS** | Official Fold 1 test partition (900 utterances, 6 test speakers) remained completely sealed throughout all audit stages. |

---

## 2. Architectural & Research Phase Recommendations

The findings of this audit provide clear, actionable direction for the next phase of the LANGFENS research campaign:

### Recommendation 1: Mitigate Prompt Dominance via Acoustic Gating / Dropout
- **Diagnostic Finding:** When true audio is swapped for mismatched audio of the same speaker, **96.76% of binary MDD decisions remain unchanged**. The cross-attention module has learned to rely primarily on canonical prompt phonotactics rather than acoustic frame realization.
- **Architectural Solution:** In subsequent Colab training iterations:
  1. Apply **Prompt Token Dropout** during training: randomly mask canonical phone embeddings (e.g. $p = 0.15 - 0.25$) to force the cross-attention module to extract discriminative cues from the acoustic encoder.
  2. Implement an **Acoustic Gating Unit**: gate the canonical phone embedding using acoustic frame confidence derived from CTC posteriors.

### Recommendation 2: Adopt Protocol Z as Primary Deployable Inference Path
- **Diagnostic Finding:** Exploratory evaluation of Protocol Z (CMUdict G2P input + Levenshtein evaluation scoring) achieved an AP of **0.312467**, within 0.46 percentage points of Protocol O (0.317021), while aligning 99.6% of gold phone positions.
- **Protocol Action:** Formalize Protocol Z as the official inference standard for real-world application, while maintaining Protocol O for backward comparability with published L2-ARCTIC literature.

### Recommendation 3: Execute Full 4-Fold Cross-Validation on Google Colab GPU
- **Diagnostic Finding:** Evaluating on two validation speakers (NJS and RRBI) leaves between-speaker variance unconstrained ($df = 1$).
- **Protocol Action:** With the CPU evaluation pipeline verified, Google Colab GPU training should be authorized to train Folds 2, 3, and 4 under identical configurations. Pooling across all 4 folds (8 validation speakers) will provide the statistical degrees of freedom necessary to establish robust cross-speaker population claims.

---

## 3. Final Operational Status

In accordance with project governance (`AGENTS.md`):
- Zero local CPU training was executed.
- Production checkpoints remain unaltered with identical SHA-256 hashes.
- Branch `feat/rq1-trn-02a-eval-r2-corr` is complete, verified, and ready for peer review.
