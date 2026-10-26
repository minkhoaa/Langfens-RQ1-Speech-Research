# RQ1-TRN-02A Zero-Oracle (Protocol Z) Verification Dossier

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Base Branch:** `feat/rq1-trn-02a-eval-r2-corr`  
**Verified SHA:** `613dc728b06cd50604c9fb902923b223671545b3`  
**Dedicated Branch:** `feat/rq1-trn-02a-zero-oracle-verify`  
**Final Evaluation Gate:** **`RQ1-TRN-02A-ZERO-ORACLE-VERIFY: PASS`**  

---

## 1. Overview and Audit Scope

This dossier independently verifies the reported findings of **Protocol Z (Zero-Oracle Text+Audio Inference)** and concludes the scientific acceptance audit for `RQ1-TRN-02A-EVAL-R2-CORR`.

All work is bounded strictly to CPU-compatible evaluation, statistical verification, and regression test suites. Production checkpoints, split manifests, and historical Protocol O results remain immutable.

---

## 2. Directory Structure and Deliverables Inventory

| Artifact / Deliverable | File Path | Scope & Description |
|---|---|---|
| **Executive Summary & Gate Decision** | [`FINAL_ACCEPTANCE_DECISION.md`](FINAL_ACCEPTANCE_DECISION.md) | Official acceptance decision (`PASS`), checklist, summary findings, and reproducible execution commands. |
| **G2P Frontend Specification** | [`G2P_FRONTEND_SPECIFICATION.md`](G2P_FRONTEND_SPECIFICATION.md) | Cryptographic provenance (`resources/CMUDict/cmudict.dict`), tokenization contract, compound rules, and OOV handling. |
| **Input / Label Isolation Audit** | [`INPUT_LABEL_ISOLATION_AUDIT.md`](INPUT_LABEL_ISOLATION_AUDIT.md) | Architectural and cryptographic verification that TextGrids are excluded from input generation and model inference. |
| **Alignment Coverage & Ambiguity** | [`ALIGNMENT_COVERAGE_AND_AMBIGUITY_REPORT.md`](ALIGNMENT_COVERAGE_AND_AMBIGUITY_REPORT.md) | Investigation of 37 unaligned gold and 37 extra G2P tokens; reconciliation of historical 40-token draft; tie-breaking sweep. |
| **Protocol O vs Protocol Z Comparison** | [`RECOMPUTED_PROTOCOL_O_VS_Z_COMPARISON.md`](RECOMPUTED_PROTOCOL_O_VS_Z_COMPARISON.md) | Comprehensive metric comparison across Aligned-Only, Train-Prior (B1), Train-Prevalence (B2), and Exact Sequence populations. |
| **Acoustic Control Erratum** | [`ACOUSTIC_CONTROL_ERRATUM.md`](ACOUSTIC_CONTROL_ERRATUM.md) | Class-separated analysis (positives vs negatives), bootstrap reconciliation with permutation sensitivity, and retraction of uncomputed CIs. |
| **Protocol Z Prediction Export** | [`PROTOCOL_Z_PREDICTION_RECORDS.json`](PROTOCOL_Z_PREDICTION_RECORDS.json) | Per-utterance predictions (300 utterances), G2P phones, gold phones, labels, aligned pairs, and alignment traces. |
| **Protocol Z Verification Report** | [`PROTOCOL_Z_VERIFICATION_REPORT.json`](PROTOCOL_Z_VERIFICATION_REPORT.json) | Complete numerical report with all metrics, sweeps, per-speaker disaggregations, and runtime measurements. |

---

## 3. Core Verification Outcomes

1. **Reproduction of Reported AP Claim:**
   The historical exploratory claim that Protocol Z achieved validation AP = **0.312467** reproduces within **0.000004** ($0.312471$) under the audited validation prevalence imputation policy.
2. **Methodological Rectification of Imputation & Metric Designation:**
   Scoring unaligned gold tokens with validation prevalence leaked validation targets into the metric.
   - **Primary Replay Metric (Aligned-Only Population, $N = 9,919$, $99.63\%$ coverage):** $\text{AP} = \mathbf{0.313530}$ ($\Delta = -0.002904$ relative to Protocol O on identical subset $0.316434$). Measures zero-oracle predictions on hypothesized queries without arbitrary score imputation.
   - **Offline Benchmark Imputation (Policy B1 Full Gold, $N = 9,956$):** $\text{AP} = \mathbf{0.314265}$ ($\Delta = -0.002756$ relative to Protocol O $0.317021$). Oracle-referenced full-gold benchmark imputation, not a deployment-time inference policy.
   - **Exact Sequence Utterances ($N = 2,338$, 82 utts):** $\text{AP} = \mathbf{0.305362}$ ($\Delta = \mathbf{0.000000}$).
   - **Sensitivity & Audit Results (Policies B2, B3, Val Prevalence):** Preserved as sensitivity controls (B2 AP: $0.313006$, B3 AP: $0.307811$, Val Prevalence AP: $0.312471$).
   - **Final Errata Addendum:** See [`final_errata/FINAL_ERRATA_REPORT.md`](final_errata/FINAL_ERRATA_REPORT.md) for complete provenance corrections and tie-break sensitivity repairs.
3. **Class-Separated Acoustic Conditioning:**
   While overall decision agreement at $\tau = 0.50$ is $96.76\%$, agreement on the **positive class (true errors)** is **86.60%** ($155$ decision flips). Matched audio increases error detection recall by $+16.86\%$ (201 vs 172 TP) and suppresses false alarms by $-16.80\%$ (208 vs 250 FP).
4. **Statistical Generalization Limitations:**
   The paired bootstrap CI ($[+0.0048, +0.0396]$) applies strictly to `shift_1` within the two validation speakers (`NJS`, `RRBI`). Under `reverse` derangement, $\Delta \text{AP}$ drops to $+0.0030$ (and is negative on `NJS`). The hypothetical CI `[+0.000, +0.039]` has been retracted. Checkpoint selection bias on validation AP is formally acknowledged.

---

## 4. Source Implementations & Automated Test Suite

- **Executable Evaluation Entrypoint:** `audit_scripts/evaluate_protocol_z.py`
- **G2P Frontend Module:** `src/langfens_prep/g2p.py`
- **Zero-Oracle Dataset:** `src/langfens_prep/dataset.py` (`ZeroOracleL2ArcticDataset`)
- **Alignment Module:** `src/langfens_prep/alignment.py` (`align_canonical_sequences`)
- **Automated Regression Tests:**
  - `tests/test_protocol_z_g2p.py` (7 tests)
  - `tests/test_zero_oracle_isolation.py` (3 tests)
  - `tests/test_protocol_z_alignment_and_scoring.py` (3 tests)
  - `tests/test_acoustic_controls_class_separation.py` (2 tests)
  - `tests/test_prediction_reconstruction.py` (1 test)
  - `tests/test_label_independence.py` (2 tests)

All 18 tests pass with 100% success on CPU in under 17 seconds.
