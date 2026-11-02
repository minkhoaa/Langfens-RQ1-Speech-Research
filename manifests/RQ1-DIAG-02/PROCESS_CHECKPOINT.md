# LANGFENS — RQ1-DIAG-02 / Temporal Alignment Root-Cause Validation

**Status:** `COMPLETED; CAUSAL_DIAGNOSIS_VALIDATED; GO_FOR_JOINT_ALIGNMENT_TRAINING; NO_GO_FOR_INFERENCE_CLIPPING`

**Recorded at:** 2026-11-02T01:40:00+07:00

**Evidence source:**
- Standalone diagnostic harness: `audit_scripts/diagnose_temporal_alignment_validation.py`
- Master diagnostic report: `manifests/RQ1-DIAG-02/evaluation/TEMPORAL_ALIGNMENT_VALIDATION_REPORT.json` (SHA-256 `9895b8a572b3528fb5bf895b242d6f8739368f24893d930434efc8c163ae61c1`)
- Disaggregated CSV tables in `manifests/RQ1-DIAG-02/evaluation/`:
  - `attention_enrichment_by_token_type.csv`
  - `head_attention_breakdown.csv`
  - `omission_counterfactual_masking_results.csv`
  - `ctc_omission_boundary_audit.csv`
  - `speaker_paired_uncertainty_summary.csv`
- Automated regression test suite:
  - `tests/test_rq1_diag_02_temporal_alignment_validation.py` (10 tests, 100% green)
  - Full repository regression suite: 357 passed, 1 skipped, 1 warning (100% green)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Individual Head Weight Saturation Auditing, Counterfactual Acoustic Masking, CTC Omission Boundary Posteriors, 8-Speaker Paired Uncertainty Quantification)

**Branch:** `feat/rq1-diag-02-temporal-alignment-validation`

---

## 1. Executive Summary & Objective

Phase **RQ1-DIAG-02** evaluates whether poor temporal acoustic localization causally contributes to phoneme omission false negatives in the Baseline R MDD system.

Executed strictly under CPU-first, inference-only constraints with zero training and 100% test set isolation, this diagnostic validates:
1. **TextGrid Timestamps & Expected Omission Spans:** Characterizes all 629 deletions. Confirms 0 zero-duration intervals, median duration = 0.040s (2 frames), and proves that in 79.49% of omissions, the Praat interval is bit-for-bit identical to the transition span between neighboring realized phones.
2. **Attention Baselines & Near-Zero Localization:** Proves in-window attention mass is only ~3.0% - 3.4% (near uniform baselines) with an average temporal error of 1.392 seconds from phoneme centers.
3. **Forensic Proof of Head Saturation:** Auditing Heads 0, 1, 2, and 3 individually reveals that each head saturates as an extreme delta function (weight $> 0.991$) cycling across merely 6 to 7 unique static frames in the entire audio recording.
4. **Counterfactual Local Acoustic Masking:** Zeroing out local acoustic frames changes prediction scores by only 0.0011 (0.1%), proving the binary head is locally acoustically insensitive.
5. **Forced Attention Counterfactual:** Forcing attention into the local window without retraining causes massive upward score drift (+0.26 across all tokens) due to downstream MLP out-of-distribution breakdown.
6. **CTC Omission Evidence:** The CTC head emits `<BLANK>` with 86.28% argmax frequency and 84.47% mean posterior probability on omission frames, proving that acoustic omission information exists in Wav2Vec2 features but is bypassed by unconstrained cross-attention.
7. **Scientific Recommendation:** **GO for joint-training alignment regularization experimentation; NO-GO for inference-only heuristic window clipping.**

---

## 2. Checkpoint Ledger

| Step | Checkpoint Title | Status | Primary Artifact | Test Verification |
|:---:|---|:---:|---|:---:|
| 01 | [Temporal Alignment Root-Cause Validation](checkpoints/01-temporal-alignment-root-cause-validation.md) | `COMPLETED` | `TEMPORAL_ALIGNMENT_VALIDATION_REPORT.json` (SHA-256 `9895b8a5...`) | `tests/test_rq1_diag_02_temporal_alignment_validation.py` (10 passed) |

---

## 3. Next Authorized Action
Await operator review of Phase RQ1-DIAG-02 diagnostic validation and authorization of next research gate before proceeding to implementation or training.

---

## Appended correction — RQ1-DIAG-02B (2026-11-02)

The original text above is preserved as historical reporting. Its **CAUSAL_DIAGNOSIS_VALIDATED / GO_FOR_JOINT_ALIGNMENT_TRAINING** conclusions are **superseded**, not current authorization. The historical forced-attention path bypassed original MHA value/output projections; high CTC blank alone did not test omission discrimination. Stale numerical values and seven artifact hash claims are enumerated in the correction ledger.

See [DIAG-02B scientific report](../RQ1-DIAG-02B/SCIENTIFIC_REPORT.md) and [process checkpoint](../RQ1-DIAG-02B/PROCESS_CHECKPOINT.md). Current disposition: **PARTIAL (archived-score parity failures); NO-GO for causal training authorization**. New bounded CPU inference uses projection-correct attention, matched CTC controls and separate waveform/feature interventions. No training executed.
