# LANGFENS — RQ1-TRN-01A-R: Regression Test & Verification Suite Report

**Execution Gate:** RQ1-TRN-01A-R — Pretrained Backbone & Inference Validity Gate  
**Deliverable Associated Path:** `manifests/RQ1-TRN-01A/reconciliation/REGRESSION_TEST_REPORT.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/`  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & PyTorch Architect  
**Gate Evaluation Status:** `REGRESSION_TESTS: PASS`  

---

## 1. Executive Summary & Verification Mandate

In accordance with scientific reproducibility standards, we established a rigorous 12-item regression test suite covering all architectural, inference, and semantic corrections. Furthermore, the complete pre-existing test suite was executed across both the CUDA environment (`.venv-cuda`) and the reference CPU environment (`.venv`).

**Zero assertions were weakened or deleted.** All new tests enforce strict mathematical and algorithmic invariants.

---

## 2. 12 Mandatory Regression Items Verification Table

All 12 items implemented in `tests/test_reconciliation_regression.py`:

| # | Regression Test Requirement | Test Function | Test Outcome | Invariant Verified |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Pretrained checkpoint weights actually loaded | `test_pretrained_checkpoint_weights_actually_loaded` | **PASS** | `is_pretrained=True`, $94,371,712$ backbone params, conv0 weights non-trivial ($>0.01$) |
| **2** | Random vs pretrained initialization distinguishable | `test_random_and_pretrained_initialization_distinguishable` | **PASS** | Parameter tensors of pretrained vs random models verified strictly non-equal ($p < 10^{-12}$) |
| **3** | Full backbone output shapes correct | `test_full_backbone_output_shapes` | **PASS** | $32,000$ samples $\to$ 99 frames, 768 hidden dimension, 40 CTC classes |
| **4** | Deployable inference requires no gold intervals | `test_deployable_inference_requires_no_gold_intervals` | **PASS** | `predict()` succeeds with audio and canonical prompt only; returns probabilities in $[0, 1]$ |
| **5** | **Mandatory Negative Test:** Gold boundaries cannot influence predictions | `test_deployment_mode_invariance_to_gold_intervals` | **PASS** | Identical predictions with no intervals vs arbitrary vs scrambled intervals (exact bit equality) |
| **6** | Insertions correctly represented in CTC targets | `test_insertions_included_in_ctc_targets` | **PASS** | Epenthetic addition (`sil,AH,a`) correctly included chronologically in realized CTC sequence |
| **7** | Deletions excluded from acoustic CTC targets | `test_deletions_excluded_from_ctc_targets` | **PASS** | Deletions (`R,sil,d`) excluded from acoustic CTC targets; no silent contamination |
| **8** | Unknown realizations / 'err' masked by policy | `test_unknown_realizations_and_err_masked` | **PASS** | 'err' tags masked with $-100$ in diagnosis; excluded from CTC targets without fabrication |
| **9** | Canonical-phone token order preserved | `test_canonical_phone_order_preservation` | **PASS** | Ordered token sequence strictly maps 1-to-1 without transposition |
| **10**| Zero cross-speaker leakage across all folds | `test_no_cross_speaker_leakage` | **PASS** | Pairwise disjointness $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ |
| **11**| All-masked batches remain mathematically valid | `test_all_masked_losses_remain_valid` | **PASS** | Batches with all $-100$ targets return $0.0$ loss with intact autograd zero-gradients |
| **12**| Full-sized synthetic GPU smoke test | `test_full_sized_synthetic_gpu_smoke` | **PASS** | 2.0s audio forward, loss, backward, and optimizer step pass cleanly under AMP fp16 on GTX 1650 |

---

## 3. Pre-Existing Test Suite Execution Outcomes

### 3.1 CUDA Environment (`.venv-cuda` — PyTorch 2.14.1+cu130, GTX 1650)
```text
Command: PYTHONPATH=src .venv-cuda/bin/pytest
Outcome: 51 passed, 0 failed, 0 warnings in 52.84s (100% PASS)
```

Test breakdown by module:
- `tests/test_audio.py`: 3 passed (Audio loading, resampling, clipping checks).
- `tests/test_dataset_and_loader.py`: 5 passed (Dataset loading, blinding, collation, end-to-end forward/loss).
- `tests/test_full_census.py`: 2 passed (Corpus totals, duration bounds).
- `tests/test_model.py`: 6 passed (CTC forward, multitask loss, feature encoder freezing, checkpoint serialization, AdamW step, GPU fp16 forward/backward).
- `tests/test_normalizer.py`: 6 passed (Phoneme normalization, AX folding, typo slips, reversibility, IPA mapping).
- `tests/test_parsers.py`: 4 passed (Praat TextGrid parser, tiers, SpeechOcean parser).
- `tests/test_provenance.py`: 1 passed (Manifest cryptographic hash and provenance).
- `tests/test_training_contracts.py`: 12 passed (Masked BCE loss, Masked CE loss, CTC contract validation, pooling guarantees).
- `tests/test_reconciliation_regression.py`: 12 passed (All 12 reconciliation regression items).

### 3.2 Reference CPU Environment (`.venv` — PyTorch 2.14.1+cpu)
```text
Command: PYTHONPATH=src .venv/bin/pytest
Outcome: 49 passed, 2 skipped (CUDA tests), 0 failed in 39.12s (100% PASS)
```

---

## 4. Verification Verdict

All tests passed with zero regressions. The implementation satisfies every technical requirement set forth in the review gate.
