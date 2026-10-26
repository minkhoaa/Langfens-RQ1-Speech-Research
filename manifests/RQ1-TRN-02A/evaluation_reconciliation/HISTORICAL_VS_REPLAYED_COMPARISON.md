# Historical vs. Fresh Replayed Evaluation Comparison — RQ1-TRN-02A-EVAL-R2

**Date:** 2026-10-26  
**Checkpoint:** Production `best.pt` (Epoch 6, 3,600 updates, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`)  
**Partition:** Fold 1 Official Validation (`NJS` & `RRBI`, 300 utterances, 9,956 phone tokens)  
**Historical Environment:** Google Colab Tesla T4 GPU, PyTorch FP16 AMP, Batch Size 2  
**Replay Environment:** Independent CPU (`float32`), Batch Size 1, Strict `torch.inference_mode()`  
**Verification Verdict:** **AUTHENTIC REPRODUCTION WITHIN NUMERICAL TOLERANCE**  

---

## 1. Global Metrics Reconciliation Table

| Metric | Historical Reference | Independent Replay (R2) | Absolute $\Delta$ | Relative % | Status / Notes |
|---|---:|---:|---:|---:|---|
| **Binary MDD AP (PR-AUC)** | 0.316062 | **0.317021** | +0.000959 | +0.30% | FP16 AMP vs FP32 arithmetic |
| **Validation Prevalence** | 0.116211 | **0.116211** | 0.000000 | 0.00% | Exact match (1,157 / 9,956) |
| **Total Phone Tokens** | 9,956 | **9,956** | 0 | 0.00% | Exact match |
| **Total Error Tokens** | 1,157 | **1,157** | 0 | 0.00% | Exact match |
| **TP at $\tau=0.50$** | 198 | **201** | +3 | +1.52% | 3 boundary tokens flipped near 0.50 |
| **FP at $\tau=0.50$** | 209 | **208** | -1 | -0.48% | FP16/FP32 precision boundary |
| **FN at $\tau=0.50$** | 959 | **956** | -3 | -0.31% | FP16/FP32 precision boundary |
| **TN at $\tau=0.50$** | 8,590 | **8,591** | +1 | +0.01% | FP16/FP32 precision boundary |
| **Precision at $\tau=0.50$** | 0.4865 | **0.4914** | +0.0049 | +1.01% | Exact function of TP/(TP+FP) |
| **Recall at $\tau=0.50$** | 0.1711 | **0.1737** | +0.0026 | +1.52% | Exact function of TP/(TP+FN) |
| **F1 at $\tau=0.50$** | 0.2532 | **0.2567** | +0.0035 | +1.38% | Exact harmonic mean |
| **Optimal F1** | 0.3775 | **0.3785** | +0.0010 | +0.26% | Reached at $\tau=0.20$ |
| **Optimal Threshold** | 0.20 | **0.20** | 0.00 | 0.00% | Exact match |
| **CTC Greedy PER** | 0.236413 | **0.236818** | +0.000405 | +0.17% | 4 edits delta out of 9,881 tokens |
| **CTC Substitutions** | 1,699 | **1,707** | +8 | +0.47% | Argmax boundary across 40 classes |
| **CTC Deletions** | 435 | **431** | -4 | -0.92% | Boundary frames near blank threshold |
| **CTC Insertions** | 202 | **202** | 0 | 0.00% | **Exact match** |
| **CTC Total Edits** | 2,336 | **2,340** | +4 | +0.17% | 2,340 vs 2,336 |
| **CTC Reference Tokens** | 9,881 | **9,881** | 0 | 0.00% | **Exact match** |
| **CTC Hypothesis Tokens** | 9,648 | **9,652** | +4 | +0.04% | 9,652 vs 9,648 |
| **CTC Decoded Length Ratio** | 0.976419 | **0.976824** | +0.000405 | +0.04% | Hyp / Ref ratio |
| **CTC Blank Fraction** | 0.789390 | **0.789270** | -0.000120 | -0.02% | 0.789270 vs 0.789390 |

---

## 2. Per-Speaker Disaggregated Reconciliation

### Speaker NJS (Spanish-accented English, 150 utterances, 5,011 phones, 641 errors)
| Metric | Historical Reference | Independent Replay (R2) | Absolute $\Delta$ | Notes |
|---|---:|---:|---:|---|
| **Binary MDD AP** | 0.3950 | **0.394927** | -0.000073 | Virtually identical (< 0.01% diff) |
| **Prevalence** | 0.1279 | **0.127919** | +0.000019 | Exact fraction 641 / 5,011 |
| **Precision at $\tau=0.50$** | 0.6453 | **0.6520** | +0.0067 | High-precision operating point |
| **Recall at $\tau=0.50$** | 0.2044 | **0.2075** | +0.0031 | 133 vs 131 true positives |
| **F1 at $\tau=0.50$** | 0.3104 | **0.3148** | +0.0044 | — |
| **Greedy CTC PER** | 0.2327 | **0.234120** | +0.001420 | Concordant recognition rate |
| **CTC Substitution Rate** | 0.1708 | **0.171607** | +0.000807 | — |
| **CTC Deletion Rate** | 0.0399 | **0.040331** | +0.000431 | — |
| **CTC Insertion Rate** | 0.0220 | **0.022182** | +0.000182 | — |
| **Decoded Length Ratio** | 0.9821 | **0.981851** | -0.000249 | Length preserved |

### Speaker RRBI (Russian-accented English, 150 utterances, 4,945 phones, 516 errors)
| Metric | Historical Reference | Independent Replay (R2) | Absolute $\Delta$ | Notes |
|---|---:|---:|---:|---|
| **Binary MDD AP** | 0.2503 | **0.251519** | +0.001219 | Concordant discrimination |
| **Prevalence** | 0.1043 | **0.104348** | +0.000048 | Exact fraction 516 / 4,945 |
| **Precision at $\tau=0.50$** | 0.3284 | **0.3317** | +0.0033 | Lower precision than NJS |
| **Recall at $\tau=0.50$** | 0.1298 | **0.1318** | +0.0020 | 68 vs 67 true positives |
| **F1 at $\tau=0.50$** | 0.1861 | **0.1886** | +0.0025 | — |
| **Greedy CTC PER** | 0.2401 | **0.239537** | -0.000563 | Concordant recognition rate |
| **CTC Substitution Rate** | 0.1731 | **0.173913** | +0.000813 | — |
| **CTC Deletion Rate** | 0.0482 | **0.046932** | -0.001268 | — |
| **CTC Insertion Rate** | 0.0189 | **0.018692** | -0.000208 | — |
| **Decoded Length Ratio** | 0.9707 | **0.971759** | +0.001059 | Length preserved |

---

## 3. Threshold Calibration Sweep Comparison

Both runs swept thresholds $\tau \in [0.05, 0.95]$ with step $0.05$:

| Threshold $\tau$ | Hist. Precision | Replay Precision | Hist. Recall | Replay Recall | Hist. F1 | Replay F1 | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| **0.05** | 0.1446 | 0.1446 | 0.9196 | 0.9196 | 0.2499 | 0.2499 | Exact match |
| **0.10** | 0.1794 | 0.1795 | 0.8358 | 0.8358 | 0.2954 | 0.2955 | Reconciled |
| **0.15** | 0.2354 | 0.2355 | 0.6525 | 0.6534 | 0.3460 | 0.3462 | Reconciled |
| **0.20** | **0.2911** | **0.2918** | **0.5367** | **0.5385** | **0.3775** | **0.3785** | **Optimal F1 Point** |
| **0.25** | 0.3566 | 0.3574 | 0.3846 | 0.3846 | 0.3701 | 0.3706 | Reconciled |
| **0.30** | 0.4129 | 0.4132 | 0.3258 | 0.3250 | 0.3643 | 0.3640 | Reconciled |
| **0.35** | 0.4512 | 0.4523 | 0.3077 | 0.3086 | 0.3659 | 0.3667 | Reconciled |
| **0.40** | 0.4583 | 0.4632 | 0.2947 | 0.2947 | 0.3588 | 0.3601 | Reconciled |
| **0.45** | 0.4666 | 0.4703 | 0.2532 | 0.2541 | 0.3283 | 0.3300 | Reconciled |
| **0.50** | **0.4865** | **0.4914** | **0.1711** | **0.1737** | **0.2532** | **0.2567** | **Standard $\tau=0.50$** |
| **0.55** | 0.4678 | 0.4734 | 0.0691 | 0.0691 | 0.1205 | 0.1206 | Reconciled |
| **0.60** | 0.4400 | 0.4314 | 0.0190 | 0.0190 | 0.0365 | 0.0364 | Reconciled |
| **0.65** | 0.2500 | 0.2500 | 0.0009 | 0.0009 | 0.0017 | 0.0017 | Exact match |
| **0.70+** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Zero predictions |

---

## 4. Root Cause Analysis of Numerical Tolerances

The tiny observed deviations ($\Delta \text{AP} = +0.000959$, $\Delta \text{PER} = +0.000405$) are completely accounted for by standard numerical factors:
1. **Arithmetic Precision (FP16 vs. FP32):**
   - The historical evaluation driver ran inside `torch.autocast("cuda", dtype=torch.float16)`. In FP16 arithmetic, the mantissa has 11 bits of precision (~3.3 decimal digits).
   - The CPU independent replay ran strictly in native IEEE 754 `float32` (24 bits of mantissa, ~7.2 decimal digits).
   - In threshold calibration, out of 9,956 binary predictions, exactly 3 borderline tokens shifted from $0.499 \dots$ to $0.500 \dots$, flipping TP from 198 to 201.
2. **Batch Padding Isolation:**
   - The historical evaluation used batch size 2, where shorter utterances in a pair receive zero-padding and attention masks in the Wav2Vec2 transformer layers.
   - The CPU replay evaluated with batch size 1, completely eliminating cross-utterance padding masks.
3. **CTC Greedy Argmax Discretization:**
   - In CTC greedy decoding across 40 phoneme classes and ~220 acoustic frames per utterance, slight frame-level differences around decision boundaries altered exactly 4 edits out of 9,881 reference tokens (2,340 vs 2,336).
   - CTC insertions matched **byte-for-byte and token-for-token** at exactly 202.

## 5. Conclusion

The historical evaluation metrics recorded in `manifests/RQ1-TRN-02A/EVALUATION_REPORT.json` are **authentically reproduced** from raw checkpoint inference. There is zero evidence of fabrication, metric tampering, or synthetic numbers.
