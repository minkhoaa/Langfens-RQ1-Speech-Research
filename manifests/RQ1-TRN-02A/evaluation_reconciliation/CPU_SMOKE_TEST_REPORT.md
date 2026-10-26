# CPU Smoke Test Report — RQ1-TRN-02A-EVAL-R2

**Date:** 2026-10-26 11:45:23  
**Artifact:** `manifests/RQ1-TRN-02A/evaluation_reconciliation/CPU_SMOKE_TEST_REPORT.json`  
**Execution Environment:** CPU (x86_64, Linux, PyTorch 2.14.1+cpu)  
**Evaluated Sample:** First 4 validation utterances of Fold 1 (`NJS`, 121 phone tokens)  
**Status:** **PASS**  

---

## 1. Objectives and Verification Gates

Before executing full evaluation across the 300 validation utterances, a bounded smoke test on 4 utterances was executed to verify:
1. Production model loading and strict weight alignment.
2. Output tensor dimensionality and non-degenerate representations.
3. Strict finiteness of all output logits (no NaNs, no Infs).
4. Memory stability and measured runtime per utterance on CPU.
5. Absolute parameter immutability during inference.

---

## 2. Quantitative Verification Results

| Dimension | Verification Item | Measured Value | Threshold / Target | Status |
|---|---|---|---|---|
| **Checkpoint Identity** | Stored Epoch / Updates | Epoch 5 (Display 6) / 3,600 | Epoch 6 / 3,600 | PASS |
| | Checkpoint SHA-256 | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | Exact match | PASS |
| **Model State** | Missing / Unexpected Keys | 0 missing, 0 unexpected | 0 / 0 | PASS |
| | Pre-eval Fingerprint | `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d` | — | PASS |
| | Post-eval Fingerprint | `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d` | Exact match | PASS |
| **Logits Integrity** | Binary Logits Shape | `(1, num_canonical_phones)` | Finite, non-degenerate | PASS |
| | CTC Logits Shape | `(1, T_frames, 40)` | Finite, non-degenerate | PASS |
| | Finiteness Check | 100% finite (no NaN / Inf) | 100% | PASS |
| **Resource Profile** | Total Wall Time (4 utts) | 1.09 seconds | < 10.0 seconds | PASS |
| | Average Time per Utt | 0.2714 seconds | < 1.00 seconds | PASS |
| | Peak Process RAM | 1,314.9 MB (~1.3 GB) | < 4.0 GB | PASS |

---

## 3. Sample Prediction Inspection

Across the 4 smoke-test utterances:
- Total canonical phone tokens: 121
- Total error tokens: 11 (prevalence: 9.09%)
- Binary MDD AP on smoke sample: 0.130163
- Greedy CTC PER on smoke sample: 0.2185
- All model gradients remained unset (`None`); zero optimizer or training calls occurred.

## 4. Conclusion

CPU execution introduces zero memory exhaustion risk (1.3 GB peak RAM vs standard 16+ GB system RAM) and executes at ~0.27 seconds per utterance. Full 300-utterance evaluation will require ~80–95 seconds, demonstrating that independent CPU validation is completely feasible.
