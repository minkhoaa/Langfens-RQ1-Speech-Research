# Metric Reconciliation & Numerical Precision Audit — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Exhaustive reconciliation of historical Colab GPU validation vs. local CPU replay metrics  

---

## 1. Correction of the "0.3% Relative Tolerance" Generalization

In the initial EVAL-R2 summary, the claim was made that *"all historical metrics recorded during Google Colab GPU training were reproduced within sub-0.3% numerical tolerance."*

**Audit Correction:**  
This claim was an over-generalization. While the continuous ranking and global edit metrics fell within ~0.30% relative tolerance:
- **Binary Average Precision:** 0.316062 vs. 0.317021 (**+0.30%**)
- **Greedy Phoneme Error Rate (PER):** 0.236413 vs. 0.236818 (**+0.17%**)
- **Optimal F1 Score:** 0.3775 vs. 0.3785 (**+0.26%**)
- **Decoded Length Ratio:** 0.976419 vs. 0.976824 (**+0.04%**)

Several discrete thresholded metrics and confusion-matrix counts at $\tau=0.50$ differed by **more than 0.30%**:
- **True Positives at $\tau=0.50$:** 198 vs. 201 (**+1.52%**)
- **Recall at $\tau=0.50$:** 0.1711 vs. 0.1737 (**+1.52%**)
- **F1 Score at $\tau=0.50$:** 0.2532 vs. 0.2567 (**+1.38%**)
- **Precision at $\tau=0.50$:** 0.4865 vs. 0.4914 (**+1.01%**)
- **CTC Deletions:** 435 vs. 431 (**-0.92%**)
- **CTC Substitutions:** 1,699 vs. 1,707 (**+0.47%**)

These differences must be reported explicitly without masking count variations under a broad aggregate claim.

---

## 2. Taxonomical Decomposition of All Reported Metrics

All metric comparisons are partitioned into three causal classes:

### Class 1: Exact Bit-for-Bit / Mathematical Matches ($\Delta = 0.00\%$)
These quantities depend strictly on dataset indexing, token boundaries, and invariant discrete alignments:
- **Total Valid Phone Tokens:** 9,956 vs. 9,956 ($\Delta = 0$)
- **Total Ground-Truth Error Tokens:** 1,157 vs. 1,157 ($\Delta = 0$)
- **Validation Error Prevalence:** 0.116211 vs. 0.116211 ($\Delta = 0.000000$)
- **CTC Reference Tokens:** 9,881 vs. 9,881 ($\Delta = 0$)
- **CTC Insertions:** 202 vs. 202 ($\Delta = 0$)
- **Optimal Operating Threshold:** $\tau = 0.20$ vs. $\tau = 0.20$ ($\Delta = 0.00$)
- **NJS Phone / Error Count:** 5,011 tokens, 641 errors ($\Delta = 0$)
- **RRBI Phone / Error Count:** 4,945 tokens, 516 errors ($\Delta = 0$)

### Class 2: Continuous Arithmetic Precision Differences ($|\Delta_{\text{rel}}| \le 0.30\%$)
These variations arise from cross-hardware floating-point representation:
- **Historical Environment:** Google Colab Tesla T4 GPU running PyTorch AMP (`torch.float16`). In FP16, the mantissa provides 11 bits of precision (~3.3 decimal digits).
- **Replay Environment:** Intel x86_64 CPU running IEEE 754 single-precision `torch.float32` (24 bits of mantissa, ~7.2 decimal digits).
- **Batching Effect:** Historical evaluation used `batch_size: 2` (introducing zero-padding and attention masks in Wav2Vec2 self-attention layers). CPU replay evaluated with `batch_size: 1` (pure unpadded slice evaluation).

The resulting metric shifts are completely consistent with normal FP16/FP32 rounding:
- Binary AP: $+0.000959$ ($+0.30\%$)
- Greedy PER: $+0.000405$ ($+0.17\%$)
- CTC Blank Fraction: $-0.000120$ ($-0.02\%$)
- Decoded Length Ratio: $+0.000405$ ($+0.04\%$)

### Class 3: Discrete Threshold Boundary Flips ($0.30\% < |\Delta_{\text{rel}}| \le 1.52\%$)
Thresholding continuous probabilities at a fixed boundary (e.g., $\tau = 0.50$) transforms tiny continuous logit shifts into discrete count changes:
- In the validation set of 9,956 phone tokens, exactly **3 borderline tokens** shifted across $0.500000$:
  - When probabilities moved from $\approx 0.499$ to $\approx 0.501$, True Positives shifted from 198 to 201 (+3).
  - False Negatives correspondingly shifted from 959 to 956 (-3).
  - One non-error token shifted from $\approx 0.501$ to $\approx 0.499$, reducing False Positives from 209 to 208 (-1) and increasing True Negatives from 8,590 to 8,591 (+1).
- Because the total number of True Positives is relatively small (198), a change of +3 tokens produces a **+1.52% relative change**, despite representing only $3 / 9956 = 0.03\%$ of the total dataset.
- In CTC decoding, frame-level argmax decisions across 40 phonetic classes caused exactly 4 edit differences out of 9,881 tokens (2,340 vs 2,336 total edits).

---

## 3. Discrepancy Resolution: Per-Speaker Mismatched AP

In EVAL-R2, an internal discrepancy existed between the committed JSON artifact and the Markdown report for per-speaker mismatched audio AP:

| Speaker | JSON Artifact (`INDEPENDENT_VALIDATION_REPLAY.json`) | Markdown Report (`ACOUSTIC_CONTRIBUTION_CONTROLS.md`) | Recalculated from Predictions |
|---|---:|---:|---:|
| **NJS** | **0.378744** | 0.372583 | **0.378744** |
| **RRBI** | **0.234979** | 0.231267 | **0.234979** |

### Root Cause Audit:
1. Re-running exact inference and recalculating from the raw prediction arrays confirms that **the JSON artifact values (0.378744 and 0.234979) are mathematically correct** for the within-speaker cyclic derangement ($\text{shift\_1}$).
2. The values in the Markdown report (0.372583 and 0.231267) were transcribed from a preliminary subtraction draft where overall $\Delta \text{AP}$ was erroneously subtracted from the speaker matched AP ($0.394927 - 0.022344 = 0.372583$; $0.251519 - 0.020252 = 0.231267$).
3. The true per-speaker deltas under within-speaker derangement are:
   - **NJS:** Matched $0.394927$ vs. Mismatched $0.378744$ ($\Delta \text{AP} = \mathbf{+0.016183}$)
   - **RRBI:** Matched $0.251519$ vs. Mismatched $0.234979$ ($\Delta \text{AP} = \mathbf{+0.016540}$)
   - **Combined Validation Pool:** Matched $0.317021$ vs. Mismatched $0.294722$ ($\Delta \text{AP} = \mathbf{+0.022299}$)
4. The combined pool exhibits a slightly higher delta (+0.0223 vs +0.0163) due to cross-speaker distribution pooling across differing baseline prevalences (NJS prevalence 12.79% vs. RRBI prevalence 10.43%).

All reports have been updated to reflect the true recalculated values.
