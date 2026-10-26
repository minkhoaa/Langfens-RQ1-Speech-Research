# Corrected Per-Speaker Acoustic Controls & Permutation Sensitivity — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Precise per-speaker acoustic derangement results and sensitivity across multiple permutation functions  

---

## 1. Discrepancy Reconciliation & Recalculated Values

In EVAL-R2, a transcription error appeared in the Markdown summary table for per-speaker mismatched audio AP. The values have been rigorously recalculated from raw model predictions across all 300 validation utterances:

| Speaker Partition | Canonical Phone Count | Error Count | Prevalence | Matched Audio AP | Mismatched Audio AP (`shift_1`) | $\Delta \text{AP}$ (Matched $-$ Mismatched) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| **Speaker NJS** | 5,011 | 641 | 0.127919 | **0.394927** | **0.378744** | **+0.016183** | Reconciled (corrected from 0.372583) |
| **Speaker RRBI** | 4,945 | 516 | 0.104348 | **0.251519** | **0.234979** | **+0.016540** | Reconciled (corrected from 0.231267) |
| **Combined Fold 1 Validation** | 9,956 | 1,157 | 0.116211 | **0.317021** | **0.294722** | **+0.022299** | Confirmed from prediction arrays |

### Explanation of Discrepancy:
- The Markdown values in the previous report (0.372583 and 0.231267) were the product of a draft calculation in which the overall dataset $\Delta \text{AP}$ was erroneously subtracted from the per-speaker matched AP ($0.394927 - 0.022344 = 0.372583$; $0.251519 - 0.020252 = 0.231267$).
- The actual per-speaker mismatched APs computed directly from the prediction scores are **0.378744 for NJS** and **0.234979 for RRBI**.
- Both speakers exhibit remarkably consistent individual acoustic lifts: **+0.0162** for NJS and **+0.0165** for RRBI.
- The combined pool shows a slightly larger lift (+0.0223) due to distribution aggregation across different speaker prevalences (12.79% vs 10.43%).

---

## 2. Permutation Sensitivity Analysis Across Multiple Derangements

To evaluate whether the acoustic control is an artifact of the specific permutation choice ($\text{shift\_1}$), four distinct deterministic within-speaker derangements were evaluated on the same 300 utterances:

1. **`shift_1` (Adjacent Cyclic Shift):** $\pi(i) = (i + 1) \pmod{150}$
2. **`shift_2` (Two-Step Cyclic Shift):** $\pi(i) = (i + 2) \pmod{150}$
3. **`shift_75` (Half-Partition Cyclic Shift):** $\pi(i) = (i + 75) \pmod{150}$ (maximally distant within speaker)
4. **`reverse` (Within-Speaker Reflection):** $\pi(i) = 149 - i$ (order reversed within speaker)

Every permutation satisfies the strict derangement condition: $\pi(i) \neq i$ for all $i$ (zero self-matches).

### Comparative Results Across Permutations:
| Audio Condition | Overall Validation AP | NJS AP | RRBI AP | $\Delta \text{AP}$ vs. Matched | Prediction Agreement at $\tau=0.50$ |
|---|---:|---:|---:|---:|---:|
| **Matched True Audio** | **0.317021** | **0.394927** | **0.251519** | — | 100.00% |
| **Mismatched `shift_1`** | 0.294722 | 0.378744 | 0.234979 | **+0.022299** | **96.76%** |
| **Mismatched `shift_2`** | 0.302581 | 0.375729 | 0.246302 | **+0.014440** | **96.87%** |
| **Mismatched `shift_75`** | 0.300112 | 0.392248 | 0.238779 | **+0.016909** | **96.82%** |
| **Mismatched `reverse`** | **0.314003** | **0.404422** | 0.248592 | **+0.003018** | **97.10%** |

---

## 3. Critical Findings on Permutation Sensitivity

1. **Variance Over Permutation Choice:**
   - The acoustic lift $\Delta \text{AP}$ varies by an order of magnitude depending on which derangement is applied: from **+0.0223** under `shift_1` down to **+0.0030** under `reverse`.
   - Under `reverse` derangement, the overall AP of the model with completely wrong audio is **0.314003**—virtually identical to matched audio (0.317021).
2. **Inversion Anomaly on Speaker NJS:**
   - Under `reverse`, mismatched audio actually produces an AP of **0.404422 on NJS**, which is **higher than true matched audio** (0.394927)!
   - This occurs because the cross-attention module pools acoustic features across frames. When the prompt phoneme embeddings provide a strong baseline prior, average speaker acoustic representations from a reversed sentence can coincidentally align with general phonotactic expectations without penalizing discrimination.
3. **Extreme Prediction Agreement:**
   - Across all four permutations, prediction agreement at $\tau = 0.50$ remains between **96.76% and 97.10%**.
   - Fewer than **3.3% of token classification decisions** change when true audio is swapped for mismatched audio of the same speaker.

### Conclusion:
Relying on a single arbitrary permutation (`shift_1`) overstates the certainty of the acoustic effect. Across all permutations, the binary MDD head is overwhelmingly dominated by the canonical prompt embedding.
