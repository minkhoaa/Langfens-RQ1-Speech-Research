# Scientific Claim Errata — RQ1-TRN-02A

**Status:** **AUDITED & FINALIZED (R2 REPLAY COMPLETED)**  
**Auditor:** Senior PyTorch Engineer & Scientific Reproducibility Auditor  
**Date:** 2026-10-26  

---

## 1. Historical Context (R1 Recovery Note)

During R1 checkpoint recovery, the historical comparison recorded in `manifests/RQ1-TRN-02A/ACOUSTIC_BASELINE_COMPARISON.md` and `BASELINE_COMPARISON.json` was flagged as overstating what the single-fold descriptive evidence established:
- Stored neural validation AP: 0.316062 vs Canonical empirical prior AP: 0.303063 ($\Delta = +0.012999$).
- Model optimal F1 (0.3775 at $\tau=0.20$) was below the empirical prior's best F1 (0.3879 at $\tau=0.22$).

---

## 2. Final R2 Evidence & Quantitative Reconciliation

The complete independent CPU validation replay and paired acoustic controls generated the following fresh metrics on all 300 Fold 1 validation utterances (9,956 phone tokens):

| Condition / Model | Input Modalities | Validation AP | Optimal F1 | F1 at $\tau=0.50$ | Precision at $\tau=0.50$ | Recall at $\tau=0.50$ |
|---|---|---:|---:|---:|---:|---:|
| **Acoustic Neural Model (Matched)** | Real Audio + Canonical Prompt | **0.317021** | **0.3785** ($\tau=0.20$) | 0.2567 | 0.4914 | 0.1737 |
| **Acoustic Neural Model (Mismatched Audio)** | Permuted Audio + Canonical Prompt | 0.294722 | 0.3667 ($\tau=0.25$) | 0.2179 | 0.4435 | 0.1443 |
| **Canonical Empirical Prior $P_{\text{train}}$** | Canonical Prompt Only (Bayesian) | 0.303063 | 0.3756 ($\tau=0.25$) | **0.3348** | 0.4079 | **0.2835** |
| **Prevalence Baseline** | None (Constant $p=0.116211$) | 0.116211 | — | 0.0000 | 0.0000 | 0.0000 |

### Paired Within-Speaker Acoustic Controls:
- **$\Delta \text{AP}$ (Matched $-$ Mismatched):** $+0.022299$ (+2.23 percentage points).
- **1,000 Utterance-Clustered Bootstrap 95% CI:** $[+0.004782, +0.039575]$.
- **Prediction Agreement between Matched and Mismatched Audio:**
  - At $\tau = 0.50$: **96.76%** of classification decisions are completely identical.
  - At $\tau = 0.20$: **92.30%** of classification decisions are completely identical.
- **Mean Absolute Score Difference:** $0.029648$ (~3.0 percentage points). Only 4.36% of tokens shift probability by $> 0.10$ when audio is swapped.

---

## 3. Specific Scientific Claim Corrections

### Claim 1: "Significantly outperforms both empirical prompt prior and prevalence baselines, demonstrating authentic acoustic dependence"
- **Original Claim:** Stated in `ACOUSTIC_BASELINE_COMPARISON.md` and `BASELINE_COMPARISON.json`.
- **Finding:** While $\Delta \text{AP}$ between matched and mismatched audio is statistically positive (+0.0223, 95% CI strictly $> 0$), the acoustic contribution is secondary to prompt phonotactic priors. The model retains 92.97% of its AP (0.2947 vs 0.3170) and **96.76% prediction agreement** when fed completely mismatched audio.
- **Corrected Claim:** *"The acoustic neural model achieves an Average Precision of 0.3170 on Fold 1 validation, compared to 0.3031 for a train-only canonical empirical prior and 0.2947 under within-speaker audio permutation controls. While paired utterance-clustered bootstrap analysis confirms a statistically non-zero acoustic modulation ($\Delta \text{AP} = +0.0223$, 95% CI $[+0.0048, +0.0396]$), the model's binary MDD decisions remain 96.76% identical under mismatched audio, demonstrating that predictions are heavily dominated by canonical prompt phonotactics rather than acoustic realization."*

### Claim 2: "Acoustic conditioning enables high-precision operating points: at $\tau=0.50$ correctly confirming whether the speaker actually mispronounced the specific token"
- **Original Claim:** Stated in Section 2 of `ACOUSTIC_BASELINE_COMPARISON.md`.
- **Finding:** At $\tau=0.50$, precision is 49.14% overall (and 65.20% on speaker NJS), but recall is only 17.37% (failing to detect >82% of actual mispronunciations). Furthermore, the non-neural canonical empirical prior achieves an F1 of 0.3348 at $\tau=0.50$, substantially higher than the neural model's 0.2567.
- **Corrected Claim:** *"At threshold $\tau = 0.50$, the neural model achieves 49.14% precision (65.20% on speaker NJS), but operates at very low recall (17.37%), yielding F1 = 0.2567—lower than the canonical empirical prior's F1 of 0.3348 at the same threshold. Higher precision on NJS reflects speaker-specific error profiles on two validation speakers and cannot be generalized across accents."*

### Claim 3: "System demonstrates acoustic diagnosis readiness for educational or clinical feedback"
- **Original Claim:** Implied in executive summaries.
- **Finding:** The system is an experimental two-head architecture evaluated on Fold 1 validation (300 utterances across 2 speakers). Canonical phone sequences are extracted from human annotator TextGrids rather than an independent text G2P frontend (with 57.5% sequence divergence across speakers on identical sentences).
- **Corrected Claim:** *"The system is an experimental research model on Fold 1 validation. The official Fold 1 test partition remains sealed. The current pipeline requires oracle TextGrid-derived canonical phone sequences and exhibits high prompt-conditioning dominance. It is not deployable or certified for automated clinical or pedagogical pronunciation diagnosis."*
