# Acoustic Control & Statistical Interpretation Erratum — RQ1-TRN-02A-ZERO-ORACLE-VERIFY

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Class-separated acoustic conditioning diagnostics, permutation reconciliation, retraction of uncomputed intervals, and selection bias correction  

---

## 1. Executive Summary

This erratum strengthens the acoustic conclusions of RQ1-TRN-02A by rectifying prior over-generalizations and introducing class-separated diagnostics for matched vs. mismatched audio conditioning.

### Essential Rectifications:
1. **Class-Separated Decision Dynamics:**
   While overall decision agreement at $\tau = 0.50$ under audio derangement is **96.76%**, this aggregate figure is heavily masked by true correct tokens ($88.38\%$ of the dataset), where agreement is **98.09%**. On the **positive error class ($N = 1,157$)**, decision agreement drops to **86.60%** (155 decision flips). Matched audio increases True Positives by **+16.86%** (201 vs 172) and reduces False Positives by **-16.80%** (208 vs 250).
2. **Reconciliation of Bootstrap CI with Permutation Sensitivity:**
   The paired bootstrap confidence interval ($[+0.0048, +0.0396]$) applies strictly to the adjacent cyclic derangement (`shift_1`). It cannot be generalized to all permutations, as `reverse` produces $\Delta \text{AP} = +0.0030$ (outside the 95% CI).
3. **Retraction of Hypothetical Numeric Confidence Intervals:**
   The speculative interval `[+0.000, +0.039]` in `BOOTSTRAP_DESIGN_LIMITATIONS.md` was never empirically computed and is formally retracted.
4. **Validation Checkpoint Selection Bias Correction:**
   Validation AP is not an unbiased estimate of model generalization. Checkpoint `best.pt` was selected from 25 training epochs by tracking peak validation AP on this exact partition.

---

## 2. Class-Separated Score and Decision Analysis

In prior reports, model stability under audio swapping was summarized by aggregate prediction agreement ($96.76\%$ at $\tau=0.50$). Disaggregating by ground-truth class reveals the genuine acoustic sensitivity of the binary MDD head:

| Metric Dimension | Combined Dataset ($N=9,956$) | Positive Class (True Errors, $N=1,157$) | Negative Class (True Corrects, $N=8,799$) |
|---|---:|---:|---:|
| **Class Proportion** | 100.00% | 11.62% | 88.38% |
| **Mean Matched Audio Score** | 0.141905 | **0.263440** | **0.125911** |
| **Mean Mismatched Audio Score (`shift_1`)**| 0.143958 | **0.261542** | **0.128483** |
| **Mean Score Shift ($\hat{s}_{\text{m}} - \hat{s}_{\text{mm}}$)** | -0.002053 | **+0.001898** | **-0.002572** |
| **Mean Absolute Score Difference** | 0.029648 | **0.039981** | **0.028290** |
| **Decision Agreement at $\tau = 0.50$** | **96.76% (9,633 / 9,956)** | **86.60% (1,002 / 1,157)** | **98.09% (8,631 / 8,799)** |
| **Decision Flips at $\tau = 0.50$** | 323 tokens (3.24%) | **155 tokens (13.40%)** | **168 tokens (1.91%)** |
| **- Flips to Error ($\hat{s}_{\text{m}} < 0.5 \land \hat{s}_{\text{mm}} \ge 0.5$)** | 168 tokens | 63 tokens (spurious detections) | 105 tokens (new false alarms) |
| **- Flips to Correct ($\hat{s}_{\text{m}} \ge 0.5 \land \hat{s}_{\text{mm}} < 0.5$)** | 155 tokens | **92 tokens (lost true errors)** | 63 tokens (resolved false alarms)|
| **Decision Agreement at $\tau = 0.20$** | **92.30% (9,189 / 9,956)** | **91.18% (1,055 / 1,157)** | **92.44% (8,134 / 8,799)** |

### Scientific Significance:
1. **Masking by Majority Class:** High overall agreement (96.76%) is driven by the fact that $88.38\%$ of tokens are correct productions where the model defaults to low probabilities regardless of audio.
2. **Active Acoustic Contribution:** On actual pronunciation errors ($N = 1,157$), over **13.4% of decisions flip** when audio is swapped. Matched audio enables the model to detect **29 additional true errors** (201 vs 172 TP) while preventing **42 false alarms** (208 vs 250 FP).
3. The acoustic waveform therefore acts as an authentic discriminator that sharpens phonemic error boundaries, even though phonotactic text priors dominate the baseline prediction level.

---

## 3. Preserved Permutation Sensitivity Across Four Derangements

All four within-speaker deterministic derangements are preserved as permanent benchmark evidence:

| Derangement Condition | Permutation Formula | Overall Validation AP | Speaker NJS AP | Speaker RRBI AP | $\Delta \text{AP}$ vs Matched |
|---|---|---:|---:|---:|---:|
| **Condition 1: Matched True Audio** | $\pi(i) = i$ | **0.317021** | **0.394927** | **0.251519** | — |
| **Condition 2A: `shift_1` (Adjacent Cyclic)** | $\pi(i) = (i + 1) \pmod{150}$ | 0.294722 | 0.378744 | 0.234979 | **+0.022299** |
| **Condition 2B: `shift_2` (Two-Step Cyclic)** | $\pi(i) = (i + 2) \pmod{150}$ | 0.302581 | 0.375729 | 0.246302 | **+0.014440** |
| **Condition 2C: `shift_75` (Half-Partition Cyclic)**| $\pi(i) = (i + 75) \pmod{150}$| 0.300112 | 0.392248 | 0.238779 | **+0.016909** |
| **Condition 2D: `reverse` (Within-Speaker Mirror)** | $\pi(i) = 149 - i$ | **0.314003** | **0.404422** | 0.248592 | **+0.003018** |

### Statistical Reconciliation with Bootstrap CI:
- In EVAL-R2, the 1,000-resample utterance-clustered bootstrap under `shift_1` yielded a mean $\Delta \text{AP}$ of $+0.022514$ with 95% CI $[+0.004782, +0.039575]$.
- Under `reverse` derangement, the observed overall $\Delta \text{AP}$ is **$+0.003018$**, which falls **below the lower bound** of the `shift_1` bootstrap CI ($+0.0048$).
- Furthermore, for Speaker NJS under `reverse`, mismatched audio actually produces an AP of **0.404422** ($\Delta \text{AP} = -0.009495$), exceeding true matched audio.
- **Erroneous Conclusion Corrected:** The `shift_1` bootstrap CI must **never** be cited as establishing that all acoustic mismatches degrade performance by at least $+0.0048$. It estimates sampling variance within NJS and RRBI under that specific adjacent cyclic pairing only.

---

## 4. Retraction of Hypothetical Numeric Bounds

In `BOOTSTRAP_DESIGN_LIMITATIONS.md` (Section 2, Limitation 2), the text stated:
> *"A comprehensive bootstrap over permutations would reveal a confidence interval that spans down toward zero (e.g. [+0.000, +0.039]), significantly weakening claims of a robust acoustic benefit."*

**Erratum:**
- The numerical interval `[+0.000, +0.039]` was illustrative and conjectural; no joint permutation-bootstrap algorithm was executed to derive those specific numbers.
- **Action:** That numerical string is formally retracted. The scientifically accurate statement is:
  *"Because the observed point estimate of $\Delta \text{AP}$ under `reverse` is $+0.0030$ (and negative for Speaker NJS), the acoustic benefit is sensitive to the permutation structure and cannot be characterized by the `shift_1` interval alone."*

---

## 5. Checkpoint Selection Bias Correction

In prior documentation, validation AP was occasionally described as providing an "unbiased" measurement of model capabilities.

**Correction:**
1. Checkpoint `checkpoints/RQ1-TRN-02A/recovered/best.pt` was chosen out of 25 training epochs specifically because it achieved the **highest validation AP** ($0.316062$ at epoch 5, 3,600 successful updates, training Git SHA `caa56783113d40eac438e5f458cdb313b8932218`).
2. Consequently, validation AP contains **checkpoint selection bias (optimism bias)**. The model weights are tuned to random fluctuations in the validation set.
3. Neither Protocol O nor Protocol Z validation AP constitutes an unbiased estimate of generalization to unseen speakers. Unbiased generalization can only be measured on the sealed official test set (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) or external corpora. Note that speakers `SKA`, `YDCK`, and `ZHAA` belong to the Fold 1 training partition and are never official test speakers.
