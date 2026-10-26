# Acoustic Contribution Controls & Baseline Comparison — RQ1-TRN-02A-EVAL-R2

**Date:** 2026-10-26  
**Checkpoint:** Production `best.pt` (Epoch 6, 3,600 updates)  
**Partition:** Fold 1 Official Validation (`NJS` & `RRBI`, 300 utterances, 9,956 phone tokens)  
**Execution:** Independent CPU Replay with Paired Within-Speaker Derangements  

---

## 1. Experimental Design of Acoustic Controls

To determine whether the multi-task neural MDD model genuinely extracts acoustic evidence from speech waveforms—rather than simply memorizing phonotactic error priors from canonical prompt embeddings—four experimental conditions were executed on the exact same 300 validation utterances and 9,956 canonical phone positions:

1. **Condition 1 (Matched Audio):**
   - True paired acoustic speech waveform $A_i$ + true canonical prompt $C_i$.
   - Evaluated against true ground-truth binary pronunciation labels $Y_i$.
2. **Condition 2 (Mismatched Audio / Audio Permutation):**
   - Deterministic within-speaker derangement $\pi(i)$ such that $\pi(i) \neq i$ for all $i$:
     $$\forall i \in [0, 149] \text{ (NJS): } \pi(i) = (i + 1) \pmod{150}$$
     $$\forall i \in [150, 299] \text{ (RRBI): } \pi(i) = 150 + ((i - 150 + 1) \pmod{150})$$
   - Canonical prompt $C_i$ and labels $Y_i$ remain strictly intact.
   - Input waveform is replaced by $A_{\pi(i)}$, with exact waveform lengths and padding masks recalculated.
   - Evaluates whether predictions collapse when acoustic content is mismatched to the prompt.
3. **Condition 3 (Canonical-Only Empirical Prior Baseline):**
   - Prompt-only Bayesian lookup:
     $$P_{\text{train}}(\text{error} \mid c) = \frac{\sum_{j \in \text{train}, c_j = c} y_j}{\sum_{j \in \text{train}, c_j = c} 1}$$
   - Computed strictly from the **2,399 Fold 1 train utterances** (79,426 phone tokens, 11,841 errors, base rate 14.91%).
   - Zero acoustic features, zero neural weights, zero timestamps.
4. **Condition 4 (Constant Prevalence Baseline):**
   - Constant trivial prediction equal to validation error prevalence ($p = 0.116211$).

---

## 2. Comparative Findings Across Conditions

| Condition | Modalities / Inputs | Binary AP (PR-AUC) | Optimal F1 | Optimal Threshold $\tau$ | F1 at $\tau=0.50$ |
|---|---|---:|---:|---:|---:|
| **1. Matched Audio** | Real Waveform + Canonical Prompt | **0.317021** | **0.3785** | 0.20 | 0.2567 |
| **2. Mismatched Audio** | Permuted Waveform + Canonical Prompt | **0.294722** | 0.3667 | 0.25 | 0.2179 |
| **3. Canonical Empirical Prior** | Canonical Prompt Only ($P_{\text{train}}$) | **0.303063** | 0.3756 | 0.25 | **0.3348** |
| **4. Constant Prevalence** | None (Base Rate $p=0.116211$) | **0.116211** | — | — | 0.0000 |

### Per-Speaker Controls Breakdown
| Speaker | Matched Audio AP | Mismatched Audio AP | $\Delta \text{AP}$ (Matched $-$ Mismatched) | Canonical Prior AP |
|---|---:|---:|---:|---:|
| **NJS** (Spanish L1, 150 utts) | **0.394927** | 0.372583 | +0.022344 | 0.365120 |
| **RRBI** (Russian L1, 150 utts) | **0.251519** | 0.231267 | +0.020252 | 0.243890 |
| **Overall Fold 1 Validation** | **0.317021** | 0.294722 | **+0.022299** | 0.303063 |

---

## 3. Paired Token-Level Analysis (9,956 Valid Positions)

Comparing predicted probabilities between matched audio ($\hat{S}_{\text{matched}}$) and mismatched audio ($\hat{S}_{\text{mismatched}}$) on the exact same token positions reveals profound insights into model behavior:

| Analysis Dimension | Metric | Value | Interpretation |
|---|---|---:|---|
| **Score Shift** | Mean Score Difference ($\hat{S}_{\text{m}} - \hat{S}_{\text{mm}}$) | -0.002053 | Mean predicted probability is essentially invariant |
| | Std of Score Difference | 0.044284 | Standard deviation of acoustic perturbation is small (~4.4%) |
| | Median Score Difference | -0.001932 | Symmetric around 0 |
| | Interquartile Range (IQR) | 0.038051 | 50% of tokens shift by less than 3.8% |
| | Mean Absolute Difference ($|\hat{S}_{\text{m}} - \hat{S}_{\text{mm}}|$) | **0.029648** | Average shift across all tokens is only **~3.0 percentage points** |
| **Decision Agreement** | Prediction Agreement at $\tau=0.50$ | **96.76%** | **9,633 of 9,956** classification decisions are identical! |
| | Prediction Agreement at $\tau=0.20$ | **92.30%** | **9,189 of 9,956** classification decisions are identical! |
| **Audio Sensitivity** | $|\hat{S}_{\text{m}} - \hat{S}_{\text{mm}}| > 0.01$ | 69.26% | ~69% of tokens have non-zero acoustic modulation (> 0.01) |
| | $|\hat{S}_{\text{m}} - \hat{S}_{\text{mm}}| > 0.05$ | 17.74% | Only ~18% of tokens shift by more than 0.05 |
| | $|\hat{S}_{\text{m}} - \hat{S}_{\text{mm}}| > 0.10$ | **4.36%** | Only **4.36%** of tokens shift by more than 0.10 |

---

## 4. Utterance-Clustered Bootstrap Analysis (1,000 Resamples)

Because tokens within the same utterance share acoustic recording conditions, speaker characteristics, and sentence context, standard token-level i.i.d. variance underestimates uncertainty. We conducted an **utterance-clustered bootstrap** (1,000 resamples of the 300 validation utterances):

| Distribution | Mean AP | 95% Bootstrap Confidence Interval |
|---|---:|---:|
| **Matched Audio AP** | 0.318120 | $[0.287232, 0.348134]$ |
| **Mismatched Audio AP** | 0.295606 | $[0.265070, 0.327052]$ |
| **$\Delta \text{AP}$ (Matched $-$ Mismatched)** | **+0.022514** | **$[+0.004782, +0.039575]$** |

### Statistical Finding:
- The 95% confidence interval for $\Delta \text{AP}$ ($[+0.0048, +0.0396]$) strictly excludes zero. This confirms that the model's cross-attention module extracts a **statistically detectable acoustic contribution** beyond the prompt text alone.
- However, the magnitude of this acoustic contribution (+2.23 AP points) is modest when compared to the **+18.68 AP points** provided by the prompt embeddings alone over prevalence (0.3031 vs 0.1162).

---

## 5. Scientific Interpretation & Critical Caveats

### 1. Dominance of Prompt Phonotactics
When supplied with completely mismatched audio (e.g., an entirely different sentence spoken by the same speaker), the neural model still achieves an AP of **0.294722**—retaining **92.97% of its matched AP**. Furthermore, **96.76% of classification decisions at $\tau=0.50$ remain identical**. This proves empirically that the model's MDD predictions are primarily anchored in the canonical prompt phoneme embedding rather than acoustic realization.

### 2. Canonical Empirical Prior vs. Neural Model
The simple non-neural empirical prior $P_{\text{train}}(\text{err} \mid \text{phone})$ achieves an AP of **0.303063** and an F1 of **0.3348 at $\tau=0.50$**, which actually surpasses the neural model's F1 at $\tau=0.50$ (0.2567). The neural model requires threshold re-tuning ($\tau=0.20$) to reach F1 = 0.3785.

### 3. The Two-Speaker Statistical Limitation
The official Fold 1 validation partition comprises only two speakers (`NJS` and `RRBI`). In speech ML evaluation:
- Two speakers provide only **1 degree of freedom** for cross-speaker variance.
- Any speaker-specific acoustic peculiarities (e.g. microphone distance, speaking rate, L1 language background) can confound acoustic detection.
- **Cross-speaker or cross-accent statistical generalization cannot be asserted from two validation speakers.**

### 4. CTC Head vs. Binary MDD Head
The CTC phoneme recognition head exhibits unambiguous acoustic dependence:
- It achieves a greedy PER of 23.68% and a decoded length ratio of 97.68% directly from raw audio without prompt inputs.
- The binary MDD head, however, exhibits heavy prompt-conditioning dominance. Future work must address prompt regularization or acoustic gating to increase the weight of acoustic evidence in the binary decision head.
