# Unified Threshold Comparison & Fair Baseline Analysis — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Fair, protocol-aligned comparison between Acoustic Neural Model and Canonical-Only Empirical Prior  

---

## 1. Root Cause of the Baseline Discrepancy

In earlier documentation, an apparent discrepancy existed regarding the Canonical Prompt Prior's best F1 score:
- **Historical Report (`BASELINE_COMPARISON.json`):** Prior Best F1 = **0.3879** at $\tau = 0.22$.
- **R2 Report (`ACOUSTIC_CONTRIBUTION_CONTROLS.md`):** Prior Best F1 = **0.3756** at $\tau = 0.25$.

### Audit Finding:
The difference was entirely an artifact of **mismatched threshold search resolutions**:
1. In `BASELINE_COMPARISON.json`, the empirical prior was swept on a fine-grained grid ($\Delta \tau = 0.01$). It identified its global optimum at $\tau = 0.22$ with $\text{F1} = 0.3879$. However, the neural model in that report was only reported on a coarse grid ($\Delta \tau = 0.05$), finding $\text{F1} = 0.3775$ at $\tau = 0.20$.
2. In EVAL-R2, the evaluation script evaluated both models on a coarse grid ($\Delta \tau = 0.05$). On that grid, $\tau = 0.22$ was never evaluated; the prior achieved $\text{F1} = 0.3736$ at $\tau = 0.20$ and peaked at $\tau = 0.25$ with $\text{F1} = 0.3756$.

Comparing one model swept at $\Delta \tau = 0.01$ against another swept at $\Delta \tau = 0.05$ violates fair benchmarking standards.

---

## 2. Fair Comparison on Identical Search Grids

Both the trained acoustic neural model (`best.pt`, Epoch 6) and the canonical-phone empirical prior $P_{\text{train}}(\text{err} \mid \text{phone})$ (computed strictly from the 2,399 Fold 1 train utterances) were evaluated on the exact same 300 validation utterances and 9,956 phone tokens using identical grid protocols:

### Grid A: Coarse Grid ($\Delta \tau = 0.05$, 19 operating points from $0.05$ to $0.95$)
| Model | Optimal Threshold $\tau^*$ | Precision at $\tau^*$ | Recall at $\tau^*$ | Optimal F1 |
|---|---:|---:|---:|---:|
| **Acoustic Neural Model** | **0.20** | 0.2918 | 0.5385 | **0.3785** |
| **Canonical Empirical Prior** | **0.25** | 0.3594 | 0.3933 | **0.3756** |
| *Difference (Neural $-$ Prior)* | — | — | — | **+0.0029** |

### Grid B: Fine-Grained Grid ($\Delta \tau = 0.01$, 99 operating points from $0.01$ to $0.99$)
| Model | Optimal Threshold $\tau^*$ | Precision at $\tau^*$ | Recall at $\tau^*$ | Optimal F1 |
|---|---:|---:|---:|---:|
| **Acoustic Neural Model** | **0.21** | 0.3047 | 0.5143 | **0.3830** |
| **Canonical Empirical Prior** | **0.22** | 0.3156 | 0.5030 | **0.3879** |
| *Difference (Neural $-$ Prior)* | — | — | — | **-0.0049** |

### Key Insight:
- On a coarse grid (0.05 step), the neural model marginally leads (+0.0029 F1).
- On a fine-grained grid (0.01 step), the canonical prompt prior marginally leads (+0.0049 F1).
- Across both grids, their peak F1 scores differ by less than 0.005. Neither model demonstrates a decisive F1 superiority under exploratory threshold tuning.

---

## 3. Common Fixed-Threshold Comparison Table

To eliminate threshold search bias, both models are compared across identical fixed thresholds:

| Fixed Threshold $\tau$ | Neural Precision | Prior Precision | Neural Recall | Prior Recall | Neural F1 | Prior F1 | Higher F1 |
|---:|---:|---:|---:|---:|---:|---:|---|
| **0.10** | 0.1795 | 0.1672 | 0.8358 | 0.8686 | **0.2956** | 0.2804 | Neural (+0.0152) |
| **0.15** | 0.2356 | 0.2040 | 0.6525 | 0.6370 | **0.3463** | 0.3091 | Neural (+0.0372) |
| **0.20** | 0.2918 | 0.2803 | 0.5385 | 0.5601 | **0.3785** | 0.3736 | Neural (+0.0049) |
| **0.22** | 0.3173 | 0.3156 | 0.4745 | 0.5030 | 0.3803 | **0.3879** | Prior (+0.0076) |
| **0.25** | 0.3558 | 0.3594 | 0.3838 | 0.3933 | 0.3692 | **0.3756** | Prior (+0.0064) |
| **0.30** | 0.4140 | 0.4536 | 0.3267 | 0.3086 | 0.3652 | **0.3673** | Prior (+0.0021) |
| **0.35** | 0.4524 | 0.4536 | 0.3077 | 0.3086 | 0.3663 | **0.3673** | Prior (+0.0010) |
| **0.40** | 0.4576 | 0.4536 | 0.2939 | 0.3086 | 0.3579 | **0.3673** | Prior (+0.0094) |
| **0.50** | 0.4914 | 0.4724 | 0.1737 | 0.2593 | 0.2567 | **0.3348** | **Prior (+0.0781)** |

### Critical Finding at Standard Threshold ($\tau = 0.50$):
At the conventional threshold $\tau = 0.50$, **the Canonical Empirical Prior strongly outperforms the neural model** ($\text{F1} = 0.3348$ vs. $0.2567$, $+0.0781$).
- The neural model's recall collapses to **17.37%** at $\tau = 0.50$, missing more than 82% of all pronunciation errors.
- The canonical prompt prior maintains a recall of **25.93%** and precision of **47.24%** at $\tau = 0.50$ because 5 canonical phonemes in the training set exhibit empirical error rates $\ge 50\%$.

---

## 4. Threshold-Free Metric: Average Precision (PR-AUC)

Because threshold tuning on the validation set is post-hoc and prone to optimism bias, Average Precision (AP) serves as the primary scientific metric:
- **Acoustic Neural Model:** $\text{AP} = \mathbf{0.317021}$
- **Canonical Empirical Prior:** $\text{AP} = \mathbf{0.303063}$
- **Absolute Lift:** $\mathbf{+0.013958}$ (+1.40 percentage points)
- **Prevalence Baseline:** $0.116211$

### Methodological Rule:
Validation-optimized F1 scores (0.3830 and 0.3879) must **always be explicitly labeled as exploratory, post-hoc parameters**. They must not be cited as deployable performance metrics without a held-out test evaluation using pre-frozen thresholds.
