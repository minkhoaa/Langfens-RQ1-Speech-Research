# Scientific Claims Errata & Methodological Standards — RQ1-TRN-02A-EVAL-R2-CORR

**Status:** **AUTHORITATIVE ERRATUM COMPLETED**  
**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  

---

## 1. Summary of Prior Claims & Corrections

| Subject | Original Prior Claim | Scientific Audit Finding | Corrected Official Claim |
|---|---|---|---|
| **Numerical Tolerance** | "All historical metrics reproduced within sub-0.3% relative tolerance." | Continuous metrics (AP, PER) match within 0.30%. However, thresholded metrics at $\tau=0.50$ (TP, Recall, F1) changed by +1.01% to +1.52% due to 3 boundary token flips. | *"Global ranking metrics (AP 0.3170 vs 0.3161, PER 0.2368 vs 0.2364) reproduced within 0.30% arithmetic precision. At discrete threshold $\tau=0.50$, True Positives shifted from 198 to 201 (+1.52%) due to floating-point representation differences (GPU FP16 AMP vs CPU FP32)."* |
| **Baseline Superiority** | "Neural model significantly outperforms canonical-only prompt prior baseline (optimal F1 0.3775 vs 0.3879)." | The prior's 0.3879 F1 was found on a 0.01 grid, while the neural model was only evaluated on a 0.05 grid. On a common 0.01 grid, Prior F1 is 0.3879 vs Neural 0.3830. At $\tau=0.50$, Prior F1 is 0.3348 vs Neural 0.2567. | *"The multi-task neural model achieves higher Average Precision than the train-only empirical prompt prior (0.3170 vs 0.3031, +1.40 AP points). However, under exploratory fine-grained threshold tuning, their peak F1 scores are comparable (0.3830 vs 0.3879), and at standard threshold $\tau=0.50$, the prompt prior achieves higher F1 (0.3348 vs 0.2567)."* |
| **Acoustic Dependence** | "Model demonstrates authentic acoustic dependence without TextGrid timestamp dependency." | Under paired audio derangements, 96.76% of binary decisions at $\tau=0.50$ remain identical. The acoustic lift $\Delta \text{AP}$ varies from +0.0223 down to +0.0030 across different permutation functions. | *"Conditioning on the acoustic waveform yields a statistically detectable lift in Average Precision ($\Delta \text{AP} = +0.0223$, 95% CI $[+0.0048, +0.0396]$). However, the binary MDD head is heavily dominated by prompt phonotactic memorization, maintaining $\ge 96.76\%$ decision agreement when fed completely mismatched audio."* |
| **Statistical Generalization** | "Bootstrap confidence intervals confirm robust acoustic conditioning across speakers." | Resampling was performed across utterances within only two validation speakers (NJS and RRBI), providing 1 degree of freedom for cross-speaker variance. | *"The utterance-clustered bootstrap confidence interval reflects sampling variability within the two validation speakers (NJS and RRBI). It does not provide statistical evidence of population-level generalization across unseen speakers or accents."* |
| **Deployability** | "Timestamp-free inference is ready for real-world deployment without oracle information." | Model inputs rely on canonical phone sequences extracted from human annotator Gold TextGrids, which exhibit 57.5% divergence across speakers on identical sentences. | *"The current benchmark operates under Protocol O (Oracle TextGrid canonical phonemes). While timestamp boundaries are not used during inference, deployable text+audio inference requires a zero-oracle frontend (Protocol Z, CMUdict G2P), under which exploratory AP is 0.3125."* |
| **Validation Bias** | "Validation AP provides an unbiased measure of model performance." | Model checkpoint `best.pt` was selected from 25 training epochs specifically because it achieved the peak AP on this exact validation partition (NJS/RRBI). | *"Validation AP contains checkpoint selection bias (optimism bias) because the validation partition was used for model checkpoint selection. It is not an unbiased estimate of generalization to unseen speakers or accents."* |
| **Label Independence** | Ambiguity caused by passing `target_ctc` and `target_binary` during inference. | Code audit and regression tests prove bit-for-bit mathematical independence of logits ($\Delta = 0.0$). Targets affect loss only. | *"Logits are mathematically invariant to target labels. The inference pipeline has been strictly refactored to prediction-only signatures, keeping target tensors entirely outside model execution."* |

---

## 2. Mandatory Reporting Guidelines for Subsequent Research

1. **Explicit Protocol Designation:**
   - Any publication or thesis chapter presenting these results must state whether Protocol O (TextGrid-derived) or Protocol Z (G2P-derived) is reported.
2. **Threshold Labeling:**
   - Any optimal F1 cited (e.g. 0.3830) must be explicitly noted as: *"Exploratory post-hoc threshold tuned on validation partition."*
3. **Primary Metric Primacy:**
   - Average Precision (PR-AUC) must remain the primary evaluation metric for binary MDD because it evaluates the full ranking curve without threshold tuning artifacts.
