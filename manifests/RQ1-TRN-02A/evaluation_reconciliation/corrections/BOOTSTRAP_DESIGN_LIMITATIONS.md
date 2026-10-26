# Bootstrap Design Audit & Statistical Limitations — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Statistical audit of utterance-clustered bootstrap resampling and population generalization boundaries  

---

## 1. Audit of the Utterance-Clustered Bootstrap Design

In EVAL-R2, an utterance-clustered bootstrap (1,000 resamples of the 300 validation utterances) was executed to evaluate the sampling distribution of Average Precision under matched and mismatched audio:

- **Matched Audio AP Mean:** $0.318120$ | 95% CI: $[0.287232, 0.348134]$
- **Mismatched Audio AP (`shift_1`) Mean:** $0.295606$ | 95% CI: $[0.265070, 0.327052]$
- **$\Delta \text{AP}$ (Matched $-$ Mismatched) Mean:** $+0.022514$ | 95% CI: $[+0.004782, +0.039575]$

---

## 2. Statistical Limitations of the Current Bootstrap Design

### Limitation 1: Within-Sample Cross-Cluster Pairing Dependencies
When resampling utterances with replacement:
- Each resampled utterance $u_i^*$ carries its ground-truth labels $Y_i$, its matched audio score $\hat{S}_{\text{matched}, i}$, and its mismatched score $\hat{S}_{\text{mismatched}, i}$ derived from donor utterance $\pi(i)$.
- However, donor utterance $\pi(i)$ is itself a member of the same validation partition. In a bootstrap draw, utterance $i$ may be selected while $\pi(i)$ is omitted, or vice-versa.
- Because the audio donor pool is not an independent external distribution, the paired predictions share unmodeled cross-utterance acoustic covariance. This internal closed-loop permutation tends to underestimate the true variance of acoustic substitution.

### Limitation 2: Sensitivity to Permutation Function
As demonstrated in `CORRECTED_PER_SPEAKER_CONTROLS.md`, the observed $\Delta \text{AP}$ is sensitive to the specific permutation function applied:
- `shift_1`: $\Delta \text{AP} = +0.0223$
- `shift_2`: $\Delta \text{AP} = +0.0144$
- `shift_75`: $\Delta \text{AP} = +0.0169$
- `reverse`: $\Delta \text{AP} = +0.0030$

The 1,000-iteration bootstrap evaluated only a single fixed permutation (`shift_1`). Evaluating across permutations reveals that the acoustic lift $\Delta \text{AP}$ drops as low as $+0.0030$ under `reverse` (and is negative on Speaker NJS), significantly weakening claims of a uniformly robust acoustic benefit across all acoustic mismatch conditions.

### Limitation 3: The Two-Speaker Ecological Fallacy
The most critical limitation is that **Fold 1 validation contains only two speakers** (`NJS`, Spanish L1; `RRBI`, Russian L1):
1. Utterance-clustered bootstrap resamples utterances within these two specific individuals.
2. Therefore, the resulting confidence interval $[+0.0048, +0.0396]$ measures **within-speaker utterance sampling variability**—i.e., whether the acoustic difference holds across the particular sentences spoken by NJS and RRBI.
3. It **does NOT measure cross-speaker population variance**. In classical mixed-effects terminology, speakers are random effects, and two clusters provide exactly **1 degree of freedom** for between-subject variance.
4. Presenting this confidence interval as proof that the model generalizes to the broader population of L2 English learners is an ecological fallacy.

---

## 3. Methodological Rectifications & Required Reporting Guidelines

1. **Explicit Qualification of CIs:**
   - Any citation of the bootstrap interval must state: *"The 95% confidence interval ($[+0.0048, +0.0396]$) reflects utterance-level resampling within two validation speakers (NJS and RRBI) under a single adjacent cyclic derangement. It does not estimate cross-speaker or cross-accent population variance."*
2. **Prohibition of Causal Generalization Claims:**
   - Phrases such as *"proves robust acoustic conditioning across speakers"* or *"demonstrates that the model reliably extracts phonetic errors from speech waveforms"* are scientifically invalid and must be removed from all manuscripts and project reports.
3. **Requirement for Multi-Speaker Stratification:**
   - Reliable inference on acoustic contribution requires pooling across all 4 folds (8 validation speakers) or evaluating on an external multi-speaker corpus.
