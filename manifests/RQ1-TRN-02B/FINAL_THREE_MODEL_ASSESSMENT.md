# Final Three-Model Scientific Assessment: Reference R, Control C0, and Treatment C1

**Document ID:** `LANGFENS-RQ1-TRN-02B-FINAL-THREE-MODEL-ASSESSMENT`<br>
**Assessment Date:** 2026-10-29<br>
**Current Git HEAD:** `b08842567ec92d0de000207a91b516f13c46ab13` on branch `review/rq1-trn-02b-kaggle-official-runs`<br>
**Evaluation Environment:** Local Laptop CPU (`torch 2.14.1+cpu`, Python 3.14.8, x86_64 Linux 6.18.2)<br>
**Evaluated Artifacts:**
- **Reference R:** `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`)
- **Control C0:** `checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt` (SHA-256 `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e`)
- **Treatment C1:** `checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt` (SHA-256 `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4`)
- **Control C0 (`best.pt`):** `checkpoints/RQ1-TRN-02B/C0/best.pt` (SHA-256 `4689170265de6fd6d9e06608afc3b307f62fe0f62939e657a6a68bc28f2ab4ba`)
- **Treatment C1 (`best.pt`):** `checkpoints/RQ1-TRN-02B/C1/best.pt` (SHA-256 `1d643b489fc7e0c1ef39fa1d35dbe25c3898749ee3b858b990ec01f46396b89a`)
**Test Partition Isolation:** STRICTLY SEALED (0 official test utterances accessed, staged, or inspected)<br>
**Scientific Verdict:** **NOT_SUPPORTED** (Ablation hypothesis rejected under matched training exposure; Reference R retained as primary production baseline)

---

## 1. Executive Summary

This document presents the definitive scientific comparison of the three primary Fold 1 models developed under Research Question 1 of the LANGFENS initiative:
1. **Reference R (Historical Baseline Anchor):** Produced in phase RQ1-TRN-02A on Colab T4 (commit `caa5678`), saved at epoch 6 (3,600 updates), establishing the benchmark for dual-head Wav2Vec2 + Binary MDD.
2. **Control C0 ($p=0.00$ Fine-Tuning Control):** Fine-tuned from the identical Baseline R checkpoint on Kaggle Tesla T4 $\times$ 2 without prompt dropout for strictly 6 epochs / 3,600 successful optimizer updates (0 AMP skips).
3. **Treatment C1 ($p=0.25$ Whole-Phone Prompt Dropout):** Fine-tuned from the identical Baseline R checkpoint on Kaggle Tesla T4 $\times$ 2 with $p=0.25$ whole-phone binary prompt embedding dropout under matched hyperparameters and exposure for strictly 6 epochs / 3,600 successful optimizer updates (0 AMP skips).

### Key Takeaways:
- **Primary Causal Contrast ($\Delta_{\text{dropout}} = \text{C1} - \text{C0}$):** Under matched training exposure, Treatment C1 achieves a matched binary AP of **0.302386**, compared to Control C0's **0.310304**, yielding a primary delta of **-0.007918** AP points.
- **Statistical Significance & Uncertainty:** The 1,000-resample speaker-stratified cluster bootstrap 95% confidence interval for $\Delta_{\text{dropout}}$ is **`[-0.018328, +0.000385]`** (mean: $-0.008329$). In 96.3% of bootstrap resamples, C1 achieves lower matched AP than C0. Because the upper confidence limit (+0.000385) technically crosses zero, the result is characterized as **NOT_SUPPORTED** rather than a definitively proven negative effect.
- **Acoustic Reliance Gap:** Control C0 maintains an acoustic gap of **0.017543** (95% CI `[+0.001212, +0.034191]`), whereas Treatment C1's acoustic gap contracts to **0.011423** (95% CI `[-0.003024, +0.025217]`), with its lower bound falling below zero. Prompt dropout degraded rather than increased reliance on acoustic features.
- **Comparison to Static Empirical Prior:** The train-only canonical phone error prior achieves **0.303063** on validation. Treatment C1 (**0.302386**) falls below this simple prior, whereas Control C0 (+0.0072) and Reference R (+0.0140) outperform it.
- **Protocol Z (Zero-Oracle Inference):** Under realistic zero-oracle conditions (CMUdict G2P phonemes, 99.63% token coverage, zero TextGrid timestamps), Treatment C1 AP (**0.300992**) is lower than Control C0 (**0.307582**) by **-0.006590**, independently confirming Protocol O findings.
- **Baseline Recommendation:** **Reference R (`best.pt`) is retained as the authoritative production baseline.** Control C0 is retained as a valuable secondary reference demonstrating significant CTC PER improvement (22.02% vs 23.68%). Treatment C1 is rejected.

---

## 2. Independent Provenance & Cryptographic Verification

All three models originate from a common, immutable historical ancestry and have been independently verified:

### 2.1 Cryptographic Digest Registry

| Component | Path / Remote Identifier | SHA-256 Digest | Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **Bit-Identical** |
| **Reference R (`best.pt`)** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **Bit-Identical** |
| **Reference R (`last.pt`)** | `checkpoints/RQ1-TRN-02A/recovered/last.pt` | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | **Bit-Identical** |
| **Control C0 Fixed Budget** | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C0/fixed_budget_checkpoint.pt`) | `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e` | **Bit-Identical** |
| **Control C0 `best.pt`** | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C0/best.pt`) | `4689170265de6fd6d9e06608afc3b307f62fe0f62939e657a6a68bc28f2ab4ba` | **Bit-Identical** |
| **Treatment C1 Fixed Budget** | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C1/fixed_budget_checkpoint.pt`) | `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4` | **Bit-Identical** |
| **Treatment C1 `best.pt`** | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C1/best.pt`) | `1d643b489fc7e0c1ef39fa1d35dbe25c3898749ee3b858b990ec01f46396b89a` | **Bit-Identical** |
| **Authorized Audio ZIP** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **Bit-Identical** |
| **G2P CMUdict** | `cmudict-0.7b` | `81917843d7c3df95a6ec99ebff0358cf5fcf30f80bc9b7b9ea3a479ff70a0fed` | **Bit-Identical** |

### 2.2 Verification of Training Commit Parity

An audit was conducted on the source tree between the commit recorded for Control Arm C0 (`ce05e389f581612939c775d9bdbb072511a514cd`) and Treatment Arm C1 (`032b0b3dad84ac47e8772d8cbb5a1f62ad780530`).

The exact `git diff` reveals only two added files:
- `deploy/kaggle_c1/kaggle_notebook.ipynb` (Dedicated notebook orchestrator for Arm C1)
- `deploy/kaggle_c1/kernel-metadata.json` (Kaggle kernel descriptor for `minkhoaa/langfens-rq1-c1-t4x2`)

**Audit Conclusion:** There were zero changes to model architecture, training loops, loss functions, optimization hyperparameters, dataset preparation, or random seeding between C0 and C1. The two training runs differed solely in deployment configuration and the intervention parameter `binary_prompt_dropout_p` (0.00 vs. 0.25).

---

## 3. Numerical Consistency Audit & Token Discrepancy Resolution

### 3.1 Investigation of the NJS/RRBI Token-Count Discrepancy

A discrepancy was identified between early Step 15 draft prose (`5,162` tokens for `NJS` and `4,794` tokens for `RRBI`) and the evaluator JSON reports (`5,011` tokens for `NJS` and `4,945` tokens for `RRBI`).

### 3.2 Root-Cause Identification & Empirical Proof

1. **Source of Ground Truth:** The Fold 1 validation partition consists of exactly 300 utterances (150 from speaker `NJS` and 150 from speaker `RRBI`).
2. **Census Verification:** Direct token accounting via `L2ArcticMDDDataset` yields:
   - Speaker `NJS` (Utterances 0–149): Exactly **5,011** non-padding tokens (641 error tokens, 4,370 correct tokens; prevalence = 12.7919%).
   - Speaker `RRBI` (Utterances 150–299): Exactly **4,945** non-padding tokens (516 error tokens, 4,429 correct tokens; prevalence = 10.4348%).
   - Combined Validation Partition: Exactly **9,956** non-padding tokens (1,157 error tokens, 8,799 correct tokens; prevalence = 11.6211%).
3. **Artifact Consistency:**
   - `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json` records `total_tokens: 5011` for NJS and `total_tokens: 4945` for RRBI across all evaluated models.
   - `manifests/RQ1-TRN-02B/evaluation/PAIRED_PREDICTION_RECORDS.json` contains 9,956 individual token records, perfectly partitioned into 5,011 for NJS and 4,945 for RRBI.
   - Historical baseline documentation `CORRECTED_PER_SPEAKER_CONTROLS.md` (RQ1-TRN-02A) independently confirms NJS = 5,011 and RRBI = 4,945.
4. **Conclusion:** The values `5,162` and `4,794` were an inadvertent prose transcription error in the initial draft of `PAIRED_SCIENTIFIC_EVALUATION.md` (which mistakenly shifted 151 tokens from RRBI to NJS). Both `PAIRED_SCIENTIFIC_EVALUATION.md` and `15-cpu-paired-scientific-evaluation.md` have been corrected to reflect the true, verified token counts: **5,011 for NJS** and **4,945 for RRBI**.

---

## 4. Comprehensive Three-Model Consolidated Metrics

### 4.1 Primary Fixed-Budget Suite (6 Epochs / 3,600 Updates)

Evaluated under strict inference mode (`torch.inference_mode()`, `binary_prompt_dropout_p=0.0`):

| Evaluation Dimension | Metric / Estimand | Reference R (Anchor) | Control C0 ($p=0.00$) | Treatment C1 ($p=0.25$) | Causal Delta ($\text{C1} - \text{C0}$) | Bootstrap 95% CI |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Protocol O (Oracle)** | **Matched Binary AP** | **0.317021** | 0.310304 | 0.302386 | **-0.007918** | **`[-0.018328, +0.000385]`** |
| | Mismatched Binary AP | 0.294722 | 0.292761 | 0.290963 | -0.001798 | N/A |
| | **Acoustic Reliance Gap** | **0.022299** | 0.017543 | 0.011423 | **-0.006120** | **`[-0.019675, +0.005596]`** |
| | Train Canonical Prior AP | 0.303063 | 0.303063 | 0.303063 | 0.000000 | Baseline Anchor |
| | Margin over Prior | +0.013958 | +0.007241 | **-0.000677** | -0.007918 | Negative for C1 |
| **Speech Recognition** | **Greedy CTC PER** | 23.68% | 22.02% | 22.01% | -0.01% | Both pass guardrail |
| | Levenshtein Edits | 2,340 | 2,176 | 2,175 | -1 edit | Reference: 9,881 |
| **Operational $\tau = 0.50$** | Precision | **49.14%** | 47.01% | 43.50% | -3.51% | False positives rise |
| | Recall / Sensitivity | 17.37% | 9.51% | 13.31% | +3.80% | C0 is conservative |
| | Specificity | 97.64% | **98.59%** | 97.73% | -0.86% | High true negatives |
| | $F_1$ Score | **0.2567** | 0.1582 | 0.2038 | +0.0456 | Tradeoff shift |
| | Accuracy | **88.31%** | 88.24% | 87.92% | -0.32% | High base accuracy |
| **Operational $\tau = 0.20$** | Precision | 29.18% | **33.53%** | 32.38% | -1.15% | C0 most precise |
| | Recall / Sensitivity | **53.85%** | 45.20% | 44.86% | -0.34% | R has highest recall |
| | Specificity | 82.82% | **88.21%** | 87.68% | -0.53% | C0 most specific |
| | $F_1$ Score | 0.3785 | **0.3850** | 0.3761 | -0.0089 | C0 achieves peak F1 |
| **Protocol Z (Zero-Oracle)** | **Aligned-Only AP** | **0.313530** | 0.307582 | 0.300992 | **-0.006590** | Consistent drop |
| | Token Coverage | 99.63% | 99.63% | 99.63% | 0.00% | 9,919 / 9,956 tokens |
| | Aligned Pairs | 9,919 | 9,919 | 9,919 | 0 | Bit-identical DP |

---

### 4.2 Disaggregated Per-Speaker Performance

| Speaker | Dimension / Metric | Reference R | Control C0 | Treatment C1 | Causal Delta ($\text{C1} - \text{C0}$) |
|---|---|:---:|:---:|:---:|:---:|
| **Speaker NJS** | Matched Binary AP | **0.394927** | 0.388309 | 0.390072 | +0.001763 |
| (5,011 tokens) | Mismatched Binary AP | 0.378744 | 0.381481 | 0.376703 | -0.004778 |
| (641 errors) | Acoustic Reliance Gap | **0.016183** | 0.006828 | 0.013370 | +0.006542 |
| (Prevalence 12.79%) | Greedy CTC PER | 23.41% | **21.52%** | 21.54% | +0.02% |
| **Speaker RRBI** | Matched Binary AP | **0.251519** | 0.246000 | 0.236243 | **-0.009757** |
| (4,945 tokens) | Mismatched Binary AP | 0.234979 | 0.230557 | 0.231688 | +0.001131 |
| (516 errors) | Acoustic Reliance Gap | **0.016540** | 0.015444 | 0.004555 | **-0.010889** |
| (Prevalence 10.43%) | Greedy CTC PER | 23.95% | 22.53% | **22.49%** | -0.04% |

### Key Per-Speaker Takeaways:
1. **Asymmetric Impact:** On speaker `NJS`, prompt dropout produces negligible AP change (+0.0018). However, on speaker `RRBI`, Treatment C1 suffers severe degradation (-0.0098 matched AP), and its acoustic gap collapses from 0.0154 to 0.0046 (a 70% collapse in acoustic reliance).
2. **Failure on Challenging Accents:** Speaker `RRBI` (Hindi L1) presents lower baseline error prevalence and acoustic distinctiveness than `NJS` (Vietnamese L1). The prompt dropout intervention destabilized error classification on the harder speaker distribution rather than forcing robust acoustic reliance.

---

### 4.3 Pairwise Decision Agreements Between Models

Evaluated across all 9,956 validation tokens:

| Threshold | Comparison Pair | Overall Agreement | Positive Agreement (Both) | Positive Jaccard | Relative Agreement $A \to B$ | Relative Agreement $B \to A$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **$\tau = 0.50$** | **C1 vs C0** | 98.09% (9,766) | 199 tokens | 0.5116 | 56.21% (199 / 354) | 85.04% (199 / 234) |
| | **C0 vs R** | 98.08% (9,765) | 226 tokens | 0.5420 | 96.58% (226 / 234) | 55.26% (226 / 409) |
| | **C1 vs R** | 98.36% (9,793) | 300 tokens | 0.6479 | 84.75% (300 / 354) | 73.35% (300 / 409) |
| **$\tau = 0.20$** | **C1 vs C0** | 95.25% (9,483) | 1,345 tokens | 0.7398 | 83.91% (1,345 / 1,603) | 86.22% (1,345 / 1,560) |
| | **C0 vs R** | 92.70% (9,229) | 1,484 tokens | 0.6712 | 95.13% (1,484 / 1,560) | 69.51% (1,484 / 2,135) |
| | **C1 vs R** | 93.57% (9,316) | 1,549 tokens | 0.7076 | 96.63% (1,549 / 1,603) | 72.55% (1,549 / 2,135) |

---

## 5. Primary vs. Exploratory (`best.pt`) Results

To respect the pre-registered scientific protocol, results must not conflate the primary matched-budget endpoint with validation-selected secondary checkpoints:

| Dimension / Metric | Primary Contract (Epoch 6 / 3,600 Updates) | Exploratory Checkpoints (`best.pt`) | Methodological Assessment |
|---|:---:|:---:|---|
| **Control C0 AP** | 0.310304 | 0.314170 (Epoch 4 / 2,400 updates) | C0 reached peak AP at Epoch 4. |
| **Treatment C1 AP** | 0.302386 | 0.321258 (Epoch 1 / 600 updates) | C1 peaked at Epoch 1 (barely fine-tuned). |
| **Apparent $\Delta_{\text{dropout}}$** | **-0.007918** | **+0.007088** | Apparent delta is confounded by 1,800 updates! |
| **Update Budget Parity** | **Strictly Matched (3,600 vs. 3,600)** | **Severely Confounded (600 vs. 2,400)** | **Unequal exposure masquerading as treatment.** |
| **Post-Peak Dynamics** | Complete convergence trajectory | Continuous monotonic decline in C1 | Prompt dropout damages stability over time. |

### Trajectory of C1 Validation AP across Epochs:
- **Epoch 1 (600 updates):** **0.320398** (peak selection point; model barely departed from Baseline R)
- **Epoch 2 (1,200 updates):** **0.310931** (-0.0095 decline)
- **Epoch 3 (1,800 updates):** **0.302134** (-0.0088 decline)
- **Epoch 4 (2,400 updates):** **0.301170** (-0.0010 decline)
- **Epoch 5 (3,000 updates):** **0.302629** (+0.0015 plateau)
- **Epoch 6 (3,600 updates):** **0.302332** (-0.0003 final endpoint)

**Scientific Rule:** Comparing Treatment C1 at 600 updates against Control C0 at 2,400 updates violates experimental control. The primary causal comparison must strictly compare both arms at matched exposure (3,600 updates), where C1 is unequivocally inferior to C0.

---

## 6. Model Comparative Profiles & Baseline Selection

### 6.1 Reference R (Historical Baseline Anchor)
- **Strengths:** Highest matched binary AP (**0.317021**); highest acoustic reliance gap (**0.022299**); highest Protocol Z zero-oracle AP (**0.313530**); highest margin over canonical prior (+0.0140); balanced acoustic lifts across both speakers (+0.0162 on NJS, +0.0165 on RRBI).
- **Limitations:** Higher CTC Phoneme Error Rate (**23.68%**); low recall at $\tau=0.50$ (17.37%); sensitivity to acoustic derangements varies with permutation function (`reverse` AP is 0.3140).
- **Practical Suitability:** Robust, frozen reference model.
- **Role in Research:** **Authoritative primary production baseline.**

### 6.2 Control C0 ($p=0.00$ Fine-Tuning Control)
- **Strengths:** Substantial CTC PER improvement (**22.02%**, a 1.66 percentage point gain over R); high specificity at $\tau=0.50$ (98.59%); highest precision and $F_1$ score at $\tau=0.20$ (Precision 33.53%, $F_1$ 0.3850); preserves positive acoustic reliance gap (0.017543, 95% CI `[0.0012, 0.0342]`).
- **Limitations:** Mild negative drift in matched AP (-0.0067 vs. R); conservative positive-class predictions at $\tau=0.50$ (Recall drops to 9.51%).
- **Practical Suitability:** Superior speech recognition backbone with slightly conservative binary scoring.
- **Role in Research:** Validated fine-tuning control; valuable reference for joint CTC/MDD multi-task optimization.

### 6.3 Treatment C1 ($p=0.25$ Whole-Phone Prompt Dropout)
- **Strengths:** Improved CTC PER (**22.01%**).
- **Limitations & Failure Modes:** Lowest matched AP (**0.302386**); performs *below* the static canonical prior (-0.0007); acoustic reliance gap contracts to 0.011423 with 95% CI spanning zero (`[-0.0030, +0.0252]`); severe performance collapse on speaker RRBI (-0.0098 AP, 70% collapse in acoustic gap); highest false positive count at $\tau=0.50$ (FP=200).
- **Practical Suitability:** Unsuitable for deployment. Whole-phone zero-masking destabilizes the binary classification head.
- **Role in Research:** Falsified ablation candidate.

---

## 7. Hypothesis Verdict & Scientific Attribution

### 7.1 Formal Criteria Evaluation (DESIGN-R1 Section 6.3)

| Criterion | Formal Rule | Measured Result | Verdict |
|---|---|:---:|:---:|
| **1. AP Superiority** | $\text{AP}_{\text{matched}}(\text{C1}) > \text{AP}_{\text{matched}}(\text{C0})$ | 0.302386 vs. 0.310304 ($\Delta = -0.007918$) | **FAILED** |
| **2. Acoustic Gap Expansion** | $\Delta_{\text{gap}}(\text{C1}) > \Delta_{\text{gap}}(\text{C0})$ | 0.011423 vs. 0.017543 ($\Delta = -0.006120$) | **FAILED** |
| **3. Prior Superiority** | $\text{AP}_{\text{matched}}(\text{C1}) > \text{AP}_{\text{prior}}$ | 0.302386 vs. 0.303063 ($\Delta = -0.000677$) | **FAILED** |
| **4. CTC Guardrail** | $\text{PER}(\text{C1}) \le \text{PER}(\text{R}) + 1.0\%$ | 22.01% vs. 24.68% ($\Delta = -1.67\%$) | **PASSED** |

### 7.2 Nuanced Statistical Statement (Task 8)

The hypothesis that whole-phone prompt embedding dropout ($p=0.25$) enhances acoustic reliance and improves mispronunciation detection is **NOT_SUPPORTED on Fold 1**:
- The point estimate for the primary contrast is negative ($\Delta_{\text{dropout}} = -0.007918$).
- In 96.3% of 1,000 paired bootstrap resamples, C1 achieves lower AP than C0.
- **Statistical Nuance:** Because the 95% bootstrap confidence interval upper bound slightly crosses zero (`[-0.018328, +0.000385]`), we do **not** claim a statistically definitive negative effect at the two-tailed $\alpha=0.05$ level. Rather, the evidence fails to support the hypothesis of a positive effect, and any potential positive effect is bounded below +0.0004 AP points with 95% confidence.

### 7.3 Mechanistic Explanation of Failure

In the dual-head architecture, the canonical phone embedding provides the semantic target. When the embedding is completely zeroed during training:
1. The classification MLP receives a zero vector concatenated with the acoustic context vector, forcing it to classify purely from acoustics without target context.
2. Because mispronunciation detection requires comparing *what was expected* against *what was spoken*, completely removing the expected phone forces the MLP to solve an unconditioned anomaly detection task rather than a target-conditioned mismatch task.
3. Over continued training, the MLP develops noisy internal representations that degrade score calibration, leading to increased false positives (FP increases from 124 in C0 to 200 in C1 at $\tau=0.50$) and degraded average precision.

---

## 8. Threats to Validity & Methodological Boundaries

To maintain scientific integrity:
1. **Single Random Seed:** Both C0 and C1 were executed under seed `1337`. While the negative point estimate is consistent across bootstrap resamples, seed-level initialization variance in projection heads cannot be assessed without multi-seed expansion.
2. **Limited Speaker Diversity:** Validation encompasses exactly two non-native speakers (`NJS` and `RRBI`). The cluster bootstrap interval quantifies utterance sampling variability within these two speakers, but does not generalize to unseen speaker populations.
3. **Validation Selection Bias:** Both C0 and C1 initialized from `best.pt`, which was historically selected on this validation partition.
4. **Untrained Diagnosis Head:** Head 3 (39-class substitution diagnosis) remained disabled and untrained per experimental scope.
5. **Sealed Official Test Partition:** The official test partition (6 speakers, 900 utterances) remains completely sealed and uninspected.
6. **Credential Hygiene Review (Task 10):**
   - Access to private Hugging Face persistent repositories (`minkhoaa/langfens-rq1-trn-02b-checkpoints`) relied on environment tokens (`HF_TOKEN`).
   - All credentials remained excluded from git tracking via `.gitignore` (`*.token`, `.env*`, `hf_token.txt`).
   - Automated scripts read tokens dynamically without logging sensitive tokens in stdout or console output.
   - Credentials were not modified, rotated, or committed during this evaluation.

---

## 9. Research Gate & Thesis Roadmap

### 9.1 Phase Closure
**RQ1-TRN-02B is officially completed.** The ablation hypothesis is falsified. No further GPU compute will be expended on whole-phone prompt dropout ($p=0.25$) on Fold 1.

### 9.2 Thesis Synthesis & Remaining Work for RQ1 Completion
To complete the broader thesis research question for RQ1:
1. **Retain Reference R as Primary Anchor:** Use Reference R (`best.pt`, AP 0.317021, PER 23.68%) for cross-phase benchmarking.
2. **Document Multi-Task PER Gains:** Highlight Control C0's finding that continued fine-tuning significantly improves CTC PER (22.02%, p < 10^-5) without major binary degradation.
3. **Potential Subsequent Research Directions (To be authorized separately):**
   - *Cross-Attention Feature Regularization:* Rather than zeroing prompt embeddings post-attention, explore regularizing cross-attention weights or acoustic query projections.
   - *Acoustic Contrastive Objectives:* Revisit Task B/C auxiliary acoustic contrastive losses.
   - *Cross-Fold Replication:* Evaluate Reference R architecture across Folds 2, 3, and 4 to establish speaker-population generalizability.

---

## 10. Evidence & Artifact References

- Primary Evaluation JSON: `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json`
- Prediction Ledger: `manifests/RQ1-TRN-02B/evaluation/PAIRED_PREDICTION_RECORDS.json`
- Exploratory Checkpoints JSON: `manifests/RQ1-TRN-02B/evaluation/exploratory_best_pt/PAIRED_EVALUATION_REPORT.json`
- Evaluation Monograph: `manifests/RQ1-TRN-02B/PAIRED_SCIENTIFIC_EVALUATION.md`
- Step Checkpoint 15: `manifests/RQ1-TRN-02B/checkpoints/15-cpu-paired-scientific-evaluation.md`
- Step Checkpoint 16: `manifests/RQ1-TRN-02B/checkpoints/16-final-three-model-assessment-and-gate-closure.md`
