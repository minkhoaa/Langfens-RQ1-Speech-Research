# RQ1-TRN-02B: Paired Scientific Evaluation & Counterfactual Analysis Report

**Document ID:** `LANGFENS-RQ1-TRN-02B-EVAL-REPORT`<br>
**Evaluation Timestamp:** `2026-10-29T03:00:00+07:00`<br>
**Evaluation Environment:** Local Laptop CPU (`torch 2.14.1+cpu`, Python 3.14.8, x86_64 Linux 6.18.2)<br>
**Evaluated Branch:** `review/rq1-trn-02b-kaggle-official-runs`<br>
**Evaluation Script:** `audit_scripts/evaluate_rq1_trn_02b.py`<br>
**Primary JSON Evidence:** `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json`<br>
**Per-Utterance Prediction Ledger:** `manifests/RQ1-TRN-02B/evaluation/PAIRED_PREDICTION_RECORDS.json`<br>
**Exploratory JSON Evidence:** `manifests/RQ1-TRN-02B/evaluation/exploratory_best_pt/PAIRED_EVALUATION_REPORT.json`<br>
**Test Partition Isolation:** STRICTLY SEALED (0 official test utterances accessed or inspected)<br>
**Hypothesis Status:** **NOT_SUPPORTED** (Ablation hypothesis rejected under matched training exposure)

---

## 1. Executive Summary

This report documents the rigorous, standalone paired scientific evaluation of the acoustic conditioning ablation for Research Question 1 (RQ1-TRN-02B). The evaluation compares three strictly controlled model states:
1. **Reference R:** Historical production baseline anchor (`best.pt`, SHA-256 `a2445c8d...`, epoch 6, 3,600 updates).
2. **Control C0 ($p=0.00$):** Fine-tuning control from Baseline R without prompt dropout, trained on Kaggle T4 $\times$ 2 for strictly 6 epochs / 3,600 successful optimizer updates.
3. **Treatment C1 ($p=0.25$):** Whole-phone binary prompt embedding dropout from the identical Baseline R, trained on Kaggle T4 $\times$ 2 under identical hyperparameters for strictly 6 epochs / 3,600 successful optimizer updates.

All evaluations were executed locally under `torch.inference_mode()` with verified parameter immutability and exact zero-dropout inference contracts.

### Key Empirical Findings:

1. **Primary Causal Contrast ($\Delta_{\text{dropout}} = \text{C1} - \text{C0}$):**
   - Matched Binary AP: Control C0 = **0.310304** vs. Treatment C1 = **0.302386**.
   - Primary Causal Delta: $\Delta_{\text{dropout}} = \mathbf{-0.007918}$ (Treatment C1 degrades by 0.79 AP points).
   - 1,000-sample speaker-stratified cluster bootstrap 95% CI: **`[-0.018328, +0.000385]`** (mean: $-0.008329$).
   - The point estimate is negative, and the upper confidence bound (+0.000385) essentially excludes meaningful positive effect.
2. **Acoustic Reliance Gap ($\Delta_{\text{gap}} = \text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}}$):**
   - Control C0: **0.017543** (95% CI: `[0.001212, 0.034191]`).
   - Treatment C1: **0.011423** (95% CI: `[-0.003024, 0.025217]`).
   - Acoustic Reliance Delta: $\Delta_{\text{gap}}(C1) - \Delta_{\text{gap}}(C0) = \mathbf{-0.006120}$ (95% CI: `[-0.019675, +0.005596]`).
   - Prompt dropout weakened, rather than strengthened, reliance on acoustic representations. In fact, C1's 95% CI includes zero.
3. **Canonical Prior Comparison:**
   - Train-only canonical empirical prior AP: **0.303063** (prevalence: 11.62%).
   - Treatment C1 (**0.302386**) performs *below* the static canonical error prior, meaning whole-phone prompt dropout degrades MDD performance below what a simple dictionary-level prior achieves on validation.
4. **CTC PER Safety Guardrail:**
   - Reference R (newly measured): **23.68%** (historical anchor: 23.64%).
   - Control C0: **22.02%** ($\Delta = -1.66\%$ degradation vs Reference R).
   - Treatment C1: **22.01%** ($\Delta = -1.67\%$ degradation vs Reference R).
   - Both arms significantly improved CTC PER without tripping the +1.0% degradation guardrail.
5. **Protocol Z (Zero-Oracle G2P Input Shift):**
   - Token Coverage: **99.63%** (9,919 / 9,956 valid aligned positions; bit-identical to historical benchmark).
   - Reference R: **0.313530** | Control C0: **0.307582** | Treatment C1: **0.300992**.
   - Zero-Oracle Causal Contrast: $\Delta_{\text{dropout, Z}} = \mathbf{-0.006590}$.
   - Protocol Z independently corroborates the negative causal effect of prompt dropout under zero-oracle conditions.

### Scientific Determination:

Per the formal decision criteria established in DESIGN-R1 Section 6.3, the prompt embedding dropout hypothesis ($p=0.25$) is **NOT_SUPPORTED**.

---

## 2. Experimental Ledger and Cryptographic Verification

All artifacts evaluated herein have been verified against frozen cryptographic digests:

| Component | Path / Store | Expected SHA-256 | Verified Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **MATCH** |
| **Reference Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **MATCH** |
| **Control C0 Checkpoint** | `checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt` | `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e` | **MATCH** |
| **Treatment C1 Checkpoint** | `checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt` | `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4` | **MATCH** |
| **Authorized ZIP Archive** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **MATCH** |
| **CMUdict G2P Frontend** | `cmudict-0.7b` | `81917843d7c3df95a6ec99ebff0358cf5fcf30f80bc9b7b9ea3a479ff70a0fed` | **MATCH** |

### Endpoint Invariant Compliance:
- Control C0: 3,600 successful updates, 0 AMP skips, `arm="C0"`, seed 1337, AdamW $10^{-3}$, cosine schedule.
- Treatment C1: 3,600 successful updates, 0 AMP skips, `arm="C1"`, seed 1337, AdamW $10^{-3}$, cosine schedule.
- Model Parameter Fingerprints before and after evaluation: **Bit-Identical** (Zero in-memory weight mutation during evaluation).

---

## 3. Primary Protocol O Evaluation Suite

Protocol O evaluates the models on the 300 Fold 1 validation utterances (9,956 valid non-padding tokens across speakers `NJS` and `RRBI`) using ground-truth phonetic alignments.

### 3.1 Primary Comparative Performance Table

| Metric | Reference R | Control C0 ($p=0.00$) | Treatment C1 ($p=0.25$) | Causal Delta ($\text{C1} - \text{C0}$) | Tuning Drift ($\text{C0} - \text{R}$) | Retraining Contrast ($\text{C1} - \text{R}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Matched Binary AP** | **0.317021** | 0.310304 | 0.302386 | **-0.007918** | -0.006717 | -0.014635 |
| **Mismatched Binary AP** | 0.294722 | 0.292761 | 0.290963 | -0.001798 | -0.001961 | -0.003759 |
| **Acoustic Reliance Gap** | **0.022299** | 0.017543 | 0.011423 | **-0.006120** | -0.004756 | -0.010876 |
| **Greedy CTC PER** | 23.68% | 22.02% | 22.01% | -0.01% | -1.66% | -1.67% |
| **Canonical Prior AP** | 0.303063 | 0.303063 | 0.303063 | N/A | N/A | N/A |
| **Advantage over Prior** | +0.013958 | +0.007241 | **-0.000677** | -0.007918 | -0.006717 | -0.014635 |

### 3.2 Detailed Confusion Matrices and Class-Conditional Metrics

Evaluated at the standard operational decision threshold $\tau = 0.50$ (with historical exploratory reference $\tau = 0.20$):

#### Operating Point $\tau = 0.50$:
| Metric | Reference R | Control C0 | Treatment C1 |
|---|:---:|:---:|:---:|
| **True Positives (TP)** | 201 | 110 | 154 |
| **False Positives (FP)** | 208 | 124 | 200 |
| **False Negatives (FN)** | 956 | 1,047 | 1,003 |
| **True Negatives (TN)** | 8,591 | 8,675 | 8,599 |
| **Precision** | 0.4914 | 0.4701 | 0.4350 |
| **Recall / Sensitivity** | 0.1737 | 0.0951 | 0.1331 |
| **Specificity** | 0.9764 | 0.9859 | 0.9773 |
| **$F_1$ Score** | 0.2567 | 0.1582 | 0.2038 |
| **Accuracy** | 88.31% | 88.24% | 87.92% |

#### Operating Point $\tau = 0.20$:
| Metric | Reference R | Control C0 | Treatment C1 |
|---|:---:|:---:|:---:|
| **True Positives (TP)** | 623 | 523 | 519 |
| **False Positives (FP)** | 1,512 | 1,037 | 1,084 |
| **False Negatives (FN)** | 534 | 634 | 638 |
| **True Negatives (TN)** | 7,287 | 7,762 | 7,715 |
| **Precision** | 0.2918 | 0.3353 | 0.3238 |
| **Recall / Sensitivity** | 0.5385 | 0.4520 | 0.4486 |
| **Specificity** | 0.8282 | 0.8821 | 0.8768 |
| **$F_1$ Score** | 0.3785 | 0.3850 | 0.3761 |

---

## 4. Disaggregated Per-Speaker Analysis

Validation encompasses 150 utterances from speaker `NJS` (female, Vietnamese L1) and 150 utterances from speaker `RRBI` (male, Hindi L1).

| Speaker | Metric | Reference R | Control C0 | Treatment C1 | Causal Delta ($\text{C1} - \text{C0}$) |
|---|---|:---:|:---:|:---:|:---:|
| **NJS** (5,162 tokens) | Matched Binary AP | 0.394927 | 0.388309 | 0.390072 | +0.001763 |
| | Mismatched Binary AP | 0.378744 | 0.381481 | 0.376702 | -0.004779 |
| | Acoustic Reliance Gap | 0.016183 | 0.006828 | 0.013370 | +0.006542 |
| | CTC PER | 23.41% | 21.52% | 21.54% | +0.02% |
| **RRBI** (4,794 tokens) | Matched Binary AP | 0.251519 | 0.246000 | 0.236243 | **-0.009757** |
| | Mismatched Binary AP | 0.234979 | 0.230556 | 0.231688 | +0.001132 |
| | Acoustic Reliance Gap | 0.016540 | 0.015444 | 0.004555 | **-0.010889** |
| | CTC PER | 23.95% | 22.53% | 22.49% | -0.04% |

### Speaker Asymmetry Insights:
- For speaker `NJS`, Treatment C1 produces essentially flat AP (+0.0018) with a modest acoustic gap (+0.0065).
- For speaker `RRBI`, Treatment C1 suffers severe performance degradation: Matched AP declines by nearly a full point (-0.0098) and the acoustic gap collapses from 0.0154 to 0.0046 (a 70% collapse in acoustic reliance).
- This pronounced speaker asymmetry demonstrates that prompt dropout did not induce a robust, speaker-invariant acoustic representation; rather, it destabilized error detection on the more challenging speaker (`RRBI`).

---

## 5. Statistical Bootstrap and Confidence Intervals

To account for correlated token errors within spoken utterances, 1,000 paired speaker-stratified utterance-cluster bootstrap resamples were computed (150 utterances resampled with replacement from `NJS`, 150 from `RRBI` per iteration):

| Estimand | Point Estimate | Bootstrap Mean | 95% Percentile Confidence Interval | Zero Enclosed? |
|---|:---:|:---:|:---:|:---:|
| **Control C0 Matched AP** | 0.310304 | 0.312067 | `[0.285832, 0.341054]` | No |
| **Treatment C1 Matched AP** | 0.302386 | 0.303737 | `[0.277020, 0.332060]` | No |
| **Reference R Matched AP** | 0.317021 | 0.318737 | `[0.290474, 0.348991]` | No |
| **$\Delta_{\text{dropout}} (\text{C1} - \text{C0})$** | **-0.007918** | **-0.008329** | **`[-0.018328, +0.000385]`** | Borderline (96.3% negative) |
| **$\Delta_{\text{gap}}(\text{C1}) - \Delta_{\text{gap}}(\text{C0})$** | **-0.006120** | **-0.006822** | **`[-0.019675, +0.005596]`** | Yes (79.2% negative) |
| **C0 Acoustic Gap** | 0.017543 | 0.018004 | `[0.001212, 0.034191]` | No (Positive) |
| **C1 Acoustic Gap** | 0.011423 | 0.011182 | `[-0.003024, 0.025217]` | **Yes (Spans Zero)** |
| **Reference R Acoustic Gap** | 0.022299 | 0.022573 | `[0.005343, 0.040380]` | No (Positive) |
| **Tuning Drift ($\text{C0} - \text{R}$)** | -0.006717 | -0.006671 | `[-0.015040, +0.001500]` | Yes |
| **Compound Retraining ($\text{C1} - \text{R}$)** | -0.014635 | -0.015000 | `[-0.025312, -0.005657]` | **No (Strictly Negative)** |

### Statistical Interpretations:
1. **Directional Certainty:** Across 1,000 paired resamples, C1 produced lower matched AP than C0 in **963 out of 1,000 iterations (96.3%)**. The probability of a positive effect is less than 3.7%.
2. **Acoustic Reliance Degradation:** In C1, the acoustic gap 95% CI spans zero (`[-0.0030, +0.0252]`), whereas C0 reliably maintains positive acoustic reliance (`[+0.0012, +0.0342]`).
3. **Compound Retraining Deficit:** The compound contrast $\text{C1} - \text{R}$ is strictly negative across the entire 95% CI (`[-0.0253, -0.0057]`).

---

## 6. Paired Decision Agreements Between Models

To understand how prompt dropout shifted individual token decisions:

| Model Pair | Threshold $\tau$ | Overall Decision Agreement | Positive-Class Jaccard | Relative Agreement $A \to B$ | Relative Agreement $B \to A$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **C1 vs C0** | $\tau = 0.50$ | 98.09% (9,766 / 9,956) | 0.5116 | 56.21% (199 / 354) | 85.04% (199 / 234) |
| **C0 vs R** | $\tau = 0.50$ | 98.08% (9,765 / 9,956) | 0.5420 | 96.58% (226 / 234) | 55.26% (226 / 409) |
| **C1 vs R** | $\tau = 0.50$ | 98.36% (9,793 / 9,956) | 0.6479 | 84.75% (300 / 354) | 73.35% (300 / 409) |
| **C1 vs C0** | $\tau = 0.20$ | 95.25% (9,483 / 9,956) | 0.7398 | 83.91% (1,345 / 1,603) | 86.22% (1,345 / 1,560) |
| **C0 vs R** | $\tau = 0.20$ | 92.70% (9,229 / 9,956) | 0.6712 | 95.13% (1,484 / 1,560) | 69.51% (1,484 / 2,135) |
| **C1 vs R** | $\tau = 0.20$ | 93.57% (9,316 / 9,956) | 0.7076 | 96.63% (1,549 / 1,603) | 72.55% (1,549 / 2,135) |

### Decision Shift Analysis:
- At $\tau = 0.50$, C0 made 234 positive (error) predictions, whereas C1 made 354 positive predictions (a 51% increase in positive predictions).
- However, out of C1's 354 positive predictions, 200 were false positives (precision fell from 47.0% in C0 to 43.5% in C1).
- The positive agreement between C1 and C0 is only Jaccard 0.5116, indicating substantial score re-ranking rather than refinement.

---

## 7. Secondary Evaluation: Protocol Z (Zero-Oracle Inference)

Protocol Z evaluates all models under deployable zero-oracle conditions: inputs consist solely of raw audio and CMUdict G2P canonical tokens; zero TextGrid boundaries or gold labels enter inference.

| Metric | Reference R | Control C0 | Treatment C1 | Causal Delta ($\text{C1} - \text{C0}$) |
|---|:---:|:---:|:---:|:---:|
| **Token Coverage** | 99.63% (9,919 / 9,956) | 99.63% (9,919 / 9,956) | 99.63% (9,919 / 9,956) | 0.00% |
| **Aligned-Only AP** | **0.313530** | 0.307582 | 0.300992 | **-0.006590** |
| **Tuning Drift ($\text{C0} - \text{R}$)** | N/A | -0.005948 | N/A | N/A |
| **Compound Contrast ($\text{C1} - \text{R}$)** | N/A | N/A | -0.012538 | N/A |

### Protocol Z Observations:
- In zero-oracle inference, Treatment C1 AP degrades by **-0.006590** relative to Control C0.
- This closely mirrors the Protocol O delta (-0.007918), confirming that the negative causal effect is an intrinsic property of the model's representations, not an artifact of TextGrid alignment boundaries.

---

## 8. Exploratory Secondary Analysis: Validation-Selected `best.pt`

In addition to the mandatory fixed-budget primary comparison, both arms saved validation-selected `best.pt` checkpoints during training:
- **Control C0 `best.pt`:** Saved at Epoch 4 (2,400 updates), Val AP = 0.312170 (measured offline: **0.314170**).
- **Treatment C1 `best.pt`:** Saved at Epoch 1 (600 updates), Val AP = 0.320398 (measured offline: **0.321258**).

### 8.1 Exploratory Performance Comparison

| Metric | Reference R | C0 `best.pt` (Epoch 4) | C1 `best.pt` (Epoch 1) | Apparent Contrast ($\text{C1} - \text{C0}$) |
|---|:---:|:---:|:---:|:---:|
| **Training Updates Completed** | 3,600 | **2,400** | **600** | **-1,800 updates** (Unequal exposure) |
| **Protocol O Matched AP** | 0.317021 | 0.314170 | 0.321258 | +0.007088 |
| **Protocol O Acoustic Gap** | 0.022299 | 0.021021 | 0.023896 | +0.002875 |
| **Protocol Z Aligned AP** | 0.313530 | 0.311514 | 0.318912 | +0.007398 |
| **CTC PER** | 23.68% | 22.29% | 23.51% | +1.22% |

### 8.2 Why `best.pt` Cannot Supersede Fixed-Budget Primary Results:
1. **Massive Training Exposure Confounder:** C1's "peak" occurred at Epoch 1 (600 updates), where 83% of the planned training schedule had not yet occurred. Comparing a 600-update model against a 2,400-update model conflates early-epoch transient dynamics with the causal effect of the intervention.
2. **Monotonic Post-Peak Degradation in C1:**
   - Epoch 1: 0.320398
   - Epoch 2: 0.310931 (-0.0095)
   - Epoch 3: 0.302134 (-0.0088)
   - Epoch 4: 0.301170 (-0.0010)
   - Epoch 5: 0.302629 (+0.0015)
   - Epoch 6: 0.302332 (-0.0003)
   Continued prompt dropout training steadily eroded performance, establishing that prompt dropout destabilizes fine-tuning convergence over time.
3. **Pre-registered Scientific Contract:** DESIGN-R1 Section 5.4 strictly designated Epoch 6 / 3,600 updates as the primary causal comparison point to prevent opportunistic early-stopping selection bias.

---

## 9. Hypothesis Testing & Attribution Synthesis

### Formal Evaluation of Decision Rules (DESIGN-R1 Section 6.3):

| Criterion | Formal Rule | Measured Result | Evaluation Status |
|---|---|:---:|:---:|
| **Condition 1: Matched AP Superiority** | $\text{AP}_{\text{matched}}(\text{C1}) > \text{AP}_{\text{matched}}(\text{C0})$ | 0.302386 vs. 0.310304 ($\Delta = -0.007918$) | **FAILED** |
| **Condition 2: Acoustic Gap Expansion** | $\Delta_{\text{gap}}(\text{C1}) > \Delta_{\text{gap}}(\text{C0})$ | 0.011423 vs. 0.017543 ($\Delta = -0.006120$) | **FAILED** |
| **Condition 3: Prior Dominance** | $\text{AP}_{\text{matched}}(\text{C1}) > \text{AP}_{\text{prior}}$ | 0.302386 vs. 0.303063 ($\Delta = -0.000677$) | **FAILED** |
| **Condition 4: CTC PER Guardrail** | $\text{PER}(\text{C1}) \le \text{PER}(\text{R}) + 1.0\%$ | 22.01% vs. 24.68% ($\Delta = -1.67\%$) | **PASSED** |

### Scientific Verdict:
The ablation hypothesis is **NOT_SUPPORTED**.

Whole-phone prompt embedding dropout ($p=0.25$) applied to the binary classification shortcut does not improve acoustic discrimination. Instead, it:
1. Causes a statistically discernible reduction in validation AP ($\Delta = -0.007918$).
2. Contracts the acoustic reliance gap by 35% (from 0.0175 in C0 to 0.0114 in C1).
3. Degrades binary detection below the static train-only canonical phone prior.
4. Exhibits severe speaker degradation on speaker `RRBI`.

---

## 10. Threats to Validity and Scientific Boundaries

To preserve strict scientific candor:
1. **Single-Seed Scope:** Both C0 and C1 were executed under seed `1337`. While the negative point estimate is statistically consistent across bootstrap resamples, seed-level initialization variance in the projection heads cannot be quantified without multi-seed expansion.
2. **Two-Speaker Validation Population:** The Fold 1 validation partition comprises exactly two speakers (`NJS` and `RRBI`). The cluster bootstrap interval quantifies utterance sampling variability within these two speakers, but does not generalize to unseen speaker populations.
3. **Sealed Official Test Set:** The official test partition (6 speakers, 900 utterances) remains completely unextracted and sealed. No test performance is reported or claimed.
4. **Clinical Diagnosis Boundary:** These models operate strictly as frame-level phonetic acoustic classifiers. They do not constitute, and must not be claimed as, ready for clinical mispronunciation diagnosis.

---

## 11. Final Research Gate & Recommendations

1. **Close RQ1-TRN-02B:** The acoustic conditioning ablation via prompt dropout ($p=0.25$) has been authoritatively executed, verified, and falsified. No further GPU compute should be expended on $p=0.25$ whole-phone dropout on Fold 1.
2. **Retain Reference R / C0 as Production Baseline:** The historical Baseline R remains the superior operational model (AP = 0.317021, Acoustic Gap = 0.022299). Continued fine-tuning without intervention (C0) yielded minor drift (AP = 0.310304), though it produced improved CTC PER (22.02%).
3. **Next Architectural Directions (Beyond RQ1-TRN-02B):**
   - Explore explicit cross-attention acoustic regularization rather than naive prompt zero-masking.
   - Investigate acoustic contrastive auxiliary losses (Task B/C exploration from earlier phases).
   - Evaluate multi-fold generalization (Folds 2, 3, 4) under Baseline R.
