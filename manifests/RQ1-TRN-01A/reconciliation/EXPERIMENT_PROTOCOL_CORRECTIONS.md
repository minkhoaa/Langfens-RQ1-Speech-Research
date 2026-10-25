# LANGFENS — RQ1-TRN-01A-R: Experiment Protocol Corrections Report

**Execution Gate:** RQ1-TRN-01A-R — Pretrained Backbone & Inference Validity Gate  
**Deliverable Associated Path:** `manifests/RQ1-TRN-01A/reconciliation/EXPERIMENT_PROTOCOL_CORRECTIONS.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/`  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Scientific Reproducibility Reviewer  
**Gate Evaluation Status:** `PROTOCOL_CORRECTIONS: PASS`  

---

## 1. Executive Summary & Audit Mandate

During the preflight review of `RQ1-TRN-00` and `RQ1-TRN-01A`, several documentation inconsistencies and arithmetic ambiguities were identified across evaluation denominators, partition allocations, and metric reporting standards.

This report establishes the verified, frozen protocol specifications aligned with the immutable manifests in `manifests/RQ1-SPLIT-01/`.

---

## 2. Itemized Protocol Reconciliations

### Item 1: Evaluable Substitution Diagnosis Denominator ($S_{\text{eval}} = 13,840$)
- **Issue:** Earlier drafts recorded $S_{\text{eval}} = 13,829$, mistakenly subtracting all 269 corpus `'err'` tags from substitutions ($14,098 - 269 = 13,829$).
- **Reconciliation:** A full census audit of TextGrids confirmed that exactly 258 `'err'` tags occur in substitution intervals, while 11 occur in epenthetic additions.
- **Frozen Denominator:**
  $$S_{\text{eval}} = S_{\text{total}} - N_{\text{err, sub}} = 14,098 - 258 = \mathbf{13,840}$$
- **Diagnosis Accuracy Formula:**
  $$\text{Diag-Acc} = \frac{\sum_{i=1}^{S_{\text{eval}}} \mathbb{I}(\hat{p}_i = p_i)}{13,840}$$

---

### Item 2: Vietnamese Adaptation — LOSO Regime A Speaker Allocation
- **Issue:** Protocol documentation described Regime A as "23 train / 1 test", omitting validation speakers.
- **Reconciliation:** In `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json`, each of the 4 rounds explicitly allocates 2 validation speakers to ensure threshold calibration without test contamination.
- **Frozen Allocation (Each of 4 Rounds):**
  - **Train:** 21 speakers (including 3 Vietnamese).
  - **Validation:** 2 speakers (1 male, 1 female non-Vietnamese).
  - **Test:** 1 held-out Vietnamese speaker (`HQTV`, `PNV`, `THV`, or `TLV`).
  - Total: $21 + 2 + 1 = 24$ speakers.

---

### Item 3: Vietnamese Zero-Shot Generalization — Regime B Speaker Allocation
- **Issue:** Documentation cited "20 train / 4 test", omitting validation speakers.
- **Reconciliation:** Aligned with frozen split `regime_b_zero_shot_vietnamese_transfer`:
  - **Train:** 18 non-Vietnamese speakers (balanced across Arabic, Chinese, Hindi, Korean, Spanish).
  - **Validation:** 2 non-Vietnamese speakers.
  - **Test:** All 4 Vietnamese speakers evaluated jointly for zero-shot cross-lingual transfer.
  - Total: $18 + 2 + 4 = 24$ speakers.

---

### Item 4: Fold-Specific Test Denominators vs 4-Fold Aggregate
- **Issue:** Early smoke tests inadvertently compared Fold 1 evaluations against the 4-fold aggregate denominator ($119,151$).
- **Reconciliation:** Benchmark evaluations must use the exact partition denominator of the fold under evaluation:
  - **Fold 1 Test Denominator:** $N = \mathbf{29,769}$ canonical tokens ($C = 25,249$, $S = 3,744$, $D = 776$, $E = 4,520$).
  - **Fold 2 Test Denominator:** $N = \mathbf{29,798}$ canonical tokens ($C = 25,160$, $S = 3,649$, $D = 989$, $E = 4,638$).
  - **Fold 3 Test Denominator:** $N = \mathbf{29,798}$ canonical tokens ($C = 25,578$, $S = 3,467$, $D = 753$, $E = 4,220$).
  - **Fold 4 Test Denominator:** $N = \mathbf{29,786}$ canonical tokens ($C = 25,646$, $S = 3,238$, $D = 902$, $E = 4,140$).
  - **4-Fold Benchmark Aggregate:** $N = \mathbf{119,151}$ canonical tokens ($C = 101,633$, $S = 14,098$, $D = 3,420$, $E = 17,518$).

---

### Item 5: Primary vs Sensitivity Binary Policies
- **Primary Policy (Strict):** Evaluates all non-native substitutions and deletions as errors ($E = S + D = 17,518$; error rate: $14.702\%$). Evaluated via **PR-AUC** due to $5.80 : 1$ class imbalance.
- **Sensitivity Policy (Lenient):** Reclassifies the $1,112$ base-preserving allophonic variations (`CPL,CPL*,s`) as pedagogically acceptable ($0$). Lenient errors: $E_{\text{lenient}} = 16,406$ ($13.769\%$). Reported strictly as a sensitivity delta ($\Delta \text{PR-AUC}$, $\Delta \text{F1}$).

---

### Item 6: Validation-Only Threshold Calibration
- Operating cutoff $\theta^*$ must be calibrated strictly on validation splits:
  $$\theta^* = \arg\max_{\theta \in [0.01, 0.99]} \text{F1}_{\text{val}}(\theta)$$
- Test sets remain completely blinded until after $\theta^*$ is frozen.

---

### Item 7: Speaker-Clustered Bootstrap Confidence Intervals
- Because phone tokens within the same speaker share systematic acoustic and accent biases, standard i.i.d. token resampling underestimates standard errors.
- $95\%$ Confidence Intervals must be computed via **Speaker-Clustered Bootstrap** ($B = 1,000$ resamples clustered by speaker).

---

### Item 8: Comprehensive Intellectual Property & Licensing Attribution
- **Research Codebase:** MIT License.
- **Acoustic Backbone (`facebook/wav2vec2-base-960h`):** Apache-2.0.
- **Evaluation Corpora (L2-ARCTIC v5.0):** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- Derived model weights are licensed strictly for **non-commercial academic and educational research**.
