# LANGFENS — RQ1-TRN-01B0-R: Binary Detection Calibration & Score Distribution Interpretation

**Gate Reference:** RQ1-TRN-01B0-R — Real-Data Learning Diagnostics & Evidence Reconciliation  
**Deliverable Path:** `manifests/RQ1-TRN-01B0/reconciliation/BINARY_CALIBRATION_INTERPRETATION.md`  
**Lead Speech ML Evaluation Engineer & Statistical Reviewer:** Senior Speech ML Research Engineer  
**Calibration Review Status:** `CALIBRATION_AUDIT: PASS (Default & Diagnostic Thresholds Disentangled)`  

---

## 1. Executive Summary & Review Mandate

In the executive summary of RQ1-TRN-01B0, the binary mispronunciation detection results were presented with potentially conflicting metrics:
- **Default Threshold (0.50):** F1 = $0.0000$, Precision = $0.0$, Recall = $0.0$.
- **Precision-Recall AUC (PR-AUC):** $0.4066$ (compared to a pre-training baseline of $0.1762$).
- **Optimized Threshold (0.20):** F1 = $0.4126$, Precision = $0.3632$, Recall = $0.4775$.

Presenting the threshold-optimized F1 ($0.4126$) alongside PR-AUC without explicit operational qualification created the risk that downstream consumers might misinterpret $0.4126$ as validated detector performance.

This document:
1. Explicitly reports default-threshold ($\tau = 0.5$) metrics and documents all-negative classification behavior.
2. Formally designates grid-search threshold tuning on the training subset as an **uncalibrated exploratory diagnostic (`TRAIN_ONLY_DIAGNOSTIC`)**.
3. Analyzes the empirical probability score distributions for correct vs. erroneous tokens to explain the underlying class imbalance dynamics.
4. Confirms that formal threshold calibration and temperature scaling are reserved strictly for the frozen Fold 1 validation split ($S_{\text{val}}$, 300 utterances) in a later authorized phase.

---

## 2. Canonical vs. Exploratory Evaluation Metrics

### 2.1 Default-Threshold Evaluation ($\tau = 0.50$)
Under standard uncalibrated inference with decision rule $\hat{y} = \mathbb{I}(\sigma(z) \ge 0.50)$:

| Metric | Pre-Training (Step 0) | Baseline Checkpoint (Step 100) | Repeat Sanity (Step 100) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $83.11\%$ ($876 / 1054$) | $83.11\%$ ($876 / 1054$) | $83.11\%$ ($876 / 1054$) |
| **Precision** | $0.00\%$ | $0.00\%$ | $0.00\%$ |
| **Recall** | $0.00\%$ | $0.00\%$ | $0.00\%$ |
| **F1-Score** | **$0.0000$** | **$0.0000$** | **$0.0000$** |
| **True Positives (TP)** | **$0$** | **$0$** | **$0$** |
| **False Positives (FP)** | **$0$** | **$0$** | **$0$** |
| **False Negatives (FN)** | **$178$** | **$178$** | **$178$** |
| **True Negatives (TN)** | **$876$** | **$876$** | **$876$** |
| **Classification Regime** | **All-Negative Collapse** | **All-Negative Collapse** | **All-Negative Collapse** |

**Observation:** At the default decision boundary $\tau = 0.50$, the model produces an **all-negative output**, identifying zero true mispronunciations. The apparent $83.11\%$ accuracy is an artifact of the $83.11\%$ negative class prevalence ($876$ correct vs. $178$ errors).

### 2.2 Exploratory Threshold Tuning (`TRAIN_ONLY_DIAGNOSTIC`)
When threshold $\tau$ is tuned post-hoc via grid search across $\tau \in [0.10, 0.90]$ with step $0.05$ **on the identical training data used for parameter updates**:

| Metric | Pre-Training (Step 0) | Baseline Checkpoint (Step 100) | Repeat Sanity (Step 100) |
| :--- | :---: | :---: | :---: |
| **Selected Diagnostic Threshold ($\tau^*$)** | $0.10$ | $0.20$ | $0.15$ |
| **Diagnostic Precision** | $16.89\%$ ($178 / 1054$) | $36.32\%$ ($85 / 234$) | $27.93\%$ ($119 / 426$) |
| **Diagnostic Recall** | $100.00\%$ ($178 / 178$) | $47.75\%$ ($85 / 178$) | $66.85\%$ ($119 / 178$) |
| **Diagnostic F1-Score** | **$0.2890$** | **$0.4126$** | **$0.3940$** |
| **Confusion Matrix at $\tau^*$** | TP=178, FP=876, FN=0, TN=0 | TP=85, FP=149, FN=93, TN=727 | TP=119, FP=307, FN=59, TN=569 |

**Scientific Restriction:** These values represent train-only diagnostics. Presenting $F_1 = 0.4126$ as validation performance or deployment capability is statistically invalid due to circular optimization.

---

## 3. Empirical Probability Score Distribution Analysis

To determine why all-negative prediction occurs at $\tau = 0.50$, the raw sigmoid probabilities $p_i = \sigma(z_i)$ were extracted and analyzed separately for ground-truth correct tokens ($y = 0$, $N = 876$) and error tokens ($y = 1$, $N = 178$):

### 3.1 Statistical Distribution Quantiles

```
========================================================================================
                      ERROR PROBABILITY SCORE DISTRIBUTIONS
========================================================================================
Token Class    Count     Min       P25     Median     Mean       P75       Max
----------------------------------------------------------------------------------------
Baseline Step 100 Checkpoint:
  Correct (0)    876   0.1047    0.1410    0.1486    0.1627    0.1932    0.3528
  Error (1)      178   0.1050    0.1512    0.1958    0.2145    0.2261    0.3530
----------------------------------------------------------------------------------------
Repeat Sanity Step 100 Checkpoint:
  Correct (0)    876   0.0944    0.1266    0.1370    0.1473    0.1658    0.3262
  Error (1)      178   0.0944    0.1393    0.1696    0.1942    0.2096    0.3259
========================================================================================
```

### 3.2 Root Cause Analysis: Prior-Matching Loss Equilibrium
1. **The Maximum Predicted Score is $\le 0.353$:**  
   Across all 1,054 tokens, not a single token received a predicted error probability above $0.3531$. Therefore, any threshold $\tau > 0.3531$ guarantees an all-negative output ($TP = 0, FP = 0$).
2. **Mean Prediction Matches Class Prevalence:**  
   The training subset contains $178 / 1,054 = 16.89\%$ errors. In standard unweighted binary cross-entropy loss:
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(z_i) + (1 - y_i) \ln (1 - \sigma(z_i)) \right]$$
   The global constant-output minimum occurs when the network bias satisfies $\sigma(b) = \pi_1 = 0.1689$. Because errors are a minority ($16.9\%$), the optimizer quickly drives the head bias negative ($b \approx \text{logit}(0.1689) \approx -1.59$) to achieve quick loss reduction.
3. **Genuine Ranking Separation Exists (PR-AUC):**  
   Despite the shifted distribution, the model learned genuine acoustic discrimination:
   - For error tokens, the mean predicted score is **$0.2145$**, with median **$0.1958$**.
   - For correct tokens, the mean predicted score is **$0.1627$**, with median **$0.1486$**.
   - The Precision-Recall Area Under the Curve (PR-AUC) rose from random baseline **$0.1762 \to 0.4066$**.
   - This proves that errors are systematically ranked higher than correct tokens, but the output logits require threshold calibration or cost-sensitive weighting to function as a binary classifier.

---

## 4. Calibration Protocol for Authorized Fold 1 Phase

To transition from uncalibrated raw scores to a calibrated detection system in the subsequent authorized phase:
1. **Validation Split Independence:**  
   The decision threshold $\tau^*$ must be calibrated strictly on the **Fold 1 validation split** ($S_{\text{val}} = \{\text{NJS}, \text{RRBI}\}$, 300 utterances), which has zero speaker or utterance overlap with the training set.
2. **Cost-Sensitive Loss Tuning:**  
   During full training, `pos_weight = (1 - \pi) / \pi \approx 5.0` should be evaluated in `MaskedBCEWithLogitsLoss` to penalize false negatives, centering the decision boundary around $0.50$.
3. **Evaluation Protocol:**  
   All detection metrics on held-out evaluation sets must report:
   - PR-AUC and ROC-AUC (threshold-invariant ranking quality).
   - F1, Precision, and Recall at the pre-determined validation-calibrated threshold $\tau^*_{\text{val}}$.

---

## 5. Audit Conclusion

$$\mathbf{CALIBRATION\_AUDIT: PASS}$$

**Formal Evaluation:** At the canonical threshold of $\tau = 0.50$, the current checkpoint produces zero true positives due to class imbalance prior alignment. The reported F1 of $0.4126$ is certified as an exploratory training diagnostic, while PR-AUC ($0.4066$) confirms genuine ranking capability. Formal calibration remains deferred to the frozen validation split.
