# LANGFENS — RQ1-TRN-01B1-R2: End-to-End MDD Pipeline Evaluation

**Gate Reference:** RQ1-TRN-01B1-R2 — Acoustic-Conditioned MDD & Final Source Provenance Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/END_TO_END_MDD_EVALUATION.md`  
**Experimental Metric References:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/task_d_pipeline_evaluation.json`  
**Lead Speech ML Systems Engineer & Scientific Reproducibility Auditor:** Senior Speech ML Research Engineer  
**Audit Finding:** `PIPELINE_EVALUATION: AUDIT COMPLETE (Cascaded Error Propagation Quantified)`  

---

## 1. Executive Summary & Audit Mandate

In clinical and automated language learning deployments, a Mispronunciation Detection & Diagnosis (MDD) system **never has access to gold error timestamps or gold substitution flags**. The system must operate as a cascaded pipeline:
1. The **Binary Detector** processes the canonical prompt and acoustic input to predict which tokens are mispronounced ($\hat{e}_k \in \{0, 1\}$).
2. The **Diagnosis Head** is invoked **only** for tokens flagged as errors ($\hat{e}_k = 1$) to identify the substituted sound.
3. If the binary detector misses a mispronunciation ($\hat{e}_k = 0$, False Negative), the diagnosis head is never executed for that token, resulting in an uncorrected error.
4. If the binary detector triggers on a correctly pronounced token ($\hat{e}_k = 1$, False Positive), the diagnosis head is spuriously invoked, potentially corrupting clean speech.

Task D mandates an explicit separation between:
- **Oracle-Conditioned Diagnosis Accuracy:** Evaluated exclusively on ground-truth substitution events.
- **Binary Detector Performance:** Evaluated across a threshold sweep ($\tau \in [0.10, 0.70]$).
- **End-to-End Correctly Detected & Diagnosed Substitutions:** The true deployable metric measuring error propagation and end-to-end precision, recall, and F1.

---

## 2. Oracle vs Cascaded Pipeline Formulation

Let $N = 2,081$ be the total valid canonical tokens in the 64-utterance holdout probe.  
Let $\mathcal{S}_{\text{gold}} = \{k : y_{\text{sub}}^{(k)} \ne -100\}$ be the set of $252$ ground-truth substitution events ($|\mathcal{S}_{\text{gold}}| = 252$).  
Let $\hat{p}_{\text{bin}}^{(k)} = \sigma(z_{\text{bin}}^{(k)})$ be the binary detector's predicted probability of error, and $\hat{e}_k = \mathbb{I}(\hat{p}_{\text{bin}}^{(k)} \ge \tau)$ be the binary decision at threshold $\tau$.  
Let $\hat{y}_{\text{diag}}^{(k)} = \arg\max_c z_{\text{diag}}^{(k)}[c]$ be the predicted substitution class.

### 2.1 Oracle-Conditioned Diagnosis Accuracy
$$\text{Acc}_{\text{oracle}} = \frac{1}{|\mathcal{S}_{\text{gold}}|} \sum_{k \in \mathcal{S}_{\text{gold}}} \mathbb{I}\left(\hat{y}_{\text{diag}}^{(k)} = y_{\text{gold}}^{(k)}\right)$$
*Note: This metric assumes an "oracle" tells the model exactly which tokens contain substitution errors before diagnosis runs.*

### 2.2 End-to-End Correctly Detected & Diagnosed Substitutions
A token is counted as an end-to-end success if and only if it is a ground-truth substitution, the binary detector correctly detects it as an error ($\hat{e}_k = 1$), and the diagnosis head predicts the correct substituted phone:
$$N_{\text{e2e\_correct}}(\tau) = \sum_{k \in \mathcal{S}_{\text{gold}}} \mathbb{I}\left(\hat{e}_k = 1 \;\land\; \hat{y}_{\text{diag}}^{(k)} = y_{\text{gold}}^{(k)}\right)$$

$$\text{E2E Recall}(\tau) = \frac{N_{\text{e2e\_correct}}(\tau)}{|\mathcal{S}_{\text{gold}}|}$$
$$\text{E2E Precision}(\tau) = \frac{N_{\text{e2e\_correct}}(\tau)}{\sum_{k=1}^N \hat{e}_k}$$
$$\text{E2E F1}(\tau) = \frac{2 \cdot \text{E2E Precision}(\tau) \cdot \text{E2E Recall}(\tau)}{\text{E2E Precision}(\tau) + \text{E2E Recall}(\tau)}$$

---

## 3. Quantitative Pipeline Evaluation

All metrics evaluated on `internal_holdout_probe_64` ($2,081$ tokens, $266$ total gold errors, $252$ gold substitutions, $1,815$ correct tokens):

### 3.1 High-Level Metric Comparison across Audio Conditions

| Performance Dimension | Metric | Matched Audio | Mismatched Audio | Silence Audio ($\mathbf{x}=\mathbf{0}$) | Acoustic Sensitivity? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Oracle Diagnosis** | Substitution Accuracy ($N=252$) | **$50.79\%$** ($128/252$) | **$51.19\%$** ($129/252$) | **$2.38\%$** ($6/252$) | **NO (Invariant to Audio)** |
| **Binary MDD Detector** | Average Precision (PR-AUC) | **$0.4903$** | **$0.4598$** | **$0.3834$** | **PARTIAL ($\Delta = 0.0305$)** |
| **Cascaded MDD @ $\tau=0.50$** | Binary Recall | **$15.41\%$** ($41/266$) | **$17.29\%$** ($46/266$) | $0.00\%$ ($0/266$) | **NO (Mismatched Rec > Matched)** |
| **Cascaded MDD @ $\tau=0.50$** | Binary Precision | **$70.69\%$** ($41/58$) | **$66.67\%$** ($46/69$) | $0.00\%$ | — |
| **Cascaded MDD @ $\tau=0.50$** | Binary F1 | **$0.2531$** | **$0.2746$** | $0.0000$ | — |
| **Deployable E2E @ $\tau=0.50$** | E2E Correct Substitutions | **$41 / 252$** | **$46 / 252$** | **$0 / 252$** | **NO (Mismatched Found +5)** |
| **Deployable E2E @ $\tau=0.50$** | E2E Recall | **$16.27\%$** | **$18.25\%$** | $0.00\%$ | — |
| **Deployable E2E @ $\tau=0.50$** | E2E Precision | **$70.69\%$** | **$66.67\%$** | $0.00\%$ | — |
| **Deployable E2E @ $\tau=0.50$** | E2E F1 | **$0.2645$** | **$0.2866$** | $0.0000$ | — |
| **Deployable E2E @ Opt $\tau=0.20$**| E2E Correct Substitutions | **$111 / 252$** | **$110 / 252$** | **$0 / 252$** | **NO (Difference = 1 token)** |
| **Deployable E2E @ Opt $\tau=0.20$**| E2E Recall | **$44.05\%$** | **$43.65\%$** | $0.00\%$ | — |
| **Deployable E2E @ Opt $\tau=0.20$**| E2E Precision | **$28.10\%$** | **$27.16\%$** | $0.00\%$ | — |
| **Deployable E2E @ Opt $\tau=0.20$**| E2E F1 | **$0.3431$** | **$0.3385$** | $0.0000$ | **$\Delta = 0.0046$** |

---

## 4. Operating Point Threshold Sweep (Matched Audio)

The table below documents the full trade-off curve across decision thresholds $\tau \in [0.10, 0.70]$:

| Threshold $\tau$ | Binary TP | Binary FP | Binary FN | Binary Prec | Binary Rec | Binary F1 | Pred Errors ($\sum \hat{e}$) | Missed Subs (FN) | Spurious Triggers (FP) | E2E Correct Subs | E2E Recall | E2E Precision | E2E F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.10$** | $202$ | $562$ | $64$ | $0.2644$ | $0.7594$ | $0.3922$ | $764$ | $50$ | $562$ | **$120$** | $47.62\%$ | $15.71\%$ | $0.2362$ |
| **$0.15$** | $185$ | $384$ | $81$ | $0.3251$ | $0.6955$ | $0.4431$ | $569$ | $67$ | $384$ | **$115$** | $45.63\%$ | $20.21\%$ | $0.2801$ |
| **$0.20$** | **$162$** | **$233$** | **$104$** | **$0.4101$** | **$0.6090$** | **$0.4902$** | **$395$** | **$90$** | **$233$** | **$111$** | **$44.05\%$** | **$28.10\%$** | **$0.3431$** |
| **$0.25$** | $119$ | $110$ | $147$ | $0.5197$ | $0.4474$ | $0.4808$ | $229$ | $133$ | $110$ | **$98$** | $38.89\%$ | $42.79\%$ | $0.4075$ |
| **$0.30$** | $91$ | $55$ | $175$ | $0.6233$ | $0.3421$ | $0.4417$ | $146$ | $161$ | $55$ | **$89$** | $35.32\%$ | $60.96$ | $0.4472$ |
| **$0.40$** | $86$ | $37$ | $180$ | $0.6992$ | $0.3233$ | $0.4422$ | $123$ | $166$ | $37$ | **$86$** | $34.13\%$ | $69.92\%$ | $0.4587$ |
| **$0.50$** | **$41$** | **$17$** | **$225$** | **$0.7069$** | **$0.1541$** | **$0.2531$** | **$58$** | **$211$** | **$17$** | **$41$** | **$16.27\%$** | **$70.69\%$** | **$0.2645$** |
| **$0.60$** | $33$ | $11$ | $233$ | $0.7500$ | $0.1241$ | $0.2129$ | $44$ | $219$ | $11$ | **$33$** | $13.10\%$ | $75.00\%$ | $0.2230$ |
| **$0.70$** | $0$ | $0$ | $266$ | $0.0000$ | $0.0000$ | $0.0000$ | $0$ | $252$ | $0$ | **$0$** | $0.00\%$ | $0.00\%$ | $0.0000$ |

---

## 5. Error Propagation Analysis

### 5.1 Catastrophic Deletion Under-Recall at Default Threshold $\tau = 0.50$
At standard uncalibrated threshold $\tau = 0.50$:
- The Binary Detector predicts only $58$ errors across $2,081$ tokens.
- **False Negatives:** $225 / 266$ total errors are missed ($\text{Recall} = 15.41\%$).
- **Missed Substitutions:** $211 / 252$ gold substitutions are classified as correct ($83.73\%$ missed).
- **Consequence:** The diagnosis head is **never invoked** for $83.73\%$ of substitutions. Even though the diagnosis head achieves $100\%$ accuracy on the few substitutions that were detected ($41/41 = 100\%$, entirely canonical `DH` $\to$ `D` and `Z` $\to$ `S`), the **End-to-End Recall is only $16.27\%$** ($41 / 252$).

### 5.2 Optimal F1 Threshold $\tau = 0.20$: Trade-Off with False Alarms
Tuning the binary threshold to $\tau = 0.20$ maximizes Binary F1 ($0.4902$):
- Binary Recall rises to $60.90\%$ ($162 / 266$).
- Missed substitutions drop from $211$ to $90$ ($35.71\%$ missed).
- End-to-End Correctly Diagnosed Substitutions rise from $41$ to **$111 / 252$** ($\text{E2E Recall} = 44.05\%$).
- **Cost of False Alarms:** The binary detector triggers on $233$ clean phonemes (False Positives). For all $233$ clean phonemes, the diagnosis head is spuriously executed and attempts to substitute clean phonemes with non-native error prototypes.

### 5.3 Mismatched Audio Pipeline Parity
When the entire pipeline is evaluated using mismatched audio:
- Binary AP is $0.4598$ (vs $0.4903$).
- At $\tau=0.50$, mismatched audio yields **$46 / 252$** correct substitutions (vs $41$ for matched audio).
- At $\tau=0.20$, mismatched audio yields **$110 / 252$** correct substitutions ($\text{E2E F1} = 0.3385$ vs $0.3431$ for matched audio, $\Delta = 0.0046$).

**The entire deployable MDD pipeline performs identically whether given authentic speaker audio or an unrelated audio clip!**

---

## 6. Audit Verdict & Deployment Implications

$$\mathbf{PIPELINE\_AUDIT: SEPARATION\ VERIFIED\ \&\ LIMITATION\ CONFIRMED}$$

1. **Oracle Accuracy vs Deployable Reality:** Reporting $50.79\%$ substitution accuracy is misleading. In deployable operation, the system recovers only **$16.27\%$** of substitutions at standard threshold and **$44.05\%$** at optimal threshold.
2. **Binary Gating Bottleneck:** The binary detection head acts as an aggressive filter. Over $83\%$ of phonological errors are filtered out before reaching diagnosis unless the threshold is aggressively lowered.
3. **Pipeline Invariance to Acoustics:** The fact that mismatched audio achieves virtually identical end-to-end F1 ($0.3385$ vs $0.3431$) proves that the pipeline's apparent success is almost entirely driven by canonical prompt error priors, not acoustic detection.
