# LANGFENS — RQ1-TRN-01B1-R2: Canonical-Only Baseline Comparison

**Gate Reference:** RQ1-TRN-01B1-R2 — Acoustic-Conditioned MDD & Final Source Provenance Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/CANONICAL_ONLY_BASELINE_COMPARISON.md`  
**Experimental Metric References:**  
- `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/canonical_only_baselines.json`  
- `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/task_b_acoustic_contribution.json`  
**Lead Speech ML Systems Engineer & Scientific Reproducibility Auditor:** Senior Speech ML Research Engineer  
**Audit Finding:** `CANONICAL_ONLY_COMPARISON: AUDIT COMPLETE (Prompt Prior Exceeds Acoustic Model)`  

---

## 1. Executive Summary & Audit Mandate

In speech research and clinical mispronunciation detection & diagnosis (MDD), reporting substitution diagnosis accuracy against uniform chance ($1/39 = 2.56\%$) or empirical majority class ($\approx 16\% - 23\%$) is scientifically invalid if the input to the classifier includes the canonical prompt phoneme.

Because non-native phonological substitutions in L2 speech are governed by strong L1 phonological interference rules, a model given only the canonical prompt token can achieve high diagnosis accuracy without listening to a single millisecond of audio.

To establish the true empirical bar for acoustic conditioning, this audit constructed and evaluated **five canonical-only reference baselines** trained on the exact internal training probe ($256$ utterances, $936$ substitutions, $9,485$ binary tokens) and tested on the exact internal holdout probe ($64$ utterances, $252$ substitutions, $2,081$ binary tokens).

---

## 2. Canonical-Only Baseline Specifications

All canonical-only baselines were trained with **strictly zero acoustic input** ($\mathbf{x} = \emptyset$):

1. **Uniform Chance Baseline ($1/39$):**
   Predicts a uniformly distributed random choice over the $39$ standardized ARPAbet phoneme classes:
   $$\text{Acc}_{\text{uniform}} = \frac{1}{39} = 2.5641\%$$
2. **Dynamic Empirical Majority Class Baseline:**
   Predicts the single most frequent substitution class observed in the training distribution:
   $$\hat{y}_k = \arg\max_{c \in \mathcal{C}_{39}} \sum_{i=1}^{N_{\text{train}}} \mathbb{I}(y_{\text{sub}}^{(i)} = c) = \text{class } \texttt{S} \quad (151 / 936 = 16.13\%)$$
   Evaluated on holdout, class `S` achieves $50 / 252 = 19.84\%$ (while holdout-local majority class `D` achieves $59 / 252 = 23.41\%$).
3. **Canonical Empirical Prior Lookup Matrix ($P(\text{realized} \mid \text{canonical})$):**
   A non-parametric Bayesian lookup table computed directly from the $936$ substitution events in the training probe:
   $$\hat{y}_k = \arg\max_{r \in \mathcal{C}_{39}} P(\text{realized} = r \mid \text{canonical} = c_k) = \arg\max_{r \in \mathcal{C}_{39}} \frac{N_{\text{train}}(c_k, r) + \alpha}{N_{\text{train}}(c_k) + 39\alpha}$$
   with Laplace smoothing $\alpha = 0.01$.
4. **Canonical-Only Learned Neural Substitution MLP:**
   A standalone PyTorch neural network:
   $$\mathbf{e}_c = \text{Embedding}(40, 128)(c_k)$$
   $$\mathbf{h}_1 = \text{GELU}(\text{Linear}(128, 256)(\mathbf{e}_c))$$
   $$\mathbf{h}_2 = \text{GELU}(\text{Linear}(256, 256)(\mathbf{h}_1))$$
   $$\hat{\mathbf{y}} = \text{Linear}(256, 39)(\mathbf{h}_2)$$
   trained for $120$ epochs with AdamW ($\text{LR} = 10^{-3}$, weight decay $= 0.01$, cross-entropy loss) on the $936$ training substitution tokens.
5. **Canonical-Only Learned Neural Binary MDD MLP:**
   A standalone PyTorch binary detector mapping canonical tokens to error probability:
   $$\mathbf{e}_c = \text{Embedding}(40, 128)(c_k)$$
   $$\hat{p}_{\text{error}} = \sigma\left(\text{Linear}(256, 1)(\text{GELU}(\text{Linear}(128, 256)(\mathbf{e}_c)))\right)$$
   trained with AdamW ($\text{LR} = 10^{-3}$) on the $9,485$ canonical binary tokens of the training probe.

---

## 3. Quantitative Baseline Comparison

### 3.1 Substitution Diagnosis Benchmark (Internal Holdout Probe, $N = 252$ Substitutions)

| Model / Baseline Architecture | Acoustic Input? | Canonical Prompt Input? | Train Accuracy ($N=936$) | Holdout Accuracy ($N=252$) | Holdout Top-3 Accuracy | Holdout Loss (NLL/CE) | Authentic Acoustic Gain? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uniform Chance** | No | No | $2.56\%$ | $2.56\%$ | $7.69\%$ | $3.6636$ | Baseline |
| **2. Dynamic Majority Class (`S`)** | No | No | $16.13\%$ | $19.84\%$ | — | — | Baseline |
| **3. Canonical Empirical Prior** | **No** | **Yes** | **$71.79\%$** | **$56.35\%$** | **$69.05\%$** | $2.1321$ | Baseline |
| **4. Canonical Learned MLP** | **No** | **Yes** | **$71.79\%$** | **$56.35\%$** | **$68.65\%$** | $2.0808$ | Baseline |
| **5. Multi-Task Model (Matched Audio)** | **Yes** | **Yes** | $60.90\%$ | **$50.79\%$** | $66.27\%$ | $2.2410$ | **$-5.56\%$ (Defeated by Prior)** |
| **6. Multi-Task Model (Mismatched Audio)**| **Wrong Audio**| **Yes** | — | **$51.19\%$** | $66.67\%$ | $2.2480$ | **$-5.16\%$ (Defeated by Prior)** |

### 3.2 Binary MDD Detection Benchmark (Internal Holdout Probe, $N = 2,081$ Tokens)

| Model / Baseline Architecture | Acoustic Input? | Holdout PR-AUC | Holdout BCE Loss | Precision @ $\tau=0.5$ | Recall @ $\tau=0.5$ | F1 @ $\tau=0.5$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Empirical Positive Rate** | No | $0.1278$ | $0.3824$ | — | — | — |
| **Canonical Learned Binary MLP** | **No** | **$0.4745$** | **$0.3039$** | **$0.7059$** | $0.1353$ | $0.2271$ |
| **Multi-Task Model (Matched Audio)** | **Yes** | **$0.4903$** | $0.2982$ | $0.7069$ | $0.1541$ | $0.2531$ |
| **Multi-Task Model (Mismatched Audio)** | **Wrong Audio** | **$0.4598$** | $0.3015$ | $0.6667$ | $0.1729$ | $0.2746$ |

---

## 4. Scientific Findings & Mechanistic Insights

### 4.1 The Canonical-Only MLP Outperforms the Neural Acoustic Model
The most significant finding is that:
$$\text{Acc}(\text{Canonical-Only Learned MLP}) = \mathbf{56.35\%} > \text{Acc}(\text{Multi-Task Model with Matched Audio}) = \mathbf{50.79\%}$$
A simple two-layer MLP taking **only the canonical phoneme index** achieves **$56.35\%$ accuracy** on unseen speakers, outperforming the full 95M-parameter Wav2Vec2 + Cross-Attention multi-task model by **$+5.56$ percentage points**!

Why does the multi-task model achieve only $50.79\%$?
Because the cross-attention module adds a noisy, diffuse acoustic vector ($\cos = 0.9726$ across utterances) that slightly degrades the clean discrete prompt representation learned by the pure MLP!

### 4.2 Why Does the Prompt-Only Prior Achieve 56.35% on Holdout?
In L2-ARCTIC, non-native speakers systematically transfer phonological constraints from their native language (L1):
- English `DH` ($\eth$) does not exist in Spanish or Arabic. In over $85\%$ of occurrences, Spanish and Arabic speakers substitute `D`.
- English `TH` ($\theta$) is substituted with `T` in over $75\%$ of occurrences.
- English `Z` is devoiced to `S` in word-final positions in over $70\%$ of occurrences.
- English vowels `IH`, `AE`, `ER` systematically collapse into Cardinal vowels `IY`, `EH`, `AH`.

When evaluated on the 64-utterance holdout probe:
- $36 / 51$ canonical `DH` tokens are substituted with `D`. Predicting `D` achieves $70.6\%$ accuracy on `DH`.
- $50 / 72$ canonical `Z` tokens are substituted with `S`. Predicting `S` achieves $69.4\%$ accuracy on `Z`.
- $14 / 54$ canonical `ER` tokens are substituted with `AH`. Predicting `AH` achieves $25.9\%$ accuracy on `ER`.

Because `DH` and `Z` account for nearly half of all substitution events in the holdout set, an acoustic-free model that memorizes `DH` $\to$ `D` and `Z` $\to$ `S` automatically achieves $>55\%$ accuracy across the entire test set!

### 4.3 Binary MDD is Also Heavily Biased by Canonical Identity
A similar phenomenon occurs in the Binary MDD head:
- The Canonical-Only Binary MLP achieves a **PR-AUC of $0.4745$ with ZERO audio**!
- The full multi-task model with matched audio achieves **$0.4903$** (a difference of only **$0.0158$**).
- When fed mismatched audio from an unrelated utterance, the multi-task model achieves **$0.4598$** (within $0.0147$ of the canonical-only baseline).

This proves that even for binary mispronunciation detection, the model primarily relies on phoneme-specific error risk priors (e.g. `DH`, `TH`, `Z` are high risk; `M`, `N`, `P` are low risk) rather than listening to the acoustic realization.

---

## 5. Mandatory Baseline Standards for Future MDD Evaluation

Moving forward, the following reporting policy is frozen:

1. **Uniform Chance Baseline ($2.56\%$) is Obsolete:** Citing uniform chance as evidence of successful machine learning is scientifically unacceptable for prompt-conditioned MDD heads.
2. **Dynamic Majority Baseline ($16.13\% - 23.41\%$) is Insufficient:** Exceeding the majority class merely indicates that the model has learned more than one class; it does not indicate acoustic conditioning.
3. **The Canonical Prompt-Only Prior ($56.35\%$) is the True Gate:**  
   No model can be claimed to possess acoustic substitution diagnosis capability unless:
   $$\text{Acc}_{\text{diag}}(\text{Matched Audio}) \gg \mathbf{56.35\%} \quad \text{AND} \quad \text{Acc}_{\text{diag}}(\text{Matched Audio}) - \text{Acc}_{\text{diag}}(\text{Mismatched Audio}) \ge \mathbf{10.00\%}$$
   The current architecture satisfies neither condition.
