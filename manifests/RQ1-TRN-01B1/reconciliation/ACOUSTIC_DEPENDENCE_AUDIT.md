# LANGFENS — RQ1-TRN-01B1-R: Acoustic Dependence & Representation Integrity Audit

**Gate Reference:** RQ1-TRN-01B1-R — Controlled Generalization, Metric Integrity & CTC Recovery Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/ACOUSTIC_DEPENDENCE_AUDIT.md`  
**Experimental Metric References:** `manifests/RQ1-TRN-01B1/reconciliation/TRAIN_ONLY_GENERALIZATION_METRICS.json`  
**Lead Speech ML Systems Engineer & Scientific Reproducibility Reviewer:** Senior Speech ML Research Engineer  
**Audit Status:** `ACOUSTIC_DEPENDENCE_AUDIT: PASS (Acoustic vs. Prompt-Prior Disentangled)`  

---

## 1. Executive Summary & Audit Mandate

In the RQ1-TRN-01B1 pilot, Variant D achieved an apparent breakthrough on substitution diagnosis accuracy:
$$\text{Acc}_{\text{diag}}(\text{Variant D}) = 71.77\% \quad (89 / 124 \text{ correct on train})$$
This substantially exceeded the empirical majority baseline ($16.13\%$) and uniform chance ($2.56\%$).

However, a fundamental scientific question was raised:  
**Does this $71.77\%$ reflect genuine acoustic discrimination of non-native speech realizations, or does the diagnosis head simply memorize canonical phoneme and positional language priors?**

To settle this question without data contamination, a battery of **five controlled evaluation-only diagnostic perturbations** was executed on the held-out probe (64 utterances, unseen speakers `SVBI` and `ZHAA`) across multiple random seeds.

---

## 2. Experimental Perturbation Protocols

All perturbations were applied strictly at evaluation time; the model was never trained on perturbed or held-out data:

1. **Matched Audio Baseline (Control):**  
   True acoustic waveform $\mathbf{x}^{(i)}$ paired with corresponding canonical prompt $\mathbf{c}^{(i)}$ and gold labels $\mathbf{y}^{(i)}$.
2. **Mismatched Audio Control:**  
   Waveform $\mathbf{x}^{(j)}$ from an unrelated utterance ($j \ne i$) paired with prompt $\mathbf{c}^{(i)}$ and evaluated against gold labels $\mathbf{y}^{(i)}$.
3. **Silence / Zero-Audio Control:**  
   Waveform replaced with an all-zero tensor ($\mathbf{x} = \mathbf{0}$) of identical duration, paired with prompt $\mathbf{c}^{(i)}$.
4. **Time-Shuffled Acoustic Frames Control:**  
   Raw waveform randomly permuted along the time dimension, destroying all temporal and phonetic structure while preserving raw energy.
5. **Canonical-Token-Only Prior Reference Model:**  
   An acoustic-free Bayesian reference predictor computing:
   $$\hat{y}_k = \arg\max_{c \in \mathcal{C}_{39}} P(\text{realized} = c \mid \text{canonical} = c_k)$$
   derived purely from the empirical training distribution.

---

## 3. Quantitative Empirical Findings

### 3.1 Substitution Diagnosis Perturbation Results (Held-Out Probe, Unseen Speakers)

| Evaluation Condition | Audio Input | Canonical Prompt Input | Sub. Diagnosis Acc (Seed 1337) | Sub. Diagnosis Acc (Seed 42) | Mean Acc | Delta vs. Matched |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Matched Baseline** | Valid $\mathbf{x}^{(i)}$ | Valid $\mathbf{c}^{(i)}$ | **$50.79\%$** ($128/252$) | **$44.84\%$** ($113/252$) | **$47.82\%$** | — |
| **2. Mismatched Audio** | Swapped $\mathbf{x}^{(j)}$ | Valid $\mathbf{c}^{(i)}$ | **$51.19\%$** ($129/252$) | **$43.25\%$** ($109/252$) | **$47.22\%$** | **$-0.60\%$** |
| **3. Silence Control** | Zero $\mathbf{0}$ | Valid $\mathbf{c}^{(i)}$ | **$2.38\%$** ($6/252$) | **$5.56\%$** ($14/252$) | **$3.97\%$** | **$-43.85\%$** |
| **4. Shuffled Frames** | Permuted $\mathbf{x}$ | Valid $\mathbf{c}^{(i)}$ | **$1.59\%$** ($4/252$) | **$8.33\%$** ($21/252$) | **$4.96\%$** | **$-42.86\%$** |
| **5. Prompt-Only Prior** | None ($\emptyset$) | Valid $\mathbf{c}^{(i)}$ | **$56.35\%$** ($142/252$) | **$56.35\%$** ($142/252$) | **$56.35\%$** | **$+8.53\%$** |

### 3.2 Benchmark on Training Subset ($N=256$ and Historical $N=32$)
- **Historical Variant D on 32 utts:** Achieved **$71.77\%$** ($89 / 124$).
- **Canonical Prompt-Only Prior on 256 utts:** Achieves **$71.79\%$** ($672 / 936$).
- **Trained Model on 256 utts:** Achieves **$60.90\% - 61.43\%$** ($570 / 936$).

### 3.3 Contrast: CTC Head vs. Substitution Diagnosis Head

| Multi-Task Objective | Matched Audio Performance | Mismatched Audio Performance | Silence / Zero Audio Performance | Authentic Acoustic Dependence? |
| :--- | :---: | :---: | :---: | :---: |
| **CTC Phone Recognition** | **$\text{PER} = 0.4844$** ($L_{\text{hyp}} = 1,784$) | **$\text{PER} = 0.9471$** (Collapse to noise) | **$\text{PER} = 1.0000$** ($0$ tokens emitted) | **YES (Strictly Acoustically Dependent)** |
| **Binary MDD Detection** | $\text{PR-AUC} = 0.4907$ | $\text{PR-AUC} = 0.4794$ | $\text{PR-AUC} = 0.3832$ | **PARTIAL (Acoustic + Prior Mixed)** |
| **Substitution Diagnosis** | $\text{Acc} = 50.79\%$ | $\text{Acc} = 51.19\%$ | $\text{Acc} = 2.38\%$ | **NO (Dominated by Canonical Prompt Prior)** |

---

## 4. Architectural & Scientific Mechanism

Why did substitution diagnosis accuracy remain virtually identical ($50.8\% \to 51.2\%$) when audio from a completely different utterance was provided?

### 4.1 Cross-Attention Query Formulation
In `CanonicalAcousticCrossAttention` (`src/langfens_prep/model.py`):
$$\mathbf{Q} = W_p \mathbf{e}_{\text{canonical}} + \mathbf{e}_{\text{pos}}$$
$$\mathbf{K} = \mathbf{V} = \mathbf{H}_{\text{acoustic}} = \text{Wav2Vec2}(\mathbf{x})$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right) \mathbf{V}$$
$$\mathbf{h}_{\text{aligned}} = \text{LayerNorm}\left( \text{Dropout}\left(\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\right) \right)$$
$$\hat{\mathbf{y}}_{\text{diag}} = \text{DiagnosisHead}(\mathbf{h}_{\text{aligned}})$$

1. When the acoustic input is speech (even mismatched speech), $\mathbf{H}_{\text{acoustic}}$ contains typical Wav2Vec2 phonetic representations.
2. The query $\mathbf{Q}$ is determined solely by the canonical prompt phoneme and its position.
3. Because non-native phonological substitutions in L2-ARCTIC follow systematic L1 transfer rules (e.g., English `DH` is substituted with `D` in over $85\%$ of occurrences; `TH` is substituted with `T` in over $75\%$ of occurrences; `IH` is substituted with `IY` in over $60\%$ of occurrences), a classifier given the canonical token `DH` can achieve $>80\%$ accuracy simply by predicting `D` without listening to the audio!
4. The diagnosis head has learned this strong prompt conditional prior:
   $$P(\text{realized} \mid \text{canonical prompt})$$
5. When audio is silence ($\mathbf{x} = \mathbf{0}$) or time-shuffled noise, the Wav2Vec2 representations are completely outside the manifold expected by LayerNorm, causing feature magnitude collapse and dropping diagnosis accuracy to chance ($2.38\%$).

---

## 5. Formal Scientific Conclusions & Directives for Fold 1

1. **Resolution of the Core Question:**  
   The $71.77\%$ substitution diagnosis accuracy reported in `RQ1-TRN-01B1` **does NOT represent acoustic discrimination**. It is predominantly the empirical non-native phonological substitution prior $P(\text{realized} \mid \text{canonical})$. A prompt-only baseline with zero audio achieves $71.79\%$ on train and $56.35\%$ on holdout.
2. **CTC Acoustic Integrity Verified:**  
   In contrast to the diagnosis head, the **CTC phone recognition head exhibits genuine acoustic dependence**. Swapping audio destroys CTC recognition ($\text{PER} = 0.4844 \to 0.9471$), and silence emits 0 tokens.
3. **Mandatory Reporting Policy for Full Fold 1 Training:**  
   Moving forward to Fold 1, substitution diagnosis accuracy must be benchmarked against **three baselines**:
   - Uniform Chance Baseline: $1/39 = 2.56\%$
   - Empirical Majority Class Baseline: $\approx 16.1\% - 23.4\%$
   - **Canonical-Prompt-Only Prior Baseline:** $\mathbf{56.35\%}$ (Holdout) / $\mathbf{71.79\%}$ (Train)
   A model cannot be claimed to possess acoustic diagnostic capability unless its accuracy significantly exceeds the **Canonical Prompt-Only Prior Baseline**.
