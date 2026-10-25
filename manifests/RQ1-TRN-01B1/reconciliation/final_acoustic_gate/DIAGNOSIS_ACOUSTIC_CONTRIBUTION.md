# LANGFENS — RQ1-TRN-01B1-R2: Diagnosis Acoustic Contribution Audit

**Gate Reference:** RQ1-TRN-01B1-R2 — Acoustic-Conditioned MDD & Final Source Provenance Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md`  
**Experimental Metric References:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/task_b_acoustic_contribution.json`  
**Lead Speech ML Systems Engineer & Scientific Auditor:** Senior Speech ML Research Engineer  
**Audit Finding:** `ACOUSTIC_CONTRIBUTION: INSUFFICIENT / DEFEATED BY PROMPT PRIOR`  

---

## 1. Executive Summary & Audit Mandate

In the RQ1-TRN-01B1 reconciliation campaign, substitution diagnosis accuracy on the 2-speaker internal holdout probe (64 utterances, speakers `SVBI` and `ZHAA`, 252 eligible substitutions) exhibited an alarming anomaly:
- **Matched Audio Accuracy:** $50.79\%$ ($128 / 252$)
- **Mismatched Audio Accuracy:** $51.19\%$ ($129 / 252$)
- **Empirical Prompt-Only Prior Baseline:** $56.35\%$ ($142 / 252$)

The independent review correctly stated:  
*"This is insufficient evidence for acoustic substitution diagnosis. Do not assume matched-versus-mismatched accuracy alone proves complete acoustic independence."*

In accordance with Task B instructions, this audit conducts an exhaustive token-level investigation into prediction agreement, logit distances, distribution entropy, correct-class score distributions, cross-attention alignment pathways, representation invariance, and a controlled spoken-phone sensitivity census where the canonical phone is held constant while the actual spoken phone changes.

---

## 2. Multi-Level Perturbation Battery (Evaluation-Only)

All perturbations were executed strictly at evaluation time. The model was never trained on perturbed or held-out data, and inference never accessed gold timestamps or realized labels.

| Evaluation Condition | Waveform Input $\mathbf{x}$ | Sub. Diag Acc ($N=252$) | Correct / Total | Binary PR-AUC ($N=2081$) | Mean Entropy (nats) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Matched Baseline** | Valid audio $\mathbf{x}^{(i)}$ | **$50.79\%$** | $128 / 252$ | **$0.4903$** | **$1.8509$** |
| **2. Mismatched Audio** | Swapped audio $\mathbf{x}^{(j)}$ ($j \ne i$) | **$51.19\%$** | $129 / 252$ | **$0.4598$** | **$1.8740$** |
| **3. Silence Control** | All zeros $\mathbf{x} = \mathbf{0}$ | **$2.38\%$** | $6 / 252$ | **$0.3834$** | **$3.4155$** |
| **4. Shuffled Frames** | Temporal permuted $\mathbf{x}$ | **$2.38\%$** | $6 / 252$ | **$0.3903$** | **$3.2867$** |
| **5. Gaussian Noise** | Matched RMS noise | **$1.19\%$** | $3 / 252$ | **$0.3628$** | **$3.3431$** |

### Key Observations:
1. **The Invariance Paradox:** Swapping speech audio from an unrelated speaker and utterance produces virtually zero change in accuracy ($50.79\% \to 51.19\%$, $\Delta = -0.40\%$) and virtually identical prediction entropy ($1.8509 \to 1.8740\text{ nats}$).
2. **Noise vs Silence Collapse:** When audio is replaced by silence or noise, accuracy collapses to uniform chance ($1.19\% - 2.38\%$, chance is $1/39 = 2.56\%$), and entropy doubles ($3.28 - 3.42\text{ nats}$, theoretical max is $\ln(39) = 3.6636\text{ nats}$). This collapse occurs not because the model listens to the phonetic contents of audio, but because Wav2Vec2 representations under silence/noise collapse out of the activation manifold expected by the LayerNorm layers.

---

## 3. Token-Level Matched vs Mismatched Agreement & Distances

To go beyond aggregate accuracy, we compared predictions and logit distributions at the **exact same canonical token positions** across all 64 holdout utterances.

### 3.1 Per-Token Prediction Agreement
- **All Canonical Tokens ($N = 2,081$):**
  - **Top-1 Exact Prediction Agreement:** $\mathbf{89.43\%}$ ($1,861 / 2,081$)
  - **Top-3 Overlap Rate:** $\mathbf{99.42\%}$ ($2,069 / 2,081$)
  - **Top-3 Identical Set Rate:** $\mathbf{93.99\%}$ ($1,956 / 2,081$)
- **Substitution Tokens ($N = 252$):**
  - **Top-1 Exact Prediction Agreement:** $\mathbf{93.25\%}$ ($235 / 252$)

**$93.25\%$ of substitution tokens produce the exact same predicted phone regardless of whether the audio is the actual speaker or an unrelated sentence!**

### 3.2 Logit Vector Distances ($C = 39$ Classes)

| Metric | Mean | Std | Median | p25 | p75 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | $0.3211$ | $0.2255$ | $0.2587$ | $0.1671$ | $0.4125$ | $1.6873$ |
| **$L_2$ Euclidean Distance** | $2.4042$ | $1.6705$ | $1.9616$ | $1.2583$ | $3.0768$ | $11.5173$ |
| **Cosine Similarity** | $\mathbf{0.9934}$ | $0.0106$ | $\mathbf{0.9967}$ | $0.9926$ | $0.9985$ | $1.0000$ |
| **Jensen-Shannon Divergence** | $0.0121$ | $0.0250$ | $0.0049$ | $0.0017$ | $0.0119$ | $0.4118$ |

The mean cosine similarity between the 39-dimensional diagnosis logit vectors under matched vs mismatched audio is **$0.9934$** (median **$0.9967$**), and median JS divergence is **$0.0049\text{ nats}$**. The logit distribution is virtually identical across completely different audio inputs.

---

## 4. Correct-Class Substitution Score Differences

For each of the $252$ ground-truth substitution tokens with true realized phoneme $y \in \{0, \dots, 38\}$, we computed:
$$\Delta P(y) = P_{\text{matched}}(y) - P_{\text{mismatched}}(y)$$
$$\Delta z(y) = z_{\text{matched}}[y] - z_{\text{mismatched}}[y]$$

### 4.1 Statistical Distribution:
- **Mean $\Delta P(y)$:** $\mathbf{+0.0009} \pm 0.0491$
- **Median $\Delta P(y)$:** $\mathbf{-0.0001}$
- **Interquartile Range (IQR):** $[-0.0039, +0.0036]$
- **Tokens where Matched Audio gave strictly higher probability to gold ($P_{\text{match}} > P_{\text{mis}}$):** $124 / 252$ ($49.21\%$)
- **Tokens where Mismatched Audio gave strictly higher probability to gold ($P_{\text{mis}} > P_{\text{match}}$):** $\mathbf{128 / 252}$ ($\mathbf{50.79\%}$)
- **Tied Tokens ($P_{\text{match}} == P_{\text{mis}}$):** $0 / 252$

**Mismatched audio assigned a higher probability to the true spoken phoneme on MORE tokens ($128$) than matched audio did ($124$)!**  
The mean probability difference of $+0.0009$ is statistically indistinguishable from zero ($p = 0.77$, two-tailed paired $t$-test).

---

## 5. Cross-Attention Pathway & Representation Invariance Audit

Why does the cross-attention mechanism fail to condition on acoustics?

### 5.1 Formulation
In `CanonicalAcousticCrossAttention` (`src/langfens_prep/model.py`):
$$\mathbf{Q} = W_p \mathbf{e}_{\text{canonical}} + \mathbf{e}_{\text{pos}} \in \mathbb{R}^{B \times L \times 768}$$
$$\mathbf{K} = \mathbf{V} = \mathbf{H}_{\text{acoustic}} \in \mathbb{R}^{B \times T \times 768}$$
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{768}}\right) \in \mathbb{R}^{B \times L \times T}$$
$$\mathbf{h}_{\text{aligned}} = \text{LayerNorm}\left( \text{Dropout}\left(\mathbf{A} \mathbf{V}\right) \right) \in \mathbb{R}^{B \times L \times 768}$$
$$\hat{\mathbf{y}}_{\text{diag}} = \text{DiagnosisHead}(\mathbf{h}_{\text{aligned}}) \in \mathbb{R}^{B \times L \times 39}$$

### 5.2 Attention Weight Distribution
- **Normalized Attention Entropy over Acoustic Frames ($H / H_{\max}$):**
  $$\text{Mean} = \mathbf{0.6513} \pm 0.1231 \quad (\text{Median} = 0.6519, \text{p25} = 0.5671, \text{p75} = 0.7404)$$
  The attention weights are diffuse across acoustic frames rather than sharply peaked at phoneme boundaries.

### 5.3 Invariance of Aligned Acoustic Representation $\mathbf{h}_{\text{aligned}}$
We extracted $\mathbf{h}_{\text{aligned}}$ directly from the forward hook under matched audio $\mathbf{x}_{\text{matched}}$ and mismatched audio $\mathbf{x}_{\text{mismatched}}$ for identical canonical prompts:
- **Cosine Similarity of $\mathbf{h}_{\text{aligned}}$:** $\text{Mean} = \mathbf{0.9726} \pm 0.0346$ ($\text{Median} = \mathbf{0.9836}$, $\text{p75} = 0.9914$)
- **Relative $L_2$ Difference ($\|\mathbf{h}_m - \mathbf{h}_{\text{mis}}\| / \|\mathbf{h}_m\|$):** $\text{Mean} = 0.2063 \pm 0.1111$ ($\text{Median} = 0.1813$)

Because the backbone is frozen and the cross-attention query $\mathbf{Q}$ is determined by the canonical prompt phoneme and position, $\mathbf{A} \mathbf{V}$ computes a weighted average of speech frames that collapses into an almost identical query-dominated representation ($\cos = 0.9726$). The subsequent MLP diagnosis head decodes this representation back into the empirical training distribution of non-native substitutions.

---

## 6. Controlled Spoken-Phone Sensitivity Census (Acoustic Sensitivity Index)

To settle definitively whether diagnosis outputs respond when the actual spoken phone changes, we conducted a **controlled paired census**:
- Hold canonical phone $C$ constant.
- Compare instances where Speaker A pronounced $R_A$ (e.g., correct $C$) versus instances where Speaker B pronounced $R_B \ne R_A$ (e.g., substituted phoneme).
- Measure the **Acoustic Sensitivity Index (ASI)**:
  $$\text{ASI}(C, R_B) = \mathbb{E}\left[ P(R_B \mid \mathbf{x}_B, C) - P(R_B \mid \mathbf{x}_A, C) \right]$$
  If the model listens to audio, $\text{ASI} > 0$ (the predicted probability of $R_B$ must rise when Audio B contains $R_B$). If the model relies only on the canonical prompt, $\text{ASI} \approx 0$.

### 6.1 Empirical Census Results Across 23 Phonemes ($N = 202$ Distinct Phone Pairs)
- **Overall Acoustic Sensitivity Index:** $\mathbf{ASI = +0.003461}$ ($< 0.35\%$ probability shift)

### 6.2 Case Study: Canonical `DH` ($N=51$ occurrences in holdout)
- Correct realization ($R = \text{DH}$): $15$ occurrences
- Substitution realization ($R = \text{D}$): $36$ occurrences
- **When speaker actually said `DH` (correct):**
  $$P(\text{pred } D) = \mathbf{0.9967}$$
- **When speaker actually said `D` (substitution):**
  $$P(\text{pred } D) = \mathbf{0.9965}$$
- **Acoustic Sensitivity:** $\Delta P = \mathbf{-0.0002}$!

### 6.3 Case Study: Canonical `Z` ($N=72$ occurrences in holdout)
- Correct realization ($R = \text{Z}$): $22$ occurrences
- Substitution realization ($R = \text{S}$): $50$ occurrences
- **When speaker actually said `Z` (correct):**
  $$P(\text{pred } S) = \mathbf{0.9850}$$
- **When speaker actually said `S` (substitution):**
  $$P(\text{pred } S) = \mathbf{0.9861}$$
- **Acoustic Sensitivity:** $\Delta P = \mathbf{+0.0011}$!

### 6.4 Case Study: Canonical `ER` ($N=54$ occurrences in holdout)
- Correct realization ($R = \text{ER}$): $40$ occurrences
- Substitution realization ($R = \text{AH}$): $14$ occurrences
- **When speaker actually said `ER` (correct):**
  $$P(\text{pred } AH) = \mathbf{0.2592}$$
- **When speaker actually said `AH` (substitution):**
  $$P(\text{pred } AH) = \mathbf{0.2596}$$
- **Acoustic Sensitivity:** $\Delta P = \mathbf{+0.0004}$!

**In every single case, the model outputs the exact same substitution probability regardless of whether the speaker pronounced the sound correctly or substituted another phoneme!**

---

## 7. Audit Verdict & Scientific Finding

$$\mathbf{ACOUSTIC\_CONTRIBUTION: DEFEATED\ BY\ PROMPT\ PRIOR}$$

1. **Acoustic Substitution Diagnosis is NOT Functioning:** The substitution diagnosis head does not utilize acoustic evidence to identify realized speech sounds.
2. **Dominance of Non-Native L1 Transfer Prior:** In L2-ARCTIC, non-native substitutions follow deterministic phonetic rules (`DH` $\to$ `D`, `Z` $\to$ `S`, `TH` $\to$ `T`). Because these rules are so consistent, the diagnosis head minimizes cross-entropy loss by memorizing this static prior map.
3. **Representational Decoupling:** In sharp contrast to the diagnosis head, the **CTC phone recognition head exhibits strict, verified acoustic dependence** ($\text{PER} = 0.485 \to 0.947$ on mismatched audio). The failure is localized specifically to the cross-attention alignment and diagnosis head design.
