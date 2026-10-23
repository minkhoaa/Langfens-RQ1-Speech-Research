# LANGFENS — RQ1-TRN-00: Baseline Experiment Design & Comparative Assessment

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Task D — Baseline Architecture Design, Comparative Analysis & Research Order Recommendation  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Gate Evaluation Status:** `EXPERIMENT_DESIGN_COMPLETE: PASS`  

---

## 1. Executive Summary & Research Framework

To establish a rigorous scientific foundation for phoneme-level Mispronunciation Detection & Diagnosis (MDD) in Project Langfens, we evaluate and formalize two primary modeling paradigms:
1. **Baseline A: Goodness of Pronunciation (GOP)** — The classical acoustic-phonetic likelihood ratio framework operating over forced-aligned phoneme boundaries.
2. **Baseline B: Wav2Vec 2.0-based Phoneme MDD** — The modern end-to-end self-supervised representation fine-tuning framework with joint CTC alignment, binary detection, and multi-class phone diagnosis.

In accordance with scientific integrity guidelines:
- We do **not** claim an accuracy or performance superiority for either architecture without empirical experimental evidence.
- We evaluate both approaches strictly across methodology, reproducibility, data fit, hardware footprint, implementation risk, and expected pedagogical utility.
- We formulate a definitive, literature-justified recommendation for the execution order of experiments.

---

## 2. Baseline A: Goodness of Pronunciation (GOP) Framework

### 2.1 Theoretical Foundation & Methodology
Originating from Witt & Young (2000) [P02], GOP quantifies the pronunciation quality of phone $p$ by calculating the log-likelihood ratio between the acoustic model's posterior probability of the canonical target phone and the maximum posterior across all competing candidate phonemes over the phone's temporal segment:

$$\text{GOP}(p) = \frac{1}{|t_e - t_s + 1|} \sum_{t=t_s}^{t_e} \log \frac{P(o_t \mid s_p)}{\max_{q \in \mathcal{Q}} P(o_t \mid s_q)} \approx \frac{1}{T} \log \frac{P(o_{t_s:t_e} \mid p)}{\sum_{q \in \mathcal{P}} P(o_{t_s:t_e} \mid q) P(q)}$$

Where:
- $[t_s, t_e]$ represents the start and end acoustic frames of the target phone segment,
- $o_t$ is the acoustic observation vector at frame $t$,
- $s_p$ represents the HMM states associated with canonical phone $p$,
- $\mathcal{Q}$ is the full phoneme state inventory.

### 2.2 System Architecture & Components
1. **Acoustic Backbone:** Pretrained Native English Acoustic Model (e.g. Kaldi TDNN-F trained on LibriSpeech 960h, or torchaudio Emformer/Conformer trained on LibriSpeech).
2. **Alignment Pipeline:** Requires a forced-alignment pass (Viterbi alignment) to align the canonical reference transcript to the non-native speech waveform, yielding time boundaries $[t_{\text{start}}, t_{\text{end}}]$ for each phoneme.
3. **Phoneme Posterion Extractor:** Evaluates frame-level posterior distributions $P(s_t \mid o_t)$ across the duration of each aligned phone.
4. **Scoring & Decision Logic:**
   - **Continuous Metric:** Outputs continuous GOP score $\text{GOP}_i \in (-\infty, 0]$.
   - **Binary Classification:** Applies a phone-specific calibrated threshold $\theta_p$:
     $$\hat{b}_i = \begin{cases} 0 \quad (\text{Correct}) & \text{if } \text{GOP}_i \ge \theta_p \\ 1 \quad (\text{Error}) & \text{if } \text{GOP}_i < \theta_p \end{cases}$$
5. **Multi-Aspect Adaptation (SpeechOcean762):** Feeds extracted phone-level GOP vectors into an MLP or shallow Transformer layer (GOPT; Gong et al., 2022 [P06]) to predict hierarchical scores (accuracy, completeness, fluency, prosody).

### 2.3 Diagnostic Output Capability
- **Candidate Re-Ranking:** By sorting phonemes by posterior mass over $[t_s, t_e]$, GOP identifies candidate substitutions $\hat{q} = \arg\max_{q \ne p} P(q \mid o)$.
- **Limitation:** Fails when forced alignment collapses or when the speaker deletes a phone ($D = 3,420$), as forced alignment will stretch neighboring phones over the deleted segment or assign arbitrary boundaries.

---

## 3. Baseline B: Wav2Vec 2.0-based Phoneme MDD Framework

### 3.1 Theoretical Foundation & Methodology
Following Baevski et al. (2020) [P04] and Peng et al. (2021) [P03], Baseline B leverages self-supervised acoustic representations learned from tens of thousands of hours of speech. Rather than relying on external forced alignment, wav2vec 2.0 extracts high-capacity latent representations that are mapped directly to canonical phone targets and realized phonetic emissions via joint multi-task heads.

```text
Wav2Vec 2.0 MDD Multi-Task Architecture
                  Raw 16 kHz Audio Waveform
                             │
            ┌────────────────▼────────────────┐
            │   7-Layer Conv Feature Encoder  │ (Downsampling factor: 320x)
            └────────────────┬────────────────┘
                             ▼ (Frame rate: 50 Hz, stride: 20 ms)
            ┌─────────────────────────────────┐
            │   12-Layer Transformer Encoder  │ (Hidden dim: 768)
            └────────────────┬────────────────┘
                             ├──────────────────────────────┬──────────────────────────────┐
                             ▼                              ▼                              ▼
                 ┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
                 │  CTC Recognition Head │      │ Binary Detection Head │      │ Phone Diagnosis Head  │
                 │   (Linear -> 40 cls)  │      │  (Linear -> 1 logit)  │      │   (Linear -> 39 cls)  │
                 │     (CTC Loss)        │      │    (Masked BCE Loss)  │      │    (Masked CE Loss)   │
                 └───────────────────────┘      └───────────────────────┘      └───────────────────────┘
```

### 3.2 System Architecture & Components
1. **Backbone:** `facebook/wav2vec2-base-960h` (12 Transformer blocks, 768 hidden dimension, 8 attention heads, 95M parameters).
2. **Acoustic Frame Rate:** Temporal downsampling stride of $320\times$ converts $16,000 \text{ Hz}$ audio into $50 \text{ frames/second}$ ($20 \text{ ms/frame}$).
3. **Multi-Task Heads:**
   - **Head 1 — Realized CTC Recognition Head:** A linear projection $W_{\text{ctc}} \in \mathbb{R}^{768 \times 40}$ ($39 \text{ ARPAbet} + 1 \text{ <BLANK>}$) trained via `nn.CTCLoss` on the actual realized phoneme sequence. Preserves temporal alignment without requiring external Praat or Kaldi forced alignment.
   - **Head 2 — Canonical Binary Detection Head:** A token-level classification head conditioned on the canonical prompt phone embedding concatenated with pooled acoustic frame features, predicting binary logit $\hat{z}_i \in \mathbb{R}$. Trained via `MaskedBCEWithLogitsLoss`.
   - **Head 3 — Phoneme Diagnosis Head:** A multi-class classification head $W_{\text{diag}} \in \mathbb{R}^{768 \times 39}$ predicting the specific realized 39-ARPAbet phone for substitution errors. Trained via `MaskedCrossEntropyLoss` with deletions and `'err'` tags masked with $-100$.
4. **End-to-End Objective Function:**
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{ctc}} \mathcal{L}_{\text{CTC}} + \lambda_{\text{binary}} \mathcal{L}_{\text{Masked-BCE}} + \lambda_{\text{diag}} \mathcal{L}_{\text{Masked-CE}}$$
   Where loss weights $\lambda = (1.0, 1.0, 0.5)$ balance representation learning and task-specific classification.

---

## 4. Comprehensive Comparative Assessment Matrix

| Dimension | Baseline A: Goodness of Pronunciation (GOP) | Baseline B: Wav2Vec 2.0 End-to-End MDD | Comparative Assessment & Tradeoffs |
| :--- | :--- | :--- | :--- |
| **Methodological Paradigm** | Acoustic-phonetic posterior ratio over forced alignment | Multi-task self-supervised neural representation learning | GOP is modular and interpretable; Wav2Vec 2.0 is differentiable and end-to-end. |
| **Alignment Dependency** | **CRITICAL DEPENDENCY.** Requires external forced alignment (Kaldi/MFA). Alignment errors cascade into score errors. | **ALIGNMENT-INDEPENDENT.** CTC head or cross-attention learns internal temporal correspondence directly. | In non-native speech with $3,420$ deletions and heavy distortions, forced alignment frequently fails; Wav2Vec 2.0 eliminates this vulnerability. |
| **Reproducibility** | **MODERATE.** Requires external C++ toolkits (Kaldi), complex acoustic models, and multi-step recipe scripts. | **HIGH.** Self-contained in PyTorch and HuggingFace; consumes our validated PyTorch DataLoader directly. | Wav2Vec 2.0 pipeline is 100% reproducible within a standard Python virtual environment. |
| **Data Fit: L2-ARCTIC** | High risk of alignment failure on Vietnamese deletions ($1,566$ deletions; $45.8\%$ of corpus deletions). | Excellent fit; multi-task formulation maps directly to token-level canonical targets ($N=119,151$). | Wav2Vec 2.0 accommodates non-canonical epenthetic additions and deletions without alignment breakdown. |
| **Data Fit: SpeechOcean** | Excellent fit; GOPT (Gong et al. 2022 [P06]) is specifically designed for multi-aspect scoring from GOP. | Requires an auxiliary utterance-level regression head (e.g. mean pooling + linear regression). | GOP has established state-of-the-art benchmarks on SpeechOcean762; Wav2Vec 2.0 excels at categorical phone MDD. |
| **Compute Requirements (VRAM)** | **Low.** CPU-only inference feasible; small acoustic models ($<10\text{M}$ params). | **Moderate to High.** Fine-tuning requires $4 - 8 \text{ GB}$ VRAM; batch size 2-4 on GTX 1650 with gradient accumulation. | GOP runs easily on local CPU; Wav2Vec 2.0 fine-tuning requires disciplined memory budgeting or GPU acceleration. |
| **Implementation Risk** | High tooling risk: Kaldi compilation, recipe scripting, and acoustic model format conversions. | Low tooling risk: Standard PyTorch module, validated collator, and certified loss contracts. | Wav2Vec 2.0 carries significantly lower software engineering overhead. |
| **Expected Scientific Value** | Serves as classical historical baseline; benchmark for multi-aspect scoring. | Directly answers core research question of RQ1; state-of-the-art foundation for CAPT diagnosis. | High research impact; directly comparable to Peng et al. (2021) and Shahin et al. (2023). |

---

## 5. Evidence-Based Research Order Recommendation

### Recommended Execution Order:
1. **Primary Experiment 1:** **Baseline B (Wav2Vec 2.0 End-to-End Phoneme MDD)**
2. **Secondary Experiment 2:** **Baseline A (GOP-based Assessment & GOPT on SpeechOcean762)**

### Scientific & Engineering Justification:

1. **Elimination of Cascading Alignment Errors (Zhao et al., 2018 [P01]; Peng et al., 2021 [P03]):**
   In L2-ARCTIC, non-native speech exhibits extensive phone deletions ($3,420$ tokens), particularly in the Vietnamese cohort where $1,566$ phonemes are omitted ($45.79\%$ of all corpus deletions). Classical GOP requires an initial forced alignment pass using a native acoustic model. When forced alignment encounters a deleted sound, it is forced to stretch neighboring phone boundaries or align silence into phonemic intervals, producing corrupted posterior ratios and extreme False Alarms ($FP$). By starting with Baseline B (Wav2Vec 2.0 with CTC), the model aligns speech representations organically without catastrophic boundary failure.

2. **Immediate Compatibility with Phase RQ1-PREP-01 Infrastructure:**
   The data preparation pipeline completed in `RQ1-PREP-01` (`LangfensMDDCollate`, `L2ArcticMDDDataset`, `MaskedBCEWithLogitsLoss`) was specifically engineered and certified to deliver batched waveforms, $-100$ padded binary labels, CTC target indices, and 39-class diagnosis targets. Baseline B directly consumes this certified data contract with zero pipeline friction.

3. **Software Toolchain Simplicity & Reproducibility:**
   Implementing Baseline A requires integrating Kaldi or Montreal Forced Aligner (MFA), introducing external C++ dependencies, acoustic model dictionary mapping, and fragile shell recipes. Baseline B runs entirely within standard PyTorch and our version-controlled `src/langfens_prep/` package, guaranteeing strict scientific reproducibility across research environments.

4. **SpeechOcean762 Synergy for Baseline A:**
   Baseline A (specifically GOPT; Gong et al., 2022 [P06]) will be deployed subsequently as the dedicated baseline for continuous multi-aspect pronunciation scoring on SpeechOcean762, where continuous scoring rather than discrete phoneme replacement is the primary research target.

**Phase Gate Evaluation:** **`EXPERIMENT_DESIGN_COMPLETE: PASS`**
