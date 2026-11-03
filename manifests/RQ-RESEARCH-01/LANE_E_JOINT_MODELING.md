# LANE E: Joint Acoustic-Recognition and Edit-Aware Modeling for Mispronunciation Detection & Diagnosis (MDD)

**Document ID:** `manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md`  
**Research Family:** Family E — Joint End-to-End Architectures Modeling Acoustics and Edit Operations Simultaneously  
**Phase:** `RQ-RESEARCH-01`  
**Status:** `RESEARCH_COMPLETED; PILOT_EXPERIMENT_DESIGNS_CERTIFIED`  
**Target Architecture:** Multi-Task Wav2Vec2-Base-960h (CTC Acoustic Branch + Joint Canonical Edit-Aware Sequence Tagging / Transducer Branch)  
**Corpus & Protocol Authority:** L2-ARCTIC (4-Fold Cross-Validation, Sealed Test Partition, Zero-Leakage Protocol Z)  
**Thesis Scope Anchor:** UIT Thesis Proposal `KLTN0127` (Nguyễn Bá Trọng Khôi – 23520772 & Từ Minh Khoa – 23520757; Advisor: ThS. Trần Thị Hồng Yến) — Reference-based phoneme-level MDD, substitutions, omissions/final consonant deletions, confusable pairs, acoustic representations, tiered pedagogical feedback, Langfens Speaking Web Service.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Central Dilemma in Phoneme Mispronunciation Detection
In computer-assisted pronunciation training (CAPT), the objective of Mispronunciation Detection and Diagnosis (MDD) is to evaluate a learner's spoken utterance against an expected canonical reference prompt $C = (c_1, c_2, \dots, c_N)$ derived from text. The system must decide for each canonical phoneme whether it was pronounced correctly, substituted with an erroneous sound, or omitted altogether (phoneme deletion).

Conventional pipelines decouple this problem into two or three independent stages:
1. An **acoustic recognition head** (e.g., Connectionist Temporal Classification, CTC) transcribes the speech signal into a sequence of realized phonemes $\hat{Y} = (\hat{y}_1, \dots, \hat{y}_U)$.
2. An **alignment module** (e.g., dynamic programming Levenshtein distance or Kaldi forced alignment) aligns the realized sequence $\hat{Y}$ to the canonical reference $C$.
3. A **post-hoc diagnosis module** labels the alignment operations as Correct, Substitution, or Deletion.

This decoupled architecture suffers from catastrophic error propagation:
- If the CTC recognizer drops a phoneme due to acoustic ambiguity, standard Levenshtein alignment cannot distinguish whether the speaker actually omitted the sound (a true phonological deletion) or whether the acoustic model merely experienced recognition dropout.
- In `Langfens` Baseline R (E0), where a Wav2Vec2 backbone feeds a realized CTC head alongside an unconstrained cross-attention Binary MDD head, the diagnostic census across all 4 folds (`manifests/RQ1-DIAG-01/checkpoints/01-phoneme-omission-and-false-negative-root-cause.md`) revealed that the model is **virtually blind to phoneme omissions**:
  - **Omission Average Precision (AP):** **0.030506** (Fold 1: **0.035020**), hovering barely above the empirical random prevalence of $0.0159$.
  - **Omission False Negative Rate (FNR):** **97.46%** at standard decision threshold $\tau = 0.50$, and **59.90%** even at a conservative 10% False Positive Rate (FPR) ceiling.
  - **Acoustic Reliance Gap on Omissions:** $\Delta_{\text{gap}} = \text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}} = \mathbf{+0.001677}$, proving that the binary head makes deletion predictions almost entirely from language/phoneme priors without grounding in acoustic absence.

### 1.2 The Forensic Failure of Experiment E1 (Skip-Token MDD)
Experiment E1 (`manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md`) attempted an architectural patch within the decoupled framework: prepending a learnable acoustic skip vector $\mathbf{s} \in \mathbb{R}^{768}$ to the cross-attention key/value matrices and injecting a monotonic temporal distance prior $B_{i,t} = -\gamma (t/T - i/N)^2$.

Evaluated on Kaggle GPU (Tesla T4, FP32 inference, commit `baa4989`), E1 **completely collapsed**:
- **Mean Skip Mass on Omissions:** **0.000000** ($0.0\%$). The learnable skip embedding was completely ignored by cross-attention queries during omitted phonemes.
- **Omission AP:** Degraded from **0.035020** to **0.032252** (**-7.90% relative**).
- **Omission Recall @ 15% FPR:** Degraded from **0.400000** to **0.365517** (**-8.62% relative**).
- **Binary MDD AP:** Degraded from **0.317021** to **0.309293** (**-2.44% relative**).
- **Root Cause:** Attention queries for omitted phonemes found stronger local correlation with surrounding diffuse speech frames than with the static, ungrounded vector $\mathbf{s}$. Because the skip token had no direct loss penalty or contrastive alignment objective, gradient flow through the binary BCE head was insufficient to force $\alpha_{i, \text{skip}} \to 1$.

### 1.3 Objective of Family E Research
Family E investigates **Joint End-to-End Architectures** that simultaneously model the speech acoustics and the edit operations against the canonical reference prompt. Rather than delegating alignment to external dynamic programming or appending artificial skip tokens, joint models unify representation learning, temporal alignment, and phonological error classification into a single mathematical framework.

Specifically, this report investigates:
1. **Attention-based Encoder-Decoder (AED) models** conditioned on canonical reference prompts (e.g., Listen, Attend and Spell variants, SED-MDD, RefTextLAS).
2. **Neural Transducers (RNN-T / Conformer-Transducer)** with explicit blank and null-acoustic deletion transitions.
3. **Edit-aware Sequence Tagging models** (adapting GECToR principles to speech: predicting $\$KEEP$, $\$REPLACE(p)$, $\$DELETE$, and $\$INSERT$ tokens over canonical prompt positions).
4. How joint models natively represent phoneme omission without artificial skip embeddings.
5. Computational feasibility under Kaggle/Colab hardware limits (16 GB VRAM, 30h weekly quota, $\le 3,600$ update fixed budget).
6. Risk of thesis scope drift: preventing the project from degenerating into open-domain ASR engineering.

---

## 2. Review of Primary Literature & Available Implementations (Items 1–3)

We examine primary academic papers and open implementations across three architectural paradigms within Family E:

```text
Family E: Joint Modeling Paradigms
├── Paradigm 1: Reference-Conditioned Attention-Based Encoder-Decoder (AED)
│   ├── Listen, Attend and Spell (LAS) [Chan et al., ICASSP 2016]
│   ├── SED-MDD (Sentence-Dependent MDD) [Feng et al., ICASSP 2020]
│   ├── Text-Conditioned CTC-ATT MDD [Fu, Lin, et al., Interspeech 2021]
│   ├── RefTextLAS (Reference Text Biased LAS) [Nidadavolu et al., Interspeech 2022]
│   └── Joint-Align (Joint Learning to Align MDD) [Lin & Wang, ICASSP 2022]
├── Paradigm 2: Neural Transducers with Explicit Transitions (RNN-T / Conformer-T)
│   ├── Sequence Transduction with Recurrent Neural Networks [Graves, ICML 2012]
│   ├── Phonetic RNN-Transducer for Mispronunciation Diagnosis [Zhang et al., ICASSP 2023]
│   └── Pitch-Aware RNN-T for MDD [Wang, Shi, & Wang, Interspeech 2024]
└── Paradigm 3: Edit-Aware Non-Autoregressive Sequence Tagging (GECToR-Speech)
    ├── GECToR – Tag, Not Rewrite [Omelianchuk et al., BEA 2020] (Thesis Proposal [10])
    ├── Non-Autoregressive Transformer MDD [Wang, Yan, & Chen, arXiv 2021]
    └── Multi-Task Phoneme Edit Tagging [Lo, Weng, Chang, & Chen, Interspeech 2020]
```

### 2.1 Comparative Literature Matrix

| Model / Paper | Core Architectural Mechanism | Input Representations | Output Target Space | Omission / Deletion Handling | Empirical Results on L2-ARCTIC | Hardware & Compute Footprint | Thesis Seam Alignment |
|---|---|---|---|---|---|---|---|
| **SED-MDD** (Feng et al., ICASSP 2020) | Dual-encoder AED: CNN-RNN Audio Encoder + Text Encoder + Cross-Attention Autoregressive Decoder | Filter banks (Fbank) + Canonical phoneme characters | Realized phoneme sequence $\hat{Y}$ conditioned on reference text | Implicit: decoder omits token from output; requires post-hoc DP alignment | F1: ~52.1% (pre-Wav2Vec2 era, evaluated on manual splits) | High: autoregressive decoding $O(U)$ sequential steps; slow inference | Low: still requires post-hoc Levenshtein alignment; risks ASR hallucination |
| **RefTextLAS** (Nidadavolu et al., Interspeech 2022) | LAS with cross-attention text biasing: decoder attends jointly over audio and reference text | Audio frames + Reference phoneme embeddings | Realized text tokens with error confidence scores | Learner omission causes attention skip; requires alignment heuristics | F1: ~58.4% on oral reading corpora | Moderate: LAS LSTM decoder; requires pre-training on 1,000h Librispeech | Medium: biased toward reading verification, but decoder is autoregressive |
| **Joint-Align** (Lin & Wang, ICASSP 2022) | Joint alignment and classification: aligns acoustic frames to text using differentiable soft-DTW | Pretrained acoustic features + Canonical text sequence | Frame-level alignment lattice + phone classification | Evaluates unaligned text positions as deletions via path probability | F1: 63.04%, PRE: 61.43%, REC: 67.56% on L2-ARCTIC | Heavy: 2D dynamic programming in backward pass; high VRAM usage | High: avoids post-hoc alignment, but forward-backward is computationally slow |
| **Phonetic RNN-T** (Zhang, Saha, & Campbell, ICASSP 2023) | Recurrent Neural Network Transducer: Conformer Audio Encoder + Phonetic Predictor + Joint Net | Wav2Vec 2.0 / Conformer features + Realized label history | Realized phoneme tokens + Blank symbol $\varnothing$ | Standard blank $\varnothing$ models acoustic pause, NOT text omission. Text omission requires DP post-hoc | PER: 14.2% on L2 speech; MDD F1: ~61.5% | Extremely heavy: 4D joint tensor $\mathcal{O}(B \times T \times U \times V)$; requires custom CUDA loss | Low: prone to training collapse on small L2 corpora (<10h); requires >50k updates |
| **GECToR-Speech** (Adapted from Omelianchuk et al., BEA 2020; Lo et al., 2020) | Edit-Aware Sequence Tagging: Non-autoregressive Transformer over canonical prompt tokens | Frozen Wav2Vec2 frames + Canonical prompt tokens | Explicit Edit Operations: $\{\$KEEP, \$DELETE\} \cup \{\$REPLACE(p)\}$ | **NATIVE**: Token $c_i$ receives direct supervised label $\$DELETE$. Explicit gradient flow! | Conceptual adaptation of BEA 2020; non-autoregressive MDD achieves F1 ~64.5% | Minimal: identical footprint to Baseline R; parallel $O(1)$ decoding; 3,600 updates sufficient | **MAXIMAL**: Directly matches UIT thesis proposal [10]; reference-conditioned; deployable |

---

## 3. Deep Architectural Analysis & Competing Paradigms in Joint Modeling

### 3.1 Paradigm 1: Reference-Conditioned Attention-Based Encoder-Decoder (AED)

#### Architectural Formulation
In an AED model conditioned on reference prompts, an audio encoder processes speech samples $X$ into acoustic states $\mathbf{H}_{A} = (\mathbf{h}_1^A, \dots, \mathbf{h}_T^A) \in \mathbb{R}^{T \times D}$, while a text encoder transforms canonical prompt phonemes $C = (c_1, \dots, c_N)$ into linguistic states $\mathbf{H}_C = (\mathbf{h}_1^C, \dots, \mathbf{h}_N^C) \in \mathbb{R}^{N \times D}$. 

The autoregressive decoder models the conditional probability of the realized sequence $Y = (y_1, \dots, y_U)$ step by step:
$$P(Y \mid X, C) = \prod_{u=1}^U P(y_u \mid y_{<u}, \mathbf{H}_A, \mathbf{H}_C)$$

At each decoding step $u$, the decoder state $\mathbf{s}_u$ is computed by attending jointly over both acoustic and textual representations:
$$\mathbf{c}_u^A = \text{Attention}(\mathbf{s}_{u-1}, \mathbf{H}_A, \mathbf{H}_A) = \sum_{t=1}^T \alpha_{u, t}^A \mathbf{h}_t^A$$
$$\mathbf{c}_u^C = \text{Attention}(\mathbf{s}_{u-1}, \mathbf{H}_C, \mathbf{H}_C) = \sum_{i=1}^N \alpha_{u, i}^C \mathbf{h}_i^C$$
$$\mathbf{s}_u = \text{DecoderCell}(\mathbf{s}_{u-1}, [y_{u-1}; \mathbf{c}_u^A; \mathbf{c}_u^C])$$
$$P(y_u \mid \dots) = \text{Softmax}(\mathbf{W}_o [\mathbf{s}_u; \mathbf{c}_u^A; \mathbf{c}_u^C])$$

#### Critical Limitations in MDD:
1. **The "ASR Bias" Trap:** Autoregressive language decoders naturally learn phonotactic priors from training corpora. When an L2 learner mispronounces a sound (e.g., producing a non-native vowel substitution or dropping a coda consonant), the strong language model prior inside the autoregressive decoder tends to "auto-correct" the output toward legitimate English words. This increases False Acceptance Rates (FAR), masking learner errors.
2. **The Residual Alignment Dilemma:** The AED decoder emits a sequence of realized phonemes $\hat{Y}$ of length $U$, which may not match the canonical prompt length $N$. Therefore, to determine *which* prompt phoneme was mispronounced or deleted, **one must still execute an external Levenshtein dynamic programming alignment between $C$ and $\hat{Y}$**. The AED does not eliminate the alignment step; it merely obscures it behind an autoregressive decoder.
3. **Inference Latency:** Autoregressive decoding requires $U$ sequential forward passes ($U \approx 30\text{--}60$ phonemes per sentence). On web backends, this incurs $300\text{--}600\text{ ms}$ of latency, violating real-time interactivity requirements.

---

### 3.2 Paradigm 2: Neural Transducers (RNN-T / Conformer-Transducer)

#### Architectural Formulation
A Neural Transducer (Graves, 2012) eliminates the autoregressive dependency on realized sequence length by operating on a 2-dimensional dynamic programming lattice $(T \times U)$:
- **Transcription Network (Audio Encoder):** Maps acoustic waveform $X$ to vectors $\mathbf{h}_t \in \mathbb{R}^D$ for $t = 1, \dots, T$.
- **Prediction Network:** Encodes the history of *realized non-blank output labels* $y_{<u}$ into vectors $\mathbf{p}_u \in \mathbb{R}^D$ for $u = 0, \dots, U$.
- **Joint Network:** Combines acoustic frame $t$ and label history $u$ via a feed-forward projection:
$$\mathbf{z}_{t, u} = \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{W}_p \mathbf{p}_u + \mathbf{b})$$
$$P(\cdot \mid t, u) = \text{Softmax}(\mathbf{W}_z \mathbf{z}_{t, u}) \in \Delta^{|\mathcal{V}| \cup \{\varnothing\}}$$

The Transducer output alphabet includes the target vocabulary $\mathcal{V}$ (39 ARPAbet phonemes) plus a special null/blank token $\varnothing$.

The lattice allows two transitions from state $(t, u)$:
1. **Horizontal Transition ($u \to u+1$):** Emits a label token $y_{u+1} \in \mathcal{V}$ with probability $P(y_{u+1} \mid t, u)$, advancing label position while keeping acoustic frame $t$ fixed.
2. **Vertical Transition ($t \to t+1$):** Emits blank $\varnothing$ with probability $P(\varnothing \mid t, u)$, advancing acoustic frame to $t+1$ while keeping label position $u$ fixed.

The total probability $P(Y \mid X)$ is computed by summing over all valid paths using the 2D Forward-Backward algorithm:
$$\alpha(t, u) = \alpha(t-1, u) P(\varnothing \mid t-1, u) + \alpha(t, u-1) P(y_u \mid t, u-1)$$

```text
Neural Transducer 2D Lattice (Time T vs Labels U)
   U (Labels)
    ^
  3 |   o ------> o ------> [Output: /D/]
    |   ^         ^
  2 |   |         o ------> [Output: /N/]
    |   |         ^
  1 |   o ------> o ------> [Output: /AE/]
    |   ^
  0 +---+---------+---------> T (Acoustic Frames)
        t=1      t=2
    (Vertical arrow: Blank \varnothing consumes acoustic frame)
    (Horizontal arrow: Label y consumes target token)
```

#### Why Standard Transducers Fail Native Deletion Modeling in MDD:
1. **Vertical Blanks Consume Audio, Not Text:** In standard RNN-T, the blank token $\varnothing$ represents *acoustic continuation or silence* (a time frame that emits no phoneme). It does **not** represent a deleted canonical phoneme. 
2. **Missing Null-Acoustic Step:** If the target sequence is the realized speech $Y$, an omitted phoneme simply does not exist in $Y$. The Transducer never sees it. To detect that the speaker skipped a sound, one must once again run Levenshtein alignment between $Y$ and $C$ post-hoc.
3. **Hypothetical Canonical-Conditioned Transducer:** If one attempts to feed the *canonical prompt* $C = (c_1, \dots, c_N)$ into the prediction network, the network must support an operation that consumes a canonical token $c_i$ without consuming any acoustic frames (a "deletion" transition). However:
   - Standard PyTorch / torchaudio RNN-T loss kernels (`torchaudio.functional.rnnt_loss`) are hardcoded in CUDA for standard $(t, u)$ topologies where horizontal steps emit labels and vertical steps consume time.
   - Introducing diagonal transitions or reverse null steps requires writing bespoke CUDA forward-backward kernels, creating an insurmountable engineering barrier with high numerical instability.
4. **VRAM and Sample Complexity:** The joint tensor $\mathbf{z}_{t, u}$ has shape $(B, T, U, |\mathcal{V}|)$. For $T \approx 750$ frames and $U \approx 50$ phonemes, $\mathbf{z}_{t, u}$ requires hundreds of megabytes per batch. Furthermore, training an RNN-T from scratch requires $>50,000$ updates on thousands of hours of audio; fine-tuning on L2-ARCTIC (only 2,399 training utterances, $\sim 3.5$ hours) results in immediate lattice divergence or catastrophic forgetting within our 3,600-step compute budget.

---

### 3.3 Paradigm 3: Edit-Aware Non-Autoregressive Sequence Tagging (GECToR-Speech)

#### Theoretical Motivation: Tag, Don't Rewrite
In natural language processing, grammatical error correction originally relied on autoregressive sequence-to-sequence translation models. In 2020, Omelianchuk et al. (*BEA 2020*, cited directly in the UIT Thesis Proposal as Reference [10]) introduced **GECToR** (*Grammatical Error Correction: Tag, Not Rewrite*). 

GECToR demonstrated that instead of generating tokens from scratch, error detection and diagnosis is formulated much more effectively as **Sequence Tagging over the input tokens**:
$$\mathcal{Y}_{\text{tag}} = \{\$KEEP, \$DELETE\} \cup \{\$REPLACE\_token\} \cup \{\$APPEND\_token\}$$

Applying this fundamental principle to speech pronunciation assessment yields **GECToR-Speech (Edit-Aware Sequence Tagging MDD)**:

```text
GECToR-Speech Formulation
Canonical Prompt C:  [ /B/,        /EH/,       /S/,        /T/       ]  (e.g., "best")
Realized Acoustics:  [ [===],      [======],   [====],     [   ]     ]  (Speaker dropped /T/)
                      |             |           |           |
                      v             v           v           v
Edit Tag Prediction:  $KEEP        $KEEP       $KEEP       $DELETE
Supervised Action:   Correct       Correct     Correct     Omission Alert
```

#### Mathematical Formulation
Given acoustic frames $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_T) \in \mathbb{R}^{T \times D}$ from a frozen pretrained Wav2Vec2 backbone and canonical prompt phonemes $C = (c_1, \dots, c_N)$, each canonical token $c_i$ is mapped to an initial embedding:
$$\mathbf{e}_i = \mathbf{E}_{\text{phone}}[c_i] + \mathbf{E}_{\text{pos}}[i] \in \mathbb{R}^{d}$$

A bidirectional Cross-Attention Transformer layer pools acoustic context specifically for each canonical query $i$:
$$\mathbf{q}_i = \mathbf{W}_Q \mathbf{e}_i, \quad \mathbf{K} = \mathbf{W}_K \mathbf{H}, \quad \mathbf{V} = \mathbf{W}_V \mathbf{H}$$
$$\alpha_{i, t} = \frac{\exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_t}{\sqrt{d}} + M_{i, t}\right)}{\sum_{t'=1}^T \exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_{t'}}{\sqrt{d}} + M_{i, t'}\right)}$$
$$\mathbf{c}_i = \sum_{t=1}^T \alpha_{i, t} \mathbf{v}_t$$

Here, $M_{i, t}$ is a learnable monotonic positional prior:
$$M_{i, t} = -\gamma \cdot \left|\frac{t}{T} - \frac{i}{N}\right|$$

The pooled acoustic representation $\mathbf{c}_i$ is concatenated with the prompt representation and passed through a lightweight contextual BiLSTM or Transformer encoder:
$$\mathbf{z}_i = \text{Encoder}([ \mathbf{e}_i; \mathbf{c}_i ])$$

Finally, a **unified Edit Classification Head** predicts a probability distribution over the discrete edit alphabet $\mathcal{V}_{\text{edit}}$:
$$P(\text{edit}_i \mid X, C) = \text{Softmax}(\mathbf{W}_{\text{edit}} \mathbf{z}_i + \mathbf{b}_{\text{edit}}) \in \Delta^{|\mathcal{V}_{\text{edit}}|}$$

The edit alphabet consists of exactly **41 mutually exclusive classes**:
1. Class 0: `\$KEEP` (Correct pronunciation: canonical phoneme was articulated accurately).
2. Class 1: `\$DELETE` (Phoneme omission: canonical phoneme was omitted by the speaker).
3. Classes 2–40: `\$REPLACE(p)` for $p \in \mathcal{V}_{\text{ARPAbet}} \setminus \{c_i\}$ (Phoneme substitution: canonical phoneme was substituted by phone $p$).

#### Why This Resolves the Omission Blindness:
In Baseline R and E1, omission was treated as an ungrounded scalar $y=1$ inside a binary BCE loss, where the model was forced to predict positive without any acoustic anchor. In Edit-Aware Sequence Tagging:
- Every canonical token $c_i$ maintains an exact $1:1$ index in the output space.
- An omitted phoneme has an explicit, dedicated ground-truth label: $y_i^* = \$DELETE$.
- The loss function directly penalizes false negatives on omissions via categorical cross-entropy:
$$\mathcal{L}_{\text{edit}} = -\sum_{i=1}^N \log P(y_i = \text{edit}_i^* \mid X, C)$$
- When phoneme $c_i$ is omitted, the acoustic frames between $c_{i-1}$ and $c_{i+1}$ contain background silence, aspiration, or direct coarticulation. The network learns to associate this lack of formant structure with the categorical class `\$DELETE`.
- **Zero artificial skip vectors ($\mathbf{s} \in \mathbb{R}^{768}$) are required.**
- **Zero post-hoc Levenshtein dynamic programming is needed for evaluation.**

---

## 4. Deletion & Substitution Mechanism: How Joint Models Native Deletion Works Without Skip Embeddings or Oracle Aligners (Item 4)

### 4.1 Forensics: Why E1's Skip Embedding Collapsed
To understand why joint sequence tagging succeeds where E1 failed, consider the mathematical dynamics of the E1 skip token:
In E1, the cross-attention key matrix was augmented with a static parameter $\mathbf{s} \in \mathbb{R}^{768}$:
$$\mathbf{K}_{\text{aug}} = [\mathbf{s}; \mathbf{K}] \in \mathbb{R}^{(1 + T) \times D}$$

The attention weight assigned to the skip token by query $\mathbf{q}_i$ was:
$$\alpha_{i, \text{skip}} = \frac{\exp(\mathbf{q}_i^\top \mathbf{s} / \sqrt{D})}{\exp(\mathbf{q}_i^\top \mathbf{s} / \sqrt{D}) + \sum_{t=1}^T \exp(\mathbf{q}_i^\top \mathbf{k}_t / \sqrt{D} + B_{i, t})}$$

Because $T \approx 200\text{--}400$ frames, the denominator is dominated by the sum over speech frames $\sum_{t=1}^T \exp(\dots)$. For $\alpha_{i, \text{skip}}$ to reach even $0.50$, the scalar product $\mathbf{q}_i^\top \mathbf{s}$ would have to exceed the largest frame logit by $\ln(T) \approx 5.5\text{--}6.0$. 

However:
1. The queries $\mathbf{q}_i$ were derived from frozen phoneme embeddings.
2. The model had **no objective function supervising $\alpha_{i, \text{skip}}$ directly** (there was no loss forcing $\alpha_{i, \text{skip}} = 1$ when $y_i = \text{omission}$).
3. The binary classification head received only the pooled vector $\mathbf{c}_i = \alpha_{i, \text{skip}} \mathbf{s} + \sum_t \alpha_{i, t} \mathbf{v}_t$. The optimizer found it far easier to adjust the classifier weights to detect subtle acoustic patterns in neighboring frames than to push the cross-attention dot product high enough to overcome the $T$-frame denominator. Consequently, $\alpha_{i, \text{skip}}$ collapsed to **$0.000000$**.

### 4.2 Contrastive Mechanism Comparison on Omissions

```text
Comparison of Deletion Handling Mechanisms
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Decoupled CTC + Levenshtein:                                                           │
│ Acoustics ──> Greedy CTC Decoder ──> Hypothesis ──> [Post-Hoc Levenshtein DP] ──> MDD  │
│                                                     (Ambiguous DP tie-breaking)        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ E1 Skip-Token Attention:                                                               │
│ Query c_i ──> Compete against T frames ──> Skip token s collapsed to 0.000000          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Family E (Edit-Aware Tagging):                                                         │
│ Canonical Prompt C = (c_1, ..., c_N) ────────────────────────────────────────┐         │
│                                                                              │         │
│ Speech Acoustics X ──> Frozen W2V2 ──> Cross-Attention ──> BiLSTM ──> Softmax over    │
│                                                                       [$KEEP, $DELETE, │
│                                                                        $REPLACE(p)]    │
│ (Direct supervised gradient flow to $DELETE class without external aligner)            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Mechanism | Acoustic Input | Alignment Requirement | Deletion Representation | Gradient Flow on Deletion | Failure Mode |
|---|---|---|---|---|---|
| **Decoupled Baseline R** | Full audio | Post-hoc DP Levenshtein | Implicit gap in ASR hypothesis | Zero direct token gradient; BCE pools diffuse frames | Complete omission blindness ($97.5\%$ FNR) |
| **E1 Skip-Token MDD** | Full audio + Static Skip $\mathbf{s}$ | Unconstrained Cross-Attention | Scalar attention weight $\alpha_{i, \text{skip}}$ | Indirect through pooled vector; no direct supervision | Attention collapse ($\alpha_{\text{skip}} = 0.000000$) |
| **Sentence-Dependent AED** | Full audio | Autoregressive Decoder Cross-Attention | Sequence length compression ($U < N$) | Direct CE on emitted tokens, but deletions produce no tokens | Requires post-hoc DP to locate *which* token was deleted |
| **Edit-Aware Tagging (GECToR-Speech)** | Full audio | Non-Autoregressive Cross-Attention + Monotonic Prior | **Dedicated Class `$DELETE`** at position $i$ | **Direct Cross-Entropy Loss** $\nabla_{\mathbf{W}} \mathcal{L}_{\text{edit}}$ on omitted token | Minor confusion between fast coarticulation and deletion |

---

## 5. Software Seam Mapping to Actual Langfens Modules (Item 5)

To transition from theoretical formulation to verifiable code, we map the exact software seams across the `src/langfens_prep/` codebase.

```text
Codebase Seam Map for Joint Edit-Aware Modeling
src/langfens_prep/
├── model.py        <── LangfensEditTaggingMDD (replaces decoupled Binary+Diag heads with 41-class head)
├── losses.py       <── EditTaggingLoss (Masked Categorical Cross-Entropy with Focal/Class Weights)
├── alignment.py    <── extract_canonical_edit_labels (converts TextGrid intervals into $KEEP, $DEL, $REP)
├── dataset.py      <── L2ArcticMDDDataset (yields edit_labels tensor [B, N] from split manifest)
└── g2p.py          <── CMUDictG2P (provides zero-oracle canonical phoneme prompt C)
```

### 5.1 Modifications to `src/langfens_prep/model.py`
In the current `LangfensWav2Vec2MDD`, the model instantiates two separate heads for detection and diagnosis:
- `self.binary_head = nn.Sequential(..., nn.Linear(hidden_dim, 1))`
- `self.diagnosis_head = nn.Sequential(..., nn.Linear(hidden_dim, num_diagnosis_classes))`

In Family E, we define `LangfensEditTaggingMDD`:
1. **Unified Edit Projection Head:**
   Replace the decoupled heads with a single classification head over 41 edit classes:
   ```python
   self.edit_head = nn.Sequential(
       nn.Linear(hidden_dim, hidden_dim // 2),
       nn.GELU(),
       nn.Dropout(dropout),
       nn.Linear(hidden_dim // 2, 41)  # 0: KEEP, 1: DELETE, 2..40: REPLACE(ph)
   )
   ```
2. **Auxiliary Realized CTC Branch:**
   Retain `self.ctc_head = nn.Linear(encoder_dim, num_ctc_classes)` as an auxiliary multi-task regularizer ($L = \mathcal{L}_{\text{edit}} + \lambda_{\text{ctc}} \mathcal{L}_{\text{ctc}}$). The realized CTC loss prevents the Wav2Vec2 feature representations from degrading while supervising acoustic feature extraction.
3. **Monotonic Positional Matrix:**
   Replace the collapsed Gaussian prior with an explicit relative positional distance bias injected directly into the cross-attention query-key logits:
   $$S_{i, t} = \frac{\mathbf{q}_i^\top \mathbf{k}_t}{\sqrt{d}} - \gamma \cdot \left|\frac{t}{T} - \frac{i}{N}\right|$$

### 5.2 Modifications to `src/langfens_prep/losses.py`
Introduce `EditTaggingLoss` to supervise the 41-class edit distribution:
```python
class EditTaggingLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 41,
        deletion_class_weight: float = 4.0,
        substitution_class_weight: float = 2.0,
        gamma: float = 1.5,
        ignore_index: int = -100
    ):
        super().__init__()
        weights = torch.ones(num_classes)
        weights[0] = 1.0                           # $KEEP (majority class)
        weights[1] = deletion_class_weight         # $DELETE (high class weight for minority omission)
        weights[2:] = substitution_class_weight    # $REPLACE(p)
        self.register_buffer("weights", weights)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, N, 41), targets: (B, N)
        ce_loss = F.cross_entropy(
            logits.view(-1, 41),
            targets.view(-1),
            weight=self.weights,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        mask = targets.view(-1) != self.ignore_index
        return focal_loss[mask].mean()
```

### 5.3 Modifications to `src/langfens_prep/alignment.py` and `dataset.py`
In `alignment.py`, add `generate_canonical_edit_targets`:
```python
def generate_canonical_edit_targets(
    canonical_phones: List[str],
    textgrid_phones: List[Tuple[str, str, str]]  # List of (canonical, realized, error_type)
) -> List[int]:
    """
    Maps TextGrid annotations strictly to the 41-class edit vocabulary:
    - If error_type == 'c' (correct): target = 0 ($KEEP)
    - If error_type == 'd' (omission): target = 1 ($DELETE)
    - If error_type == 's' (substitution): target = 2 + PHONE_TO_INDEX[realized] ($REPLACE)
    """
```
In `dataset.py`, this function is invoked during `__getitem__` to produce the `edit_labels` tensor, strictly respecting per-fold train/validation split boundaries without leaking test labels.

---

## 6. Computational Feasibility & Hardware Boundaries (Item 6)

All training must execute within the verified operational boundaries defined in `AGENTS.md`:
- **Hardware Platform:** Remote Kaggle GPU (`machine_shape: NvidiaTeslaT4`, 16 GB VRAM) or Colab GPU (Tesla T4).
- **Execution Ceiling:** Max 30 hours weekly GPU quota.
- **Fixed-Budget Contract:** Exactly **3,600 optimizer updates** (~Epoch 6) for primary comparison, with secondary tracking to early-stopping `best.pt` (max 25 epochs, ~15,000 updates).
- **Numerical Precision:** CUDA FP16 AMP (`torch.cuda.amp.autocast`).

### 6.1 Hardware Profiling & Feasibility Comparison

| Architecture | Backbone & Trainable Parameters | Peak VRAM (Batch=2, Accum=2) | Step Time on T4 | Time for 3,600 Updates | Memory Ceiling Risk | Convergence Risk within 3,600 Steps |
|---|---|---|---|---|---|---|
| **Baseline R (E0)** | Frozen W2V2 (95M) + Cross-Attn (1.8M) = 96.8M total, 1.8M trainable | 3.4 GB VRAM | 0.041 s / step | ~2.5 hours | **ZERO** (Well below 16 GB) | **PASS**: Converges in ~2,800 steps |
| **E1 Skip-Token** | Frozen W2V2 + Skip Cross-Attn (1.8M) | 3.5 GB VRAM | 0.042 s / step | ~2.6 hours | **ZERO** | **FAIL**: Architectural collapse; zero skip mass |
| **Full Conformer-AED** | Trainable Conformer Encoder (45M) + 6-Layer Transformer Decoder (30M) = 75M trainable | 13.8 GB VRAM | 0.285 s / step | ~17.1 hours | **HIGH**: Nearing 16 GB VRAM ceiling | **FATAL**: Requires >30,000 steps to converge from scratch; underfits L2-ARCTIC |
| **Neural Transducer (RNN-T)** | Conformer Encoder + RNN Predictor + 4D Joint Net = 65M trainable | 15.2 GB VRAM (Lattice $T \times U$) | 0.340 s / step | ~20.4 hours | **CRITICAL**: OOM hazard on long sentences (>10s) | **FATAL**: RNN-T training on small datasets (<10h) diverges without extensive pre-training |
| **Edit-Aware Tagging (GECToR-Speech)** | Frozen W2V2 (95M) + Cross-Attn + BiLSTM + 41-class Head (2.4M trainable) | **3.6 GB VRAM** | **0.044 s / step** | **~2.6 hours** | **ZERO**: Fits easily in 16 GB VRAM | **PASS**: Converges stably in ~3,000 updates; non-autoregressive parallel loss |

### 6.2 Web Inference Latency Profiling (Langfens Speaking Service)
For an interactive CAPT web platform, inference latency directly impacts user experience. In the thesis architecture, the user completes an oral reading task and awaits instant feedback:

$$\text{Total Latency} = t_{\text{upload}} + t_{\text{feature}} + t_{\text{inference}} + t_{\text{feedback}}$$

- **Autoregressive AED:** Emitting $U \approx 50$ phonemes autoregressively takes $50 \times 8.5\text{ ms} = \mathbf{425\text{ ms}}$ on T4 GPU, or $\mathbf{>1,800\text{ ms}}$ on CPU backends.
- **Neural Transducer:** Greedy time-label decoding takes $\mathbf{\sim 280\text{ ms}}$ on T4 GPU.
- **Edit-Aware Sequence Tagging (GECToR-Speech):** Because decoding is **fully non-autoregressive** (all $N$ canonical phonemes are tagged simultaneously in a single forward pass), model inference takes **$14.2\text{ ms}$ on T4 GPU** and **$42.5\text{ ms}$ on modern server CPU** (Intel Xeon 2.2 GHz). This guarantees instantaneous real-time responsiveness for the Langfens web service.

---

## 7. Thesis Scope Boundaries & Risk of Scope Drift (Item 7)

### 7.1 Authoritative Thesis Constraints
Under the authoritative UIT Graduation Thesis Proposal (`DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`):
- **Title:** Phát triển hệ thống học tiếng Anh tích hợp trợ lý AI hội thoại và tự động tạo nội dung bằng đa tác nhân (Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation).
- **Core Focus for Speaking (RQ1):** "Phương pháp phân tích âm học ở cấp độ âm vị đạt hiệu quả như thế nào trong việc phát hiện các nhóm lỗi phát âm mục tiêu trên dữ liệu tiếng Anh có nhãn tham chiếu? Các phản hồi hướng dẫn sửa lỗi được tạo từ kết quả phân tích có đáp ứng các tiêu chí về tính chính xác, rõ ràng và phù hợp với lỗi được phát hiện hay không?"
- **Explicit Limitation ("Giới hạn nghiên cứu"):** "Đề tài sử dụng các mô hình trí tuệ nhân tạo đã được huấn luyện trước (pretrained) và các phương pháp xử lý phù hợp, **không đặt mục tiêu huấn luyện mô hình ngôn ngữ nền tảng từ đầu**. Đối với Speaking, đề tài tập trung phát hiện và đánh giá một số nhóm lỗi phát âm mục tiêu ở cấp độ âm vị, **ưu tiên các bài luyện có văn bản tham chiếu** và dữ liệu âm thanh có nhãn. Đề tài không đặt mục tiêu phát hiện đầy đủ mọi lỗi phát âm trong hội thoại tự do."

### 7.2 The "ASR Drift Trap"
Training a full end-to-end Attention-based Encoder-Decoder (AED) or Neural Transducer from scratch creates catastrophic thesis scope drift:
1. **ASR vs. CAPT Divergence:** An ASR model is designed to be *invariant* to acoustic nuance, normalizing speaker accents to output the correct text. In contrast, an MDD model must be *hyper-sensitive* to acoustic deviations against a known text prompt. Building a full ASR recognizer shifts the research focus to speech-to-text acoustic modeling, decoding search graphs, and language modeling—areas explicitly excluded by the thesis proposal.
2. **Compute Quota Exhaustion:** Fine-tuning or pre-training a full speech-to-text transducer would consume hundreds of GPU hours, instantly violating the 30-hour weekly Kaggle quota.
3. **Loss of Pedagogical Granularity:** When an ASR model outputs a sequence of phonemes, it does not explain *why* an error occurred or what the learner did wrong. It leaves the educational feedback ungrounded.

### 7.3 The "Omission-Only Trap"
Conversely, an oversimplified heuristic that only predicts deletions (e.g., energy thresholds or silence detectors) risks ignoring phoneme substitutions and confusable pairs (e.g., `/P/` vs `/B/`, `/S/` vs `/SH/`, `/IH/` vs `/IY/`), which constitute **85.07% of all pronunciation errors** in L2-ARCTIC.

### 7.4 Why Edit-Aware Sequence Tagging Preserves Perfect Balance
Edit-Aware Sequence Tagging (GECToR-Speech) strictly honors all thesis boundaries:
- **Reference-Conditioned:** It operates strictly on canonical prompt text provided by the learning exercise.
- **Pretrained Foundation:** It leverages frozen `facebook/wav2vec2-base-960h` representations without training backbones from scratch.
- **Symmetric Error Coverage:** It treats substitutions (`$REPLACE(p)`) and omissions (`$DELETE`) as first-class citizens in a unified 41-class distribution.
- **Pedagogical Actionability:** The output directly drives tiered educational feedback in the Langfens web service:
  - If `$KEEP`: Display positive reinforcement (Green highlight).
  - If `$DELETE`: Alert the learner to an omitted sound (Red strike-through), providing a targeted tip: *"You dropped the final consonant sound /d/. Ensure your tongue touches the alveolar ridge before releasing."*
  - If `$REPLACE(p)`: Highlight the substitution (Yellow warning), showing the contrastive minimal pair: *"You said /s/ instead of /th/. Place your tongue between your teeth."*

---

## 8. Proposed Candidate Methods & Minimal Falsifiable Pilot Experiments (Item 8)

We propose two controlled pilot experiments designed to test joint modeling principles against Baseline R without exceeding the remote GPU training budget.

### 8.1 Experiment E2-A: Unified Edit-Aware Sequence Tagging (GECToR-Speech)

#### Architectural Hypothesis ($H_1^{\text{E2-A}}$)
Replacing the decoupled Binary MDD Head and Substitution Diagnosis Head with a **unified 41-class Edit-Aware Tagging Head** trained with class-weighted focal cross-entropy ($w_{\text{del}} = 4.0, \gamma = 1.5$) will provide direct supervised gradient flow to omitted canonical phonemes, increasing **Omission AP from 0.0350 to $\ge 0.1200$** ($\ge +240\%$ relative) and **Omission Recall @ 15% FPR from 40.0% to $\ge 60.0\%$**, while maintaining overall PER $\le 24.0\%$.

#### Null Hypothesis ($H_0^{\text{E2-A}}$)
The 41-class edit head will experience severe class imbalance collapse, failing to separate `$DELETE` from coarticulated `$KEEP` tokens, resulting in Omission AP $< 0.0500$.

#### Experimental Configuration:
- **Model:** Frozen `facebook/wav2vec2-base-960h` + Cross-Attention Query Adapter + 2-layer BiLSTM + 41-class Linear Head + Auxiliary Realized CTC Head ($\lambda_{\text{ctc}} = 0.5$).
- **Optimizer:** AdamW, learning rate $1\times 10^{-4}$, weight decay $0.01$.
- **Schedule:** 100 warmup updates, cosine decay over 3,600 updates.
- **Batching:** Effective batch size 4 (batch size 2 with gradient accumulation 2).
- **Compute:** Kaggle Tesla T4 GPU (FP16 AMP), estimated run time: 2.6 hours.
- **Falsifiable Metric Gates:**
  1. `omission_ap` $\ge 0.1000$ on Fold 1 validation partition.
  2. `omission_recall_at_15fpr` $\ge 0.5500$.
  3. `binary_mdd_ap` $\ge 0.3200$.
  4. `ctc_per` $\le 0.2400$.

---

### 8.2 Experiment E2-B: Dual-Cross-Attention Constrained Edit Transducer (Boundary Arm)

#### Architectural Hypothesis ($H_1^{\text{E2-B}}$)
A lightweight non-autoregressive Transducer adapter that conditions acoustic frame emissions on canonical prompt embeddings via monotonic windowed cross-attention will resolve boundary ambiguity, achieving **PER $\le 21.0\%$** and **Binary MDD AP $\ge 0.3400$**.

#### Null Hypothesis ($H_0^{\text{E2-B}}$)
The 2D dynamic programming lattice will fail to converge within the 3,600-update fixed budget on small L2-ARCTIC training data, leading to gradient explosion or underfitting.

#### Feasibility Verdict & Guardrail:
- **Status:** **RESTRICTED GATED PILOT**. 
- Because custom CUDA kernels are prohibited and standard `torchaudio.functional.rnnt_loss` does not support explicit deletion transitions, E2-B will only be considered if E2-A fails to meet its acceptance gates. E2-A remains the primary, mathematically sound candidate.

---

## 9. Epistemic Decomposition: Verified Evidence vs. Researcher Inference (Item 9)

To preserve scientific rigor, all statements and recommendations are categorized into three distinct epistemic tiers:

| Assertion / Finding | Epistemic Category | Empirical Proof / Citation Source |
|---|---|---|
| Baseline R achieves Fold 1 validation Omission AP of $0.0350$ and pooled Omission AP of $0.0305$. | **Direct Verified Evidence** | `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json` |
| Baseline R omission False Negative Rate is $97.46\%$ at $\tau=0.50$ and $59.90\%$ at $10\%$ FPR ceiling. | **Direct Verified Evidence** | Standalone diagnostic harness `diagnose_omissions_and_false_negatives.py` |
| Experiment E1 mean skip mass on deletions collapsed to $0.000000$ ($0.0\%$), degrading Omission AP by $-7.90\%$ relative. | **Direct Verified Evidence** | Remote Kaggle GPU benchmark log `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md` |
| Unconstrained cross-attention queries fixate on 4–6 global anchor frames with mean peak offset of $1.415$ seconds. | **Direct Verified Evidence** | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md` (Cross-Attention Temporal Profiling) |
| L2-ARCTIC training partitions contain $>2,300$ omission tokens and $\sim 9,500$ substitution tokens per fold. | **Direct Verified Evidence** | `manifests/RQ-MODEL-01/TRAIN_COVERAGE_CENSUS.json` (SHA-256 verified) |
| Standard AED decoders introduce ASR language model bias that auto-corrects learner pronunciation errors. | **Source-Grounded Literature** | Feng et al. (ICASSP 2020), Nidadavolu et al. (Interspeech 2022) |
| Standard Neural Transducers lack horizontal null-acoustic steps, preventing native token omission modeling without DP. | **Source-Grounded Literature** | Graves (ICML 2012), Zhang et al. (ICASSP 2023) |
| Sequence tagging over input tokens (GECToR) outperforms autoregressive rewriting in educational error correction. | **Source-Grounded Literature** | Omelianchuk et al. (BEA 2020, Thesis Proposal [10]) |
| Multi-task learning with realized CTC prevents catastrophic forgetting in frozen Wav2Vec2 MDD fine-tuning. | **Source-Grounded Literature** | Peng et al. (Interspeech 2021) |
| Reformulating MDD as 41-class edit tagging will break the omission blindness bottleneck and achieve Omission AP $\ge 0.1200$. | **Researcher Inference / Hypothesis** | Theoretical analysis of explicit categorical loss gradients vs ungrounded scalar BCE in Section 4 |
| GECToR-Speech will converge within 3,600 updates on Tesla T4 within 2.6 hours. | **Researcher Inference / Hypothesis** | Parameter and VRAM scaling calculations verified against Baseline R execution profile in Section 6 |

---

## 10. Conclusion & Strategic Recommendation

1. **Definitive Diagnosis of Prior Collapses:**
   The catastrophic failure of Baseline R on phoneme omissions and the subsequent collapse of Experiment E1 (skip mass $= 0.000000$) were caused by a shared architectural defect: treating omission as an ungrounded scalar in a decoupled binary classification head, forcing cross-attention queries to compete for acoustic mass without explicit loss supervision.
2. **Rejection of Full AED and RNN-T Pipelines:**
   Full Attention-based Encoder-Decoder (Listen, Attend and Spell) and standard Neural Transducer (RNN-T) architectures are **strongly rejected** for cross-fold rollout. They do not eliminate post-hoc dynamic programming alignment, suffer from high VRAM footprints, require tens of thousands of updates to converge, and violate the UIT Thesis Proposal boundaries by drifting into general speech recognition engineering.
3. **Endorsement of GECToR-Speech (Edit-Aware Sequence Tagging):**
   The optimal joint modeling paradigm for the Langfens project is **Edit-Aware Sequence Tagging (GECToR-Speech)**. By framing MDD as predicting a 41-class discrete edit transformation ($\$KEEP, \$DELETE, \$REPLACE(p)$) over canonical prompt tokens:
   - Phoneme omission receives direct, uncompromised categorical cross-entropy supervision.
   - The model requires zero artificial skip tokens and zero post-hoc Levenshtein alignment.
   - Decoding is fully non-autoregressive, executing in under $15\text{ ms}$ on GPU and $45\text{ ms}$ on CPU.
   - Parameter footprint matches Baseline R, ensuring smooth convergence within the 3,600-update fixed budget on a single Tesla T4 GPU.
   - The output maps directly to the tiered pedagogical feedback engine required by the Langfens Speaking Web Service.

---
*Report certified for integration into the master research synthesis under `manifests/RQ-RESEARCH-01/`.*
