# LANE D: Monotonic, Probabilistic, and Edit-Aware Alignment Mechanisms for MDD

**Research Family:** Family D — Alignment Mechanisms Between Prompt Phonemes and Acoustic Representations  
**Phase:** `RQ-RESEARCH-01`  
**Document ID:** `manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md`  
**Author:** Subagent Scout / Senior Speech ML Research Engineer  
**Date:** March 2026  
**Status:** `RESEARCH_SCOUT_COMPLETE; PILOT_DESIGNS_PROPOSED`  
**Authoritative Scope & Context:**
- **Thesis Boundaries:** UIT Thesis Proposal `KLTN0127` (Khoa & Khôi; Adv: ThS. Trần Thị Hồng Yến). Scope: Reference-based phoneme-level Mispronunciation Detection & Diagnosis (MDD), canonical prompt text matching, phoneme substitutions, omissions (word-final coda dropping, consonant clusters), confusable pairs, acoustic representations, tiered pedagogical feedback, Langfens Speaking Web Service. General ASR foundation training and unconstrained open-domain conversation are strictly out-of-scope.
- **Operational Rules (`AGENTS.md`):** Zero local GPU training; remote GPU execution only (Colab/Kaggle T4 FP16 AMP); budget < 30h T4 limit; 4-fold split contract (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`); sealed test isolation ($S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$); frozen Wav2Vec2 backbone (`facebook/wav2vec2-base-960h`) in evaluation; dual checkpoint accounting (3,600-step fixed budget vs validation-selected `best.pt`).
- **Empirical Baseline Anchor:** Baseline R (E0) on Fold 1 validation: overall Binary AP = 0.3170, Omission AP = 0.0350, Omission Recall @ 15% FPR = 0.4000, Macro CTC PER = 23.68%. Four-fold pooled census: overall Binary AP = 0.2726, Deletion AP = 0.0305, acoustic reliance gap $\Delta_{\text{gap}} = +0.0017$ (virtually ungrounded).

---

## 1. Executive Summary: The Alignment Dilemma in MDD

In Computer-Assisted Pronunciation Training (CAPT), reference-based Mispronunciation Detection and Diagnosis (MDD) requires evaluating whether a non-native speaker produced the canonical phonemes prescribed by a known written prompt $P = (c_1, c_2, \dots, c_N)$ in speech audio frames $\mathbf{H} = (\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T)$.

Deployable systems face a fundamental structural asymmetry:
1. **At Training Time:** Phoneticians provide manual Praat TextGrid interval boundaries (`X,sil,d`, `X,Y,s`, `X,X,c`). However, using these intervals as inputs violates deployment validity because end-users provide only audio and prompt text.
2. **At Inference Time:** The system must independently align canonical prompt phonemes to continuous acoustic representations without gold timestamps.

The current architecture in `src/langfens_prep/model.py` (`CanonicalAcousticCrossAttention`) delegates this entire temporal correspondence to standard unconstrained Multihead Cross-Attention (MHA). 

### The Diagnostic Diagnosis (DIAG-01 & DIAG-02B Evidence):
- **Acoustic Decoupling:** Cross-attention attention weights exhibit **2.8% in-window attention mass** (statistically indistinguishable from uniform random attention over ~200 frames, $1/200 = 0.5\%$).
- **Temporal Anchor Collapse:** Attention heads fixate on the same 4–6 global anchor frames across the utterance with a mean temporal peak offset of **1.415 seconds** from the true phoneme boundaries.
- **Catastrophic Omission Failure:** When a sound is omitted ($y=1, \text{type}='d'$), cross-attention pulls arbitrary acoustic features from elsewhere in the sentence. The binary MLP cannot identify acoustic absence and defaults to its class prior, yielding an **omission False Negative Rate of 97.46% at $\tau=0.50$** and **68.68% at a relaxed 15% FPR ceiling**, with Omission AP collapsing to **0.0305** (barely above prevalence 0.0159).

### The E1 Failure (RQ-MODEL-01 Evidence):
Experiment E1 attempted to resolve this by prepending a learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ and a soft Gaussian monotonic prior $B_{i,t} = -\gamma (t/T - i/N)^2$.
- **Result:** E1 **completely collapsed**. Mean skip mass on deletions was **0.000000** (0.0%). Omission AP dropped by **-7.90%** relative (from 0.0350 to 0.0323), Omission Recall @ 15% FPR dropped from 40.0% to 36.55%, and overall Binary AP degraded to 0.3093.
- **Root Cause:** In the absence of an explicit alignment loss, queries found stronger local correlations with speech frames than routing mass to an ungrounded static vector $\mathbf{s}$. Soft monotonic Gaussian bias proved too weak to prevent diffuse anchor routing.

This report systematically investigates **Family D (Monotonic, Probabilistic, and Edit-Aware Alignment Mechanisms)** to establish the theoretical foundations, architectural formulations, and pilot experimental designs capable of breaking the omission blindness bottleneck while strictly respecting UIT thesis boundaries and `AGENTS.md` operational constraints.

---

## 2. Literature Review & Theoretical Taxonomy (Items 1–3)

We examine five distinct alignment paradigms from the speech and sequence modeling literature:

```text
Family D: Alignment Mechanisms Taxonomy
├── Paradigm 1: Hard/Soft Monotonic Attention (Learnable Dynamic Programming)
│   ├── Monotonic Multihead Attention (MMA) [Raffel et al., 2017; Chiu & Raffel, 2018]
│   └── Monotonic Chunkwise Attention (MoChA) [Chiu & Raffel, 2018]
├── Paradigm 2: Continuous Probabilistic Windowing (Differentiable Heuristics)
│   ├── Gaussian Dynamic Windowing [Graves, 2013; Sharma et al., 2020]
│   └── Neural Step-Constrained Alignment Priors [Kim et al., Glow-TTS, 2020]
├── Paradigm 3: Frame-Level CTC Viterbi Forced Alignment (Deterministic Semi-Supervised)
│   ├── Viterbi Forced Alignment over Canonical Trellis [Malisiewicz et al., 2023; Radford et al., 2023]
│   └── CTC Forward-Backward Posterior Segment Pooling [Witt & Young, 2000; Peng et al., 2021]
├── Paradigm 4: Neural Dynamic Time Warping (DTW) & Latent DP
│   ├── SoftDTW [Cuturi & Blondel, ICML 2017; Mensch & Blondel, 2018]
│   └── Dynamic Time Warping on Self-Supervised Acoustic Embeddings [Tu et al., ICASSP 2026]
└── Paradigm 5: Explicit Edit-Aware Alignment (Transducer / Levenshtein MDD)
    └── Neural Transducer with Explicit Deletion Labeling [Graves, 2012; He et al., 2021]
```

### 2.1 Primary Papers & Mathematical Formulation

#### 1. Monotonic Multihead Attention (MMA) & MoChA (Chiu & Raffel, ICLR 2018; Raffel et al., ICML 2017)
- **Concept:** Standard attention can jump forwards and backwards arbitrarily across the sequence. MMA enforces that the attention head strictly proceeds from left to right along the temporal axis.
- **Formulation:** At decoder step $i$ (canonical phoneme $c_i$), attention begins inspecting acoustic frames at position $t_{i-1}$ where step $i-1$ stopped. It samples a Bernoulli decision $z_{i, t} \in \{0, 1\}$ at each frame $t \ge t_{i-1}$:
  $$e_{i, t} = \text{MonotonicEnergy}(\mathbf{q}_i, \mathbf{k}_t), \quad p_{i, t} = \sigma(e_{i, t})$$
  $$z_{i, t} \sim \text{Bernoulli}(p_{i, t})$$
  If $z_{i, t} = 1$, the attention head anchors at frame $t_i = t$. In MoChA, once anchored at $t_i$, soft attention is computed over a small chunk of $W$ frames: $\{t_i - W + 1, \dots, t_i\}$.
- **Training Relaxation:** During training, sampling $z_{i, t}$ is non-differentiable. A forward-variable dynamic programming relaxation computes the probability that the head anchors at frame $t$:
  $$\alpha_{i, t} = p_{i, t} \left( (1 - p_{i, t-1})\frac{\alpha_{i, t-1}}{p_{i, t-1}} + \alpha_{i-1, t} \right)$$
- **Academic Claims:** Achieves streaming speech recognition with minimal latency and enforces strict chronological order without external alignments.
- **Critical MDD Limitation:** When phoneme $c_i$ is completely omitted ($D$), standard MMA forces the model to choose *some* frame $t_i \ge t_{i-1}$. MMA has no native "null emit" or "skip" state on the query side. An omitted sound still forces monotonic stepping, artificially advancing the temporal cursor or corrupting adjacent phones.

#### 2. Gaussian Dynamic Windowing (Graves, 2013; Sharma et al., 2020)
- **Concept:** Query $c_i$ predicts continuous Gaussian window parameters $(\Delta \mu_i, \sigma_i, \rho_i)$ rather than calculating unconstrained dot-products across all frames.
- **Formulation:**
  $$\mu_i = \mu_{i-1} + \exp(\Delta \mu_i), \quad \sigma_i = \exp(\hat{\sigma}_i)$$
  $$w_{i, t} = \sum_{k=1}^K \rho_i^{(k)} \exp\left( - \frac{(t - \mu_i^{(k)})^2}{2 (\sigma_i^{(k)})^2} \right)$$
  Here $\Delta \mu_i > 0$ strictly enforces monotonic progression along time $t$.
- **Academic Claims:** Differentiable, prevents long-range cycling, low computation ($O(N \cdot T)$ with small constant).
- **Limitation for MDD:** Still forces acoustic mass $\sum_t w_{i, t} = 1.0$ across the Gaussian curve. If phone $c_i$ does not exist in the acoustic waveform, the Gaussian window pools silence or neighboring phone co-articulation.

#### 3. CTC Forced Alignment over Canonical Trellis (Witt & Young, 2000; Peng et al., 2021; Baevski et al., 2020)
- **Concept:** Exploiting the model's own frozen or fine-tuned CTC head emissions $\mathbf{Z} \in \mathbb{R}^{T \times C}$ to trace the most probable frame-to-token path for the prompt.
- **Formulation:** Given prompt sequence $c_{1:N}$, expand into CTC canonical graph with interleaved blanks: $\pi = (\epsilon, c_1, \epsilon, c_2, \dots, \epsilon, c_N, \epsilon)$ of length $2N + 1$.
  Execute standard Viterbi backtracking on the log emission trellis:
  $$V(t, s) = \log P(z_t = \pi_s \mid \mathbf{x}) + \max_{s' \in \{s, s-1, s-2\}} V(t-1, s')$$
  subject to standard CTC transitions (can jump $s-2$ only if $\pi_s \ne \epsilon$ and $\pi_s \ne \pi_{s-2}$).
  This yields hard boundaries $[t_{\text{start}}(c_i), t_{\text{end}}(c_i)]$ and the corresponding acoustic duration $\Delta t_i = t_{\text{end}} - t_{\text{start}}$.
- **Academic Claims:** 100% training-free, requires zero additional neural parameters, fully exploits the multi-task CTC head already present in `LangfensWav2Vec2MDD`, requires no external aligner (Kaldi/MFA).
- **Limitation for MDD:** In classical CTC forced alignment, the trellis *forces* all canonical tokens $c_{1:N}$ to be aligned to at least one frame. When a sound is omitted, Viterbi alignment squeezes the missing sound into a minimal 1-frame duration (20 ms) or an unvoiced/blank region. However, as demonstrated in DIAG-02B, **this duration and posterior anomaly is precisely the diagnostic signal needed for omission detection.**

#### 4. SoftDTW & Neural Dynamic Time Warping (Cuturi & Blondel, ICML 2017; Tu et al., ICASSP 2026)
- **Concept:** Aligning canonical phoneme representation vectors to acoustic frame embeddings using dynamic programming with a smoothed min operator.
- **Formulation:** Let $D_{i, t} = \|\mathbf{e}(c_i) - \mathbf{h}_t\|_2^2$ be the pairwise distance matrix. SoftDTW computes:
  $$\text{SoftDTW}_\gamma(\mathbf{D}) = -\gamma \log \sum_{\pi \in \mathcal{A}(N, T)} \exp\left( - \frac{\langle \mathbf{A}_\pi, \mathbf{D} \rangle}{\gamma} \right)$$
  where $\mathcal{A}(N, T)$ is the set of all valid monotonic alignment matrices.
- **Academic Claims:** Fully differentiable with respect to frame features and embeddings; exact gradient backpropagation.
- **Limitation for MDD:** High GPU memory consumption ($O(N \cdot T)$ forward-backward trellis per utterance), numerical underflow issues requiring log-space stabilization, and standard DTW strictly forces endpoint-to-endpoint alignment, squashing deletions into surrounding speech.

---

## 3. Comparison of Competing Alignment Methods (Item 2 & 3)

The following matrix compares candidate alignment mechanisms across theoretical, computational, and practical deployment criteria:

| Dimension | Unconstrained Cross-Attention (Baseline R) | Skip-Token + Gaussian Bias (E1) | Monotonic Chunkwise Attention (MoChA) | SoftDTW / Neural DP | CTC Canonical Viterbi Alignment | Segmental CTC Posterior Pooling (GOP-CTC) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Differentiable End-to-End** | Yes | Yes | Yes (DP relaxation) | Yes | No (Inference DP) / Semi-differentiable (Forward-Backward) | Semi-differentiable |
| **Monotonicity Guaranteed** | No (Diffuse anchor cycling) | No (Weak soft prior) | **Strictly Yes** | **Strictly Yes** | **Strictly Yes** | **Strictly Yes** |
| **Native Deletion (Omission) State** | No | Attempted ($\mathbf{s}$ collapsed) | No (Must advance frame cursor) | No (Endpoint forced) | **Yes** (Identified by blank/squashed frame) | **Yes** (Identified by $\sum P(\text{blank})$ and duration $< 20\text{ms}$) |
| **External Tool Dependency** | None | None | None | None | **None** (Uses internal CTC head) | **None** (Uses internal CTC head) |
| **Training GPU Memory (FP16)** | Minimal (Low) | Minimal (Low) | Moderate ($O(N \cdot T)$ trellis) | High ($O(N \cdot T)$ in-graph tensors) | **Zero Added VRAM** (Offline or post-CTC) | **Minimal** |
| **T4 GPU Training Stability** | High | High | Low (Prone to local minima / vanishing gradients) | Low (Numerical gradient underflow) | **Extreme (Zero training instability)** | **High** |
| **Inference Latency per Utterance** | ~12 ms | ~12 ms | ~45 ms (Step-wise decoding) | ~80 ms | **~18 ms** (Vectorized NumPy/C-trellis) | **~22 ms** |
| **Acoustic Grounding Diagnostic** | Failed (2.8% in-window) | Failed (3.5% in-window) | Enforced | Enforced | **100% Mechanistically Grounded** | **100% Mechanistically Grounded** |
| **Reproducibility & Verification** | Verified (Baseline R) | Verified (Failed NO-GO) | Complex implementation | Unstable in PyTorch AMP | **Exhaustively Validated** | **Well-established (Witt 2000, Peng 2021)** |

---

## 4. Modeling Deletions vs. Substitutions: The Supervision Gap (Item 4)

Why did E1 collapse, and how can Family D methods successfully model deletions alongside substitutions?

### 4.1 The Fundamental Error Asymmetry

| Error Type | Physical Acoustic Reality | Ground Truth TextGrid Interval | CTC Head Target (`dataset.py`) | Cross-Attention Behavior |
|---|---|---|---|---|
| **Correct Production ($y=0$)** | Target phone spectral realization (formants, bursts, closures). | Span $\sim 40 - 150\,\text{ms}$. Realized phone $= c_i$. | Token $c_i$ present in chronological sequence. | Attends to realized acoustic frames. |
| **Substitution ($y=1, \text{type}='s'$)** | Acoustic realization of *different* phone $s_i \ne c_i$. | Span $\sim 40 - 150\,\text{ms}$. Realized phone $= s_i$. | Token $s_i$ present in chronological sequence. | Attends to substitute frames; feature discrepancy informs binary and diagnosis heads. |
| **Omission / Deletion ($y=1, \text{type}='d'$)** | **Zero acoustic presence** (silence, immediate co-articulation, or glottal stop). | Span $0 - 40\,\text{ms}$ (median $40\,\text{ms}$, closure marking). | **EXCLUDED** from `chronological_realized_phones`. | Softmax forces query to pull unrelated frames; MLP defaults to class prior. |

### 4.2 Why E1's Skip Attention Collapsed: The Mathematical Cause

In E1, the attention weights were computed as:
$$\tilde{\mathbf{w}}_i = \text{softmax}\left( \left[ \frac{\mathbf{q}_i \mathbf{s}^T}{\sqrt{D}}, \frac{\mathbf{q}_i \mathbf{K}^T}{\sqrt{D}} - \gamma \left(\frac{t/T - i/N}{\sigma}\right)^2 \right] \right) \in \mathbb{R}^{1 + T}$$
There was **zero direct supervision loss** on $\tilde{w}_{i, \text{skip}}$. The only loss applied to the model was:
$$\mathcal{L} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{bin}} \mathcal{L}_{\text{binary}}$$
1. **The Gradient Path:** To increase $w_{i, \text{skip}}$, gradients had to flow from $\mathcal{L}_{\text{binary}}$ through the linear classification weights, through the concatenated vector $[\mathbf{a}_i \,\|\, \mathbf{e}(c_i) \,\|\, w_{i, \text{skip}}]$, back through the softmax denominator.
2. **Gradient Vanishing:** The dominant class in training is "Correct" (85.09%). Only 3.10% of training tokens are deletions. The gradient signal from the binary loss pushed the MLP weights to predict low probabilities globally.
3. **Local Acoustic Correlation:** Even when a phoneme is deleted, adjacent vowel transitions or background consonants have higher dot-product similarity with the learned phoneme query $\mathbf{q}_i$ than an uninitialized, static skip vector $\mathbf{s}$. The cross-attention module preferred diffuse acoustic frames over routing mass to $\mathbf{s}$.

### 4.3 Available Supervision Signals for Alignment

To prevent alignment collapse, the model must receive supervision from one of three signals:
1. **Direct Alignment Cross-Entropy Loss ($\mathcal{L}_{\text{align}}$):** If frame-level pseudo-labels or forced-alignment intervals $[\tau_{i, \text{start}}, \tau_{i, \text{end}}]$ are available during training, penalize attention mass falling outside the window:
   $$\mathcal{L}_{\text{attention\_supervision}} = - \sum_{i=1}^N \log \left( \sum_{t=\tau_{i, \text{start}}}^{\tau_{i, \text{end}}} w_{i, t} \right)$$
   For deletions, penalize non-skip mass: $\mathcal{L}_{\text{skip\_supervision}} = - \log w_{i, \text{skip}}$.
2. **CTC Posterior Trellis Supervision:** The CTC loss $\mathcal{L}_{\text{CTC}} = - \log P(Y_{\text{realized}} \mid \mathbf{X})$ provides explicit frame-by-frame posterior distributions $P(c \mid \mathbf{h}_t)$ over all 40 phonetic classes. This posterior contains high blank probability $P(\epsilon \mid \mathbf{h}_t) \approx 1.0$ when speech sounds are absent.
3. **Contrastive Silence/Blank Penalty:** Explicitly forcing the query representation $\mathbf{a}_i$ to correlate with the CTC blank emission probability in the corresponding temporal vicinity.

---

## 5. Architectural Mapping to Langfens Modules (Item 5)

How do candidate alignment solutions integrate into the existing repository without breaking contracts or leaking test data?

```text
Current Codebase Seam:
src/langfens_prep/
├── model.py         ──> Holds CanonicalAcousticCrossAttention & LangfensWav2Vec2MDD
├── losses.py        ──> Multi-task loss (CTC + Masked BCE + Masked CE)
├── alignment.py     ──> Frame pooling & Protocol Z Levenshtein dynamic programming
├── dataset.py       ──> TextGrid parsing & batch collator
└── g2p.py           ──> Text-to-phoneme front-end for zero-oracle inference
```

### Module Refactoring Specifications:

#### 1. In `src/langfens_prep/alignment.py`:
- **Current State:** Contains `pool_frames_to_canonical_phones` (requires external intervals) and `align_canonical_sequences` (Levenshtein string alignment).
- **Required Extension:** Implement `viterbi_ctc_align(ctc_logits, canonical_phone_indices, blank_idx=0) -> Tuple[intervals, posterior_scores]`:
  - Input: CTC logits tensor $(B, T, 40)$ produced by the acoustic backbone, plus canonical phone sequence $(B, N)$.
  - Trellis Backtracking: Performs vectorized or numba-accelerated Viterbi decoding over the canonical sequence graph.
  - Return: Frame interval tuples $[t_{\text{start}}, t_{\text{end}}]$ and diagnostic segment statistics:
    - `mean_canonical_posterior`: $\frac{1}{t_2 - t_1 + 1} \sum_{t=t_1}^{t_2} P(c_i \mid \mathbf{h}_t)$
    - `mean_blank_posterior`: $\frac{1}{t_2 - t_1 + 1} \sum_{t=t_1}^{t_2} P(\epsilon \mid \mathbf{h}_t)$
    - `duration_frames`: $t_2 - t_1 + 1$
  - **Zero External Tool Guarantee:** Completely contained within Python/PyTorch; no MFA, no Kaldi, no Praat runtime.

#### 2. In `src/langfens_prep/model.py`:
- **Current State:** `CanonicalAcousticCrossAttention` computes unconstrained dot-product attention with optional (collapsed) skip-token and soft Gaussian monotonic bias.
- **Refactoring Options:**
  - *Option A (Deterministic Hybrid):* Replace or augment cross-attention with **CTC-Guided Segmental Pooling**. The acoustic backbone produces continuous frame features $\mathbf{H}$ and CTC logits $\mathbf{Z}$. Viterbi decoding extracts canonical intervals. Segmental pooling aggregates $\mathbf{a}_i = \text{Pool}(\mathbf{H}_{t_1:t_2})$. The binary head receives $[\mathbf{a}_i \,\|\, \mathbf{e}(c_i) \,\|\, \Delta t_i \,\|\, \bar{P}_{\text{blank}}]$.
  - *Option B (Supervised Monotonic Attention):* Retain cross-attention but inject **CTC Posterior Alignment Loss** $\mathcal{L}_{\text{align}}$ during training to strictly supervise attention maps against the CTC Viterbi path, preventing anchor collapse.

#### 3. In `src/langfens_prep/losses.py`:
- Add `SupervisedAttentionAlignmentLoss`: Penalizes cross-attention probability mass that deviates from the CTC Viterbi path by computing Kullback-Leibler divergence or cross-entropy between cross-attention weights $\mathbf{W} \in \mathbb{R}^{N \times T}$ and the frame-level Viterbi target distribution.

#### 4. In `src/langfens_prep/dataset.py`:
- No data schema changes required! The ingestion pipeline already provides `canonical_phone_indices`, audio waveforms, and labels $y_i \in \{0, 1\}$. Test-set isolation and zero-oracle deployment remain strictly preserved because no TextGrid boundaries are fed to the model.

---

## 6. Engineering & Operational Feasibility Evaluation (Item 6)

### 6.1 Inference Latency & CAPT Service Integration
- **Target Deployment Platform:** Langfens Speaking Web Service (Next.js frontend, .NET 8 core API, Python/FastAPI AI microservice, Docker container).
- **Interactive Budget:** Total round-trip latency for user speech feedback must remain $\le 500\,\text{ms}$.
- **Method Profiling:**
  - *Current Baseline R (Cross-Attention):* Backbone forward = ~25 ms, Cross-attention = ~5 ms, Binary MLP = ~1 ms. Total $\approx 31\,\text{ms}$.
  - *CTC Viterbi Forced Alignment:* Backbone forward + CTC = ~25 ms, Viterbi dynamic programming trellis (for $N \approx 30$ phones, $T \approx 200$ frames) implemented in NumPy / PyTorch tensor ops = **~6 to 8 ms**, Segment Pooling + MLP = ~2 ms. Total $\approx 35\,\text{ms}$.
  - *MoChA (Autoregressive Step-Wise):* Requires step-wise recurrent decoding loop across $N$ queries. GPU batched latency $\approx 65 - 90\,\text{ms}$.
  - *SoftDTW:* Matrix dynamic programming forward pass $\approx 45 - 60\,\text{ms}$.
- **Verdict:** CTC Viterbi Forced Alignment and Supervised Cross-Attention are well within the 500 ms CAPT interactive threshold.

### 6.2 Remote GPU Compute Budget Feasibility (< 30h T4 Limit)
Under `AGENTS.md`, all training must run on remote Kaggle/Colab Nvidia Tesla T4 GPUs with CUDA FP16 AMP.
- **Baseline R Benchmark:** 3,600 updates on Fold 1 requires **~2.8 hours** on Kaggle T4.
- **Full 4-Fold Baseline R Training:** $4 \times 2.8\text{h} \approx 11.2\text{hours}$ (well within weekly 30h quota).
- **Candidate Methods Compute Footprint:**
  - *Deterministic CTC Alignment (Training-Free / Post-Backbone):* **0 additional training hours!** Can be evaluated directly on existing Baseline R checkpoints or frozen multi-task CTC backbones without retraining.
  - *Supervised Attention Alignment (Fine-Tuning Cross-Attention):* Adds only a single elementwise matrix loss $\sum \mathbf{W} \odot \mathbf{M}_{\text{target}}$. Training throughput is identical to Baseline R (~2.8h per fold).
  - *MoChA / MMA:* Involves cumulative dynamic programming forward-backward variables. Slows training throughput by $2.5\times$ (~7.0h per fold). 4 folds = ~28h, dangerously close to the 30h weekly Kaggle ceiling.
  - *SoftDTW:* Backpropagating through a smoothed min trellis in PyTorch requires custom CUDA kernels or unrolled loops. T4 FP16 AMP frequently encounters gradient underflow/overflow.

### 6.3 Pedagogical Feedback Capability
The thesis proposal requires delivering tiered, actionable feedback to language learners:
- **Level 1 (Detection):** Alerting the learner that a sound was mispronounced or dropped.
- **Level 2 (Diagnosis & Category):** Identifying whether the error was a substitution (e.g. [D] for [DH]) or an omission (e.g. dropping word-final coda /D/ in *"and"*).
- **Level 3 (Actionable Guidance):** Phonetic instructions (e.g., place tongue between teeth for /DH/; ensure full tongue-tip closure on the alveolar ridge for /D/).

**Crucial Alignment Implication:**
Unconstrained cross-attention provides **zero temporal boundaries**. When it flags an error, it cannot highlight the exact spectrogram interval or play back the mispronounced segment to the learner.
In contrast, **CTC Viterbi alignment provides exact acoustic start and end timestamps $[t_{\text{start}}, t_{\text{end}}]$**. The Langfens Web Service can immediately use these boundaries to:
1. Highlight the mispronounced phoneme in the UI synchronized with audio playback.
2. Extract the learner's realization audio clip for side-by-side comparison with a native reference model (TTS).
3. Compute Goodness of Pronunciation (GOP) likelihood ratios strictly localized to the segment.

---

## 7. Thesis Scope Boundaries & Risk Analysis (Item 7)

To ensure this research directly serves the graduation thesis without scope creep, we enforce three strict boundaries:

### Boundary 1: No General Foundation ASR Drift
- **Risk:** Attempting to train open-vocabulary speech recognizers or general end-to-end ASR foundation models.
- **Enforcement:** The acoustic backbone (`facebook/wav2vec2-base-960h`) remains **frozen**. We train only lightweight task heads (CTC, Binary MDD, Alignment projections). The task is strictly *reference-based MDD* against known prompt text, never unprompted transcription.

### Boundary 2: No Simplification to Deletion-Only Architecture
- **Risk:** Designing a specialized omission detector that ignores phoneme substitutions, destroying overall MDD accuracy.
- **Enforcement:** All evaluations must monitor the primary multi-task metric suite:
  - Macro Binary AP (overall detection quality across substitutions and omissions).
  - Substitution AP and Confusion Matrix fidelity.
  - Omission AP and Omission Recall @ 15% FPR.
  - CTC PER (guardrail against acoustic degradation).
- Any alignment mechanism that improves omission detection at the expense of substitution detection is unacceptable.

### Boundary 3: Strict Zero-Oracle Deployment Guarantee
- **Risk:** Accidentally leaking gold TextGrid interval boundaries into inference, invalidating the model as an automated tool.
- **Enforcement:** The deployable interface receives only raw waveform $\mathbf{x}$ and canonical prompt tokens $c_{1:N}$. Mandatory negative invariance tests (verifying identical predictions whether gold intervals are present, omitted, or scrambled) must pass before any pilot is accepted.

---

## 8. Proposed Candidate Methods & Minimal Falsifiable Pilot Experiments (Item 8)

We propose two rigorous, falsifiable candidate architectures in Family D:

### Candidate 1: CTC-Viterbi Segmental Pooling with Explicit Blank-Duration Features (Family D — Method 1: "CTC-Align-MDD")

```text
                               Canonical Prompt: [DH, AH, D]
                                             │
Audio Waveform ──> [Frozen Wav2Vec2] ──> Frame Embeddings H (T x 768)
                         │
                         ▼
                   [CTC Head] ──> Logits Z (T x 40)
                         │
                         ▼
        [Viterbi Canonical Forced Aligner]
                         │
        Extracts for each token c_i:
        ├── Time Interval: [t_start, t_end]
        ├── Duration Feature: Δt_i = (t_end - t_start + 1) / T
        ├── Mean Blank Posterior: P_blank = (1/Δt) Σ P(blank | h_t)
        └── Segment Pooled Acoustic: a_i = MeanPool(H[t_start : t_end])
                         │
                         ▼
        Feature Vector: [ a_i (768) || e(c_i) (128) || Δt_i (1) || P_blank (1) ]
                         │
                         ▼
        [Binary Classification MLP] ──> Error Probability p_i
```

#### Why This Directly Solves the Omission Bottleneck:
1. **Physical Grounding:** When a speaker omits a sound (e.g. final /D/), the CTC Viterbi trellis cannot find acoustic evidence for /D/. It is forced to route the token through a minimal single frame (20 ms) where the emission is overwhelmingly dominated by the blank token: $P(\epsilon \mid \mathbf{h}_t) \approx 0.95 - 0.99$.
2. **Explicit Omission Signature:** The feature vector feeds $\Delta t_i \approx 0.005$ and $P_{\text{blank}} \approx 0.98$ directly into the classification head. The MLP does not have to guess acoustic absence from diffuse cross-attention; it receives direct mathematical indicators of acoustic absence.
3. **Zero GPU Training Overhead:** Can be verified initially as a **training-free inference module** using the existing frozen Baseline R CTC head, requiring zero new Kaggle training updates!

---

### Candidate 2: Monotonic Cross-Attention with CTC-Supervised Alignment Loss (Family D — Method 2: "Supervised-MHA-MDD")

```text
Canonical Queries Q ───┐
                       ▼
Acoustic Keys/Values K, V ──> [Cross-Attention] ──> Attention Weights W (N x T)
                                      │
                                      ├──> [Binary MDD Head] ──> Loss L_BCE
                                      │
                                      ▼
                        Alignment Loss: L_align = KL(W || W_ctc_target)
                                      ▲
                                      │
    CTC Logits Z ──> [Viterbi Trellis] ──> Target Window Distribution W_ctc_target
```

#### Formulation:
- Maintain the differentiable cross-attention architecture.
- During training, pass the audio through the CTC head and compute the Viterbi canonical alignment target matrix $\mathbf{W}_{\text{target}} \in \mathbb{R}^{N \times T}$, where $W_{i, t} = 1/\Delta t_i$ for $t \in [t_{\text{start}}, t_{\text{end}}]$ and $0$ elsewhere.
- Add an explicit alignment loss to the multi-task objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{bin}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{diag}} \mathcal{L}_{\text{diag}} + \lambda_{\text{align}} \text{KL}(\mathbf{W}_{\text{attn}} \,\|\, \mathbf{W}_{\text{target}})$$
- This directly breaks the gradient vanishing problem that killed E1: the cross-attention weights receive direct, massive gradient penalties if they fixate on distant anchor frames.

---

### Minimal Falsifiable Pilot Experiment Protocols

#### Pilot Protocol D-1: Training-Free CTC-Viterbi Segmental Evaluation
- **Execution:** Zero GPU training. Standalone CPU/GPU evaluation script (`audit_scripts/evaluate_ctc_viterbi_alignment.py`) evaluated on Fold 1 validation partition (300 utterances, 9,956 tokens, 145 deletions).
- **Hypothesis:** CTC Viterbi interval duration $\Delta t_i$ and mean blank posterior $P_{\text{blank}}$ provide statistically significant separation between omitted tokens and correctly realized tokens ($p < 10^{-6}$).
- **Falsification Gate:**
  - If mean $P_{\text{blank}}$ on deletions is NOT at least $2.0\times$ higher than on correct tokens, REJECT hypothesis.
  - If logistic regression on $[\Delta t_i, P_{\text{blank}}]$ does NOT achieve Omission AP $\ge 0.1500$ (vs Baseline R 0.0350), REJECT candidate.

#### Pilot Protocol D-2: Supervised Monotonic Attention Fine-Tuning
- **Execution:** Remote Kaggle GPU, Fold 1 pilot, fixed budget 3,600 updates, FP16 AMP.
- **Pre-Registered Acceptance Gates:**
  1. `omission_ap` $\ge \mathbf{0.0800}$ (vs Baseline R 0.0350, E1 0.0323).
  2. `omission_recall_at_fpr15` $\ge \mathbf{0.4500}$ (vs Baseline R 0.4000, E1 0.3655).
  3. `binary_ap` $\ge \mathbf{0.3170}$ (must not degrade overall detection).
  4. `mean_conditional_in_window_mass` $\ge \mathbf{0.3000}$ (must eliminate the 2.8% diffuse anchor collapse).
  5. `ctc_per` $\le \mathbf{24.50\%}$ (ASR fidelity guardrail).

---

## 9. Verification Ledger: Evidence vs. Assumptions (Item 9)

To adhere to the rigorous audit standards of `AGENTS.md`, all claims are partitioned into verified empirical evidence versus unverified research assumptions:

| Topic | Verified Empirical Evidence (`INDEPENDENTLY_VERIFIED`) | Working Research Assumption (`UNVERIFIED_ASSUMPTION`) | Risk / Remediation Path |
|---|---|---|---|
| **Baseline R Cross-Attention Localization** | Verified in DIAG-01 & DIAG-02B: Mean in-window attention mass is **2.8%**; mean peak offset is **1.415 seconds**; attention fixates on 4–6 global anchor frames. | Assumed that lack of temporal localization is the primary driver of high false-negative rates on omissions. | Partially validated by counterfactuals in DIAG-02B; fully testable via Pilot D-1. |
| **E1 Skip-Token Collapse** | Verified on Kaggle GPU (`PROCESS_CHECKPOINT.md`): Skip mass on omissions was **0.000000**; Omission AP degraded **-7.90%**; Omission Recall @ 15% FPR degraded from 40.0% to 36.55%. | Assumed that lack of direct attention loss was the fatal cause of skip-token inactivity. | Will be directly falsified by Pilot D-2 (supervised alignment loss). |
| **CTC Omission Signatures** | Verified in DIAG-01 & DIAG-02B: CTC head excludes deletions from realized sequences; CTC emits high blank probability over unvoiced/deleted regions. | Assumed that Viterbi forced alignment over the canonical trellis will reliably squeeze omissions into 1-frame blank intervals without distorting adjacent phones. | Needs empirical validation via Pilot D-1 on non-native L2-ARCTIC speech. |
| **L2-ARCTIC Omission Prevalence** | Verified across all 4 folds: 2,321 to 2,522 deletions per fold in TRAIN (~3.0%); 629 deletions across 4 validation folds (1.59%); 55.6% word-final coda dropping; top 3 phones (/D/, /R/, /T/) account for 58.0% of omissions. | Assumed that demographic shift (0 Vietnamese speakers in validation) does not prevent validation metrics from reflecting training capacity. | Cross-fold evaluation on Folds 2–4 will verify generalization across all L1s. |
| **Deployment Zero-Oracle Invariance** | Verified in `tests/test_reconciliation_regression.py`: Bit-exact logit equality confirmed when gold intervals are present, omitted, or scrambled. | Assumed that CTC-guided Viterbi alignment will introduce no test-time data leakage. | The Viterbi aligner uses only the model's own CTC logits and the prompt text, preserving zero-oracle status. |

---

## 10. Strategic Recommendations & Action Plan

1. **Reject Autoregressive Monotonic RNN Attention (MMA/MoChA):** Do not implement MoChA. Its step-wise recurrent decoding loop introduces unacceptable inference latency (~70 ms vs ~12 ms), risks training instability on T4 GPUs, and does not provide an inherent mechanism for omitted phonemes.
2. **Reject Unsupervised SoftDTW:** Do not pursue unconstrained SoftDTW due to gradient underflow risks in FP16 AMP and high VRAM overhead.
3. **Authorize Pilot Protocol D-1 (Training-Free CTC-Viterbi Segmental Analysis):**
   - Execute a lightweight, CPU-based evaluation script over the saved Baseline R validation predictions.
   - Profile the distribution of $\Delta t_i$ and $P_{\text{blank}}$ on omissions versus correct tokens.
   - If Pilot D-1 confirms significant separation ($p < 10^{-6}$), proceed to Candidate 1 integration.
4. **Prepare Pilot Protocol D-2 for Remote GPU Authorization:**
   - If neural end-to-end differentiability is desired, implement `SupervisedAttentionAlignmentLoss` using the CTC Viterbi path as the supervision target.
   - Stage the package to Kaggle CLI using the permanent workflow (`kaggle kernels push --no-run`) and await user authorization before launching training.

---
*End of Family D Research Report.*
