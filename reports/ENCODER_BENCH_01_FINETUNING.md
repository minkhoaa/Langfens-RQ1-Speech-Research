# LANGFENS RQ1-ENCODER-BENCH-01 — Encoder Adaptation & Fine-Tuning Monograph

**Campaign:** `LANGFENS-RQ1-ENCODER-BENCH-01`  
**Lane:** Agent B — Encoder Adaptation & Fine-Tuning  
**Standard:** EVAL-V3.1, 4-fold Speaker-Grouped TRAIN OOF (1,800 utterances, 12 speakers)  
**Registered Seed:** 101  
**Hardware:** NVIDIA GeForce GTX 1650 (4GB VRAM) / Intel Core i5-12450H  

---

## 1. Executive Summary & Core Findings

Agent B conducted a controlled empirical investigation into **Hypothesis H2**:
> *Task adaptation makes Wav2Vec2 representations more discriminative for mispronunciation detection (MDD).*

Using candidates selected and fitted strictly on inner speaker-held-out folds with zero outer-test exposure, we compared:
1. **Frozen Encoder Baseline:** Standard frozen layer-6 representation from `facebook/wav2vec2-base-960h`.
2. **Learnable Layer-Weighted Mixture (Scalar Mixing):** Differentiable softmax-weighted combination across all 13 transformer layers ($l \in \{0, \dots, 12\}$) with a learnable scalar multiplier $\gamma$.
3. **Parameter-Efficient Bottleneck Residual Adapter:** A low-rank residual adapter ($2304 \to 256 \to 2304$, 1.8M params) inserted between the acoustic representation and MDD classification heads.
4. **Capacity-Matched MLP Control:** A widened classification head ($2343 \to 600 \to 300 \to 3$, 1.6M params) to isolate whether adaptation gains stem from acoustic feature transformation or raw capacity expansion.
5. **Upper-Layer Unfreezing Feasibility:** Profiling memory, gradient flow, and training dynamics of unfreezing upper encoder layers.

### Headline Results

| Candidate | Architecture / Strategy | Parameters | Macro-3 AP | AP-S | AP-O | AP-I | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| **Frozen L6 Baseline** | Single Layer 6 + Standard Head | 633 K | **0.241297** | 0.355160 | 0.226723 | 0.142008 | 0.128257 | 0.2733 | Matched Control |
| **Layer-Weighted Mixture** | Softmax Mixing (13 layers) | 633 K + 28 | **0.277558** | **0.389407** | **0.280638** | **0.162630** | **0.157628** | 0.2806 | **+0.0363 Gain (+15.0%)** |
| **Bottleneck Adapter** | $2304 \to 256 \to 2304$ ResAdapter | 1,815 K | **0.187414** | 0.311947 | 0.167397 | 0.082897 | 0.077940 | 0.2489 | **Overfitting (-0.0539)** |
| **Capacity-Matched MLP** | Widened Head ($600 \to 300$) | 1,588 K | **0.252620** | 0.374010 | 0.249450 | 0.134400 | 0.135108 | 0.2722 | +0.0113 (Diminishing) |

---

## 2. Methodology & Experimental Design

### 2.1 Learnable Layer-Weighted Mixture (Scalar Mixing)
In speech self-supervised learning (SUPERB benchmark standard), different transformer layers encode different linguistic hierarchies: lower layers capture acoustic/spectral formants, middle layers capture phonetic boundaries and articulatory features, and upper layers encode lexical/semantic context.

Rather than committing arbitrarily to a single layer, scalar mixing defines the composite acoustic feature $\mathbf{h}_{\text{mixed}}$ as:
$$\mathbf{h}_{\text{mixed}} = \gamma \sum_{l=0}^{12} w_l \mathbf{h}^{(l)}, \quad w_l = \frac{\exp(\alpha_l)}{\sum_{k=0}^{12} \exp(\alpha_k)}$$
where $\boldsymbol{\alpha} \in \mathbb{R}^{13}$ and scalar scale $\gamma \in \mathbb{R}$ are learnable parameters. The composite representation is concatenated with canonical phone reference embeddings $\mathbf{r} \in \mathbb{R}^{39}$ and fed into the standard H-002 target head. An identical, independent scalar mixture is trained for the MB03 gap insertion head.

### 2.2 Bottleneck Residual Adapter
Adapters have shown high parameter-efficiency in NLP and speech adaptation. We implemented a Houlsby-style residual bottleneck block directly on the 2304-dimensional target acoustic crop:
$$\mathbf{h}_{\text{adapted}} = \mathbf{h} + W_{\text{up}} \cdot \text{GELU}(W_{\text{down}} \mathbf{h})$$
where $W_{\text{down}} \in \mathbb{R}^{256 \times 2304}$ and $W_{\text{up}} \in \mathbb{R}^{2304 \times 256}$, with a matching adapter ($1536 \to 128 \to 1536$) for the gap head.

### 2.3 Capacity-Matched Control
To rigorously answer whether performance changes are attributable to representation adaptation rather than parameter count, we widened the standard classification MLP from $[256, 128]$ to $[600, 300]$, matching the adapter's parameter footprint (1.59M vs 1.82M).

---

## 3. Detailed Results & Component Attribution

### 3.1 Learned Layer Weights Distribution
The learned layer weights $\mathbf{w}$ from the 4 outer folds reveal the empirical information distribution across Wav2Vec2 depths:

| Layer $l$ | Mean Weight $w_l$ | Cumulative Mass | Functional Interpretation |
|---|---|---|---|
| **Layer 0 (CNN Output)** | 0.064 | 6.4% | Raw waveform spectral representations |
| **Layer 1** | 0.070 | 13.4% | Local acoustic context |
| **Layer 2** | 0.073 | 20.7% | Early subphonemic transitions |
| **Layer 3** | 0.075 | 28.2% | Coarticulation modeling |
| **Layer 4** | 0.084 | 36.6% | Articulatory feature onset |
| **Layer 5** | 0.099 | 46.5% | Phonetic identity & vowel quality |
| **Layer 6 (Phase-01 Single Layer)** | **0.104** | 56.9% | **Peak phonetic discriminability** |
| **Layer 7** | 0.099 | 66.8% | Phoneme boundary contrast |
| **Layer 8** | 0.095 | 76.3% | Broad phonological category |
| **Layer 9** | 0.082 | 84.5% | Sub-lexical transitions |
| **Layer 10** | 0.059 | 90.4% | Lexical constraints begin |
| **Layer 11** | 0.055 | 95.9% | ASR character decoding context |
| **Layer 12 (Final Layer)** | 0.042 | 100.0% | Strict ASR token projection |

#### Key Insights from the Layer Weighting Profile:
1. **Middle Layer Concentration:** Layers 4 through 8 account for **48.1% of the total attention mass**. Layer 6 remains the individual mode ($w_6 = 0.104$), validating the Phase-01 intuition that Layer 6 is the strongest single layer.
2. **Complementarity of Flanking Layers:** Single-layer extraction discards subphonemic acoustic details from layers 4–5 and phonological context from layers 7–8. By fusing these depths, AP-Substitution increases from 0.3552 to **0.3894 (+0.0342)** and AP-Omission increases from 0.2267 to **0.2806 (+0.0539)**.
3. **Severe Downweighting of Top Layers:** Layers 10–12 are systematically penalized ($w_{12} = 0.042$). Because Wav2Vec2-base-960h was fine-tuned with CTC loss to predict native English orthography, top layers become invariant to non-native phonetic deviations—treating non-native substitutions as native phonetic variants. Layer weighting automatically suppresses this unwanted invariance!

### 3.2 The Adapter Overfitting Failure Mode
While layer weighting succeeded (+0.0363 AP), the Bottleneck Adapter suffered a catastrophic drop to **0.1874 Macro-3 AP** (-0.0539 below baseline).

#### Failure Mechanism:
- **Data Sparsity vs Adapter Capacity:** In MDD, positive error events are severely sparse (only 7,381 substitutions, 1,825 omissions, and 560 insertions across 1,800 training utterances).
- Adding 1.8M parameters in an unconstrained bottleneck adapter caused the network to memorize speaker-specific non-native pronunciation idiosyncrasies during inner fitting.
- When evaluated on held-out speakers during outer cross-validation, the adapted features failed to generalize, producing noisy scores and degrading ranking monotonicity.
- Conversely, the Capacity-Matched MLP control ($0.2526$) performed modestly better than baseline ($0.2413$), but far below layer weighting ($0.2776$).
- **Conclusion:** MDD performance is **not limited by classification head capacity**. Adding parameters to the head or adapter yields diminishing or negative returns. The bottleneck is the **multi-scale phonetic representation**, which is effectively addressed by layer weighting without adding parameter capacity.

---

## 4. Compute, Memory, and Runtime Profile

| Metric | Frozen L6 Baseline | Layer-Weighted Mixture | Bottleneck Adapter | Capacity-Matched MLP | Upper-Layer Unfreezing Profile |
|---|---|---|---|---|---|
| **Trainable Params** | 633,734 | 633,762 (+28) | 1,815,555 (+1.18M) | 1,587,603 (+954K) | 19,292,544 (20.4% backbone) |
| **Peak VRAM** | 126 MB | 346 MB | 150 MB | 138 MB | 531 MB |
| **GPU Utilization** | 67% | 44% | 58% | 61% | 88% |
| **Training Time (16 fits)** | 44.9s | 321.6s | 78.3s | 63.6s | ~1,850s (projected) |
| **Feasibility Assessment** | Trivially Feasible | Fully Feasible | Fully Feasible | Fully Feasible | Feasible (High Compute) |

---

## 5. Architectural Recommendations for Phase-03

1. **Adopt Learnable Layer Weighting in Phase-03:**
   - Multi-layer scalar mixing improves Macro-3 AP by $+0.0363$ without risking overfitting.
   - It requires only 28 additional scalar parameters, preserving model compactness.
2. **Reject Heavy Bottleneck Adapters:**
   - Unconstrained nonlinear adapters overfit to training speakers on small L2 corpora.
3. **Keep ASR Orthographic Top Layers Suppressed:**
   - Retain frozen backbone representations with layer weighting, ensuring top layers ($l \ge 10$) cannot overwrite acoustic phonetic cues.
