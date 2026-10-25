# LANGFENS — RQ1-TRN-01A: Model Architecture Implementation Report

**Phase:** `RQ1-TRN-01A` — Training Implementation & GPU Smoke Test  
**Focus Area:** Phase 4 — Wav2Vec2 MDD Architecture Implementation & Interface Verification  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Reviewer  
**Deliverable Path:** `manifests/RQ1-TRN-01A/MODEL_ARCHITECTURE_REPORT.md`  
**Implemented Module:** `src/langfens_prep/model.py` and `src/langfens_prep/alignment.py`  
**Status:** `MODEL_ARCHITECTURE_VALIDATED: PASS`  

---

## 1. Executive Summary & Architectural Overview

The primary acoustic model architecture for Project Langfens is implemented in `src/langfens_prep/model.py` as `LangfensWav2Vec2MDD`.

The architecture addresses the core challenge of phoneme-level MDD: non-native speech involves asynchronous temporal shifts (substitutions, elisions, and epenthesis). To handle this without fragile external forced alignment, the model integrates:
1. An **acoustic CTC recognition head** that models the continuous sequence of realized speech sounds,
2. A **token-level binary mispronunciation detection head** conditioned on the prompt's canonical phoneme identity, and
3. A **token-level substitution diagnosis head** classifying the specific realized sound across 39 ARPAbet classes.

```text
LangfensWav2Vec2MDD Architecture Schematic
                     Raw Audio Waveform (16 kHz)
                                │
        ┌───────────────────────▼───────────────────────┐
        │  7-Layer Conv Feature Extractor (320x Stride) │  (Freezable via freeze_feature_encoder)
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │   Positional Conv Embedding + LayerNorm       │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │       Transformer Context Network (Pre-Norm)  │  (Supports gradient checkpointing)
        └───────────────────────┬───────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   CTC Head    │   │  Frame-Phone  │   │  Frame-Phone  │
    │  (40 Classes) │   │ Time Pooling  │   │ Time Pooling  │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   ▼                   ▼
            │           ┌───────────────┐   ┌───────────────┐
            │           │  Concat with  │   │   Diagnosis   │
            │           │ Canonical Emb │   │     Head      │
            │           └───────┬───────┘   │ (39 Classes)  │
            │                   ▼           └───────┬───────┘
            │           ┌───────────────┐           │
            │           │  Binary Head  │           │
            │           │   (1 Logit)   │           │
            │           └───────┬───────┘           │
            ▼                   ▼                   ▼
        CTC Loss            Masked BCE          Masked CE
        (Blank = 0)     (ignore_index=-100) (ignore_index=-100)
```

---

## 2. Component Specifications

### 2.1 7-Layer Convolutional Feature Extractor
- **Class:** `Wav2Vec2FeatureEncoder`
- **Specification:** Seven 1D convolutional layers with GroupNorm and GELU activations:
  - Layer 1: kernel 10, stride 5, 512 channels
  - Layer 2: kernel 3, stride 2, 512 channels
  - Layer 3: kernel 3, stride 2, 512 channels
  - Layer 4: kernel 3, stride 2, 512 channels
  - Layer 5: kernel 3, stride 2, 512 channels
  - Layer 6: kernel 2, stride 2, 512 channels
  - Layer 7: kernel 2, stride 2, 512 channels
- **Total Temporal Downsampling:** $5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = \mathbf{320\times}$.
- **Output Frame Rate:** At $16,000 \text{ Hz}$, each frame corresponds to $20 \text{ ms}$ of audio ($50 \text{ frames/second}$).
- **Freezing Interface:** `model.freeze_feature_encoder()` sets `requires_grad = False` on all conv parameters, preventing catastrophic forgetting during initial fine-tuning.

### 2.2 Relative Positional Encoding & Transformer Context Network
- **Positional Encoding:** Depthwise 1D convolution with kernel size 128, 16 groups, and weight normalization (`PositionalConvEmbedding`).
- **Transformer Encoder:** Pre-norm Transformer Encoder blocks with GELU activation.
  - Smoke / Unit Test Configuration: 2–4 layers, 256 hidden dimension, 4 attention heads.
  - Production Base Configuration: 12 layers, 768 hidden dimension, 8 attention heads, 2048 feed-forward dimension.
- **Memory Optimization:** `model.gradient_checkpointing_enable()` activates non-reentrant activation checkpointing during training forward passes.

### 2.3 Multi-Task Output Heads
1. **Head 1 — CTC Realized Phoneme Head:**
   - Architecture: `nn.Linear(hidden_dim, 40)`.
   - Classes: Index 0 = `<BLANK>`, indices 1..39 = standard ARPAbet phonemes.
   - Loss Function: `nn.CTCLoss(blank=0, zero_infinity=True)`.
2. **Head 2 — Canonical Binary Detection Head:**
   - Architecture: Token-level MLP:
     $$\text{Linear}(D_{\text{hidden}} + D_{\text{embed}}, 256) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(256, 1)$$
   - Input: Pooled acoustic frame concatenated with 128-dim canonical prompt phone embedding (`nn.Embedding(40, 128)`).
   - Loss Function: `MaskedBCEWithLogitsLoss(ignore_index=-100)`.
3. **Head 3 — Substitution Diagnosis Head:**
   - Architecture: Token-level MLP:
     $$\text{Linear}(D_{\text{hidden}}, 256) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(256, 39)$$
   - Output: Unnormalized logits over the 39 standard ARPAbet classes.
   - Loss Function: `MaskedCrossEntropyLoss(ignore_index=-100)`.

---

## 3. Frame-to-Token Time-Aligned Pooling (`alignment.py`)

To evaluate token-level MDD on canonical prompt phones, continuous acoustic frames must be pooled into discrete phone representations:

- **Module:** `src/langfens_prep/alignment.py::pool_frames_to_canonical_phones`
- **Mapping Formulation:** Given start time $t_{\text{start}}$ and end time $t_{\text{end}}$ in seconds:
  $$\text{frame}_{\text{start}} = \left\lfloor \frac{t_{\text{start}}}{\text{stride}} \right\rfloor, \quad \text{frame}_{\text{end}} = \left\lceil \frac{t_{\text{end}}}{\text{stride}} \right\rceil$$
  Where $\text{stride} = \frac{320}{16,000} = 0.020 \text{ s}$ ($20 \text{ ms}$).
- **Boundary Guarantees:**
  1. Clamped strictly to $[0, T_{\text{frames}}]$.
  2. Guarantees $\text{frame}_{\text{end}} \ge \text{frame}_{\text{start}} + 1$, ensuring **every phoneme receives at least 1 acoustic frame** even if duration $< 20\text{ ms}$.
  3. Supports mean-pooling and max-pooling modes.

---

## 4. Multi-Task Loss Objective

The combined training objective optimizes joint phoneme recognition, detection, and diagnosis:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{ctc}} \mathcal{L}_{\text{CTC}} + \lambda_{\text{bin}} \mathcal{L}_{\text{Masked-BCE}} + \lambda_{\text{diag}} \mathcal{L}_{\text{Masked-CE}}$$

Default loss weights:
- $\lambda_{\text{ctc}} = 1.0$ (drives acoustic feature alignment),
- $\lambda_{\text{bin}} = 1.0$ (drives binary mispronunciation detection),
- $\lambda_{\text{diag}} = 0.5$ (drives substitute phone classification).

---

## 5. Checkpoint Serialization & State Dict Schema

Model weights are serialized cleanly via PyTorch:
```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": {
        "hidden_dim": model.hidden_dim,
        "ctc_weight": model.ctc_weight,
        "binary_weight": model.binary_weight,
        "diagnosis_weight": model.diagnosis_weight
    }
}
torch.save(checkpoint, "checkpoints/best_model.pt")
```
Roundtrip loading was verified in `tests/test_model.py::test_checkpoint_serialization`.

**Phase Gate Evaluation:** **`MODEL_ARCHITECTURE_VALIDATED: PASS`**
