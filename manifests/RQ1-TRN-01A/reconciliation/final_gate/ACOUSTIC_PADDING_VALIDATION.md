# LANGFENS — RQ1-TRN-01A-R2: Acoustic Padding Validation Report

**Gate Reference:** RQ1-TRN-01A-R2 — Final Training Correctness & Evidence Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01A/reconciliation/final_gate/ACOUSTIC_PADDING_VALIDATION.md`  
**Target Architecture:** Full 12-Layer Wav2Vec 2.0 Base MDD (`facebook/wav2vec2-base-960h` @ `22aad52d`)  
**Lead Auditor / Speech ML Engineer:** Senior Speech ML Research Engineer & PyTorch Architect  
**Gate Evaluation Status:** `ACOUSTIC_PADDING_VALIDATION: PASS`  

---

## 1. Executive Summary & Problem Formulation

An independent audit of `LangfensWav2Vec2MDD` in phase `RQ1-TRN-01A-R` identified that the forward call:
```python
self.wav2vec2(waveforms)
```
did not explicitly apply available waveform durations (`audio_lengths`) or boolean padding masks (`padding_mask`) to the acoustic backbone.

In variable-duration speech recognition batches, utterances are padded with trailing zeros to match the maximum batch duration $T_{\text{max}}$. While standard text transformers handle padding via attention masks, convolutional speech encoders require careful architectural analysis.

### Primary Audit Requirements:
1. Determine the correct padding policy for `facebook/wav2vec2-base-960h` accounting for feature extractor normalization.
2. Avoid blindly passing `attention_mask` without compatibility verification.
3. Guarantee that variable-duration padded batches cannot contaminate valid acoustic representations.
4. Preserve exact acoustic output lengths.
5. Ensure cross-attention masks invalid acoustic frames.
6. Verify output logits/probabilities for an utterance evaluated alone match evaluation inside a variable-length padded batch within single-precision floating-point tolerance ($\le 10^{-5}$).
7. Ensure canonical padding is excluded from prediction and loss aggregation.

---

## 2. Architectural Root-Cause Analysis: GroupNorm Contamination

### 2.1 Upstream Checkpoint Configuration
Inspection of `Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")` reveals:
- `feat_extract_norm = "group"`
- The 7-layer convolutional waveform feature extractor has the following layer 0 structure:
  ```text
  Wav2Vec2GroupNormConvLayer(
    (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
    (activation): GELUActivation()
    (layer_norm): GroupNorm(512, 512, eps=1e-05, affine=True, bias=True)
  )
  ```

### 2.2 Mathematical Mechanism of Corruption
In PyTorch, `nn.GroupNorm(num_groups=512, num_channels=512)` normalizes each individual channel across its temporal spatial dimension:
$$\mu_c = \frac{1}{T} \sum_{t=1}^T x_{c, t}, \quad \sigma_c^2 = \frac{1}{T} \sum_{t=1}^T (x_{c, t} - \mu_c)^2, \quad \hat{x}_{c, t} = \frac{x_{c, t} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

When variable-duration speech waveforms are zero-padded to $T_{\text{max}}$, the trailing zeros produce near-zero convolutional responses for $t > T_{\text{valid}}$. When `GroupNorm` reduces across all $T_{\text{max}}$ time steps:
1. The denominator $T$ includes padded time steps.
2. The empirical channel mean $\mu_c$ is shifted toward zero.
3. The empirical variance $\sigma_c^2$ is suppressed by the zero tail.
4. As a direct consequence, **every valid time step $t \le T_{\text{valid}}$ is scaled and shifted differently solely due to the presence of padding in the batch.**

### 2.3 Upstream Hugging Face Compatibility Warning
Hugging Face's official `Wav2Vec2FeatureExtractor` documentation specifically warns:
> *"Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as `facebook/wav2vec2-base-960h`, have **not** been trained using `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask` should be passed."*

Crucially, in Hugging Face's `Wav2Vec2Model.forward()`:
```python
extract_features = self.feature_extractor(input_values)
```
The raw `attention_mask` is **never passed to `feature_extractor`**. It is only converted to downsampled feature masks and fed to the context Transformer encoder.

### 2.4 Empirical Proof of Contamination Under Naive Approaches
We measured the maximum absolute difference ($\max |y_{\text{alone}} - y_{\text{batched}}|$) on valid frames of a 1.0s speech sample ($16,000$ samples) padded to 2.0s with zeros:

| Approach | Conv Layer 0 Diff | Encoder Hidden State Diff | Output Logits Diff | Invariance Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Zero-Padding (No Mask)** | $3.1104$ | $1.6071$ | $10.5291$ | **FATAL CONTAMINATION** |
| **HF Attention Mask on Raw Waveform** | $3.1104$ | $0.7274$ | $6.7447$ | **FATAL CONTAMINATION** |

Blindly passing `attention_mask` does not fix feature extractor normalization, causing a logit shift of up to $6.74$ on identical speech.

---

## 3. Padding-Safe Architecture Implementation

To achieve 100% mathematical invariance without fine-tuning distortion, `LangfensWav2Vec2MDD.forward()` implements an unpadded-segment feature extraction policy:

```python
# 1. Feature extraction & contextual encoding via Wav2Vec2
# PADDING-SAFE ACOUSTIC BACKBONE EXECUTION:
if (audio_lengths == waveforms.shape[1]).all():
    # Fast vectorized path for uniform unpadded batches or batch_size == 1
    fe = self.wav2vec2.feature_extractor(waveforms).transpose(1, 2)
    frame_output_lengths = self._get_feat_extract_output_lengths(audio_lengths)
    T_max = fe.shape[1]
    padded_features = fe
else:
    # Per-sample unpadded feature extraction prevents GroupNorm contamination
    feats = [
        self.wav2vec2.feature_extractor(waveforms[i:i+1, :audio_lengths[i]]).transpose(1, 2)
        for i in range(batch_size)
    ]
    frame_output_lengths = torch.tensor([f.shape[1] for f in feats], device=device, dtype=torch.long)
    T_max = frame_output_lengths.max().item()
    padded_features = torch.zeros(batch_size, T_max, feats[0].shape[-1], device=device, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        padded_features[i, :f.shape[1]] = f[0]

# Feature attention mask (True for valid frames, False for padding)
frame_indices = torch.arange(T_max, device=device).unsqueeze(0)
feat_attention_mask = frame_indices < frame_output_lengths.unsqueeze(1)

# Feature projection with explicit zeroing of padding positions
hidden_states, _ = self.wav2vec2.feature_projection(padded_features)
hidden_states = hidden_states.masked_fill(~feat_attention_mask.unsqueeze(-1), 0.0)

# Context Transformer Encoder with boolean attention mask
encoder_outputs = self.wav2vec2.encoder(hidden_states, attention_mask=feat_attention_mask)
hidden_states = encoder_outputs.last_hidden_state
```

### Key Guarantees:
1. **Uncontaminated GroupNorm:** For each utterance $i$, `self.wav2vec2.feature_extractor` evaluates strictly over the active duration $[0, \text{audio\_lengths}[i]]$. The empirical mean and variance are identical to evaluating the sample in isolation.
2. **Latent Zero-Padding:** The extracted feature vectors $z_i \in \mathbb{R}^{T_i \times 512}$ are padded to $T_{\text{max}} = \max_i T_i$ with zero vectors.
3. **LayerNorm Bias Neutralization:** In `feature_projection`, `LayerNorm` converts zero padding into bias constants; `hidden_states.masked_fill(~feat_attention_mask.unsqueeze(-1), 0.0)` resets padding frames strictly to zero before the positional convolution layer.
4. **Boolean Encoder Masking:** The 12-layer context Transformer applies `feat_attention_mask` to prevent valid speech frames from attending to padding representations.

---

## 4. Cross-Attention Acoustic Frame Masking

In `CanonicalAcousticCrossAttention`:
```python
max_frames = hidden_states.shape[1]
frame_indices = torch.arange(max_frames, device=device).unsqueeze(0)
acoustic_padding_mask = frame_indices >= frame_output_lengths.unsqueeze(1)

aligned_acoustic, phone_embeds, alignment_weights = self.cross_attention_module(
    frame_features=hidden_states,
    canonical_phone_indices=canonical_phone_indices,
    acoustic_padding_mask=acoustic_padding_mask
)
```
`acoustic_padding_mask` is passed to `nn.MultiheadAttention(..., key_padding_mask=acoustic_padding_mask)`. Attention weights across invalid acoustic frames are set to $-\infty$ prior to softmax, ensuring canonical phone queries attend strictly to valid acoustic frames.

---

## 5. Verification Evidence: Numerical Invariance Benchmarks

We evaluated a 1.0s speech utterance ($16,000$ samples) with 4 canonical phonemes evaluated in isolation versus evaluated inside a padded batch alongside a 2.0s speech utterance ($32,000$ samples) with 6 canonical phonemes.

### 5.1 Custom Lightweight Model (Hidden Dim 256, 2 Layers)
| Representation / Head Output | Max Absolute Difference ($\max \|y_{\text{alone}} - y_{\text{batched}}\|$) | Single-Precision Limit ($10^{-5}$) | Verdict |
| :--- | :---: | :---: | :---: |
| **CTC Logits** | $9.54 \times 10^{-7}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Binary MDD Logits** | $5.96 \times 10^{-8}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Diagnosis Logits** | $3.73 \times 10^{-7}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Cross-Attention Weights** | $1.49 \times 10^{-8}$ | $< 1.00 \times 10^{-5}$ | **PASS** |

### 5.2 Full Pretrained 12-Layer Backbone (`facebook/wav2vec2-base-960h`)
| Representation / Head Output | Max Absolute Difference ($\max \|y_{\text{alone}} - y_{\text{batched}}\|$) | Single-Precision Limit ($10^{-5}$) | Verdict |
| :--- | :---: | :---: | :---: |
| **CTC Logits** | $4.58 \times 10^{-7}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Binary MDD Logits** | $2.38 \times 10^{-7}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Diagnosis Logits** | $5.74 \times 10^{-7}$ | $< 1.00 \times 10^{-5}$ | **PASS** |
| **Cross-Attention Weights** | $3.73 \times 10^{-9}$ | $< 1.00 \times 10^{-5}$ | **PASS** |

Both architectures exhibit differences on the order of $10^{-7}$, strictly conforming to IEEE 754 float32 floating-point precision bounds.

---

## 6. Canonical Padding Exclusion from Predictions and Losses

1. **Prediction Interface (`model.predict`):**  
   When `canonical_phone_lengths` is provided, `predict()` computes `canonical_mask = frame_indices < canonical_phone_lengths.unsqueeze(1)` and returns `"canonical_mask": canonical_mask`. Downstream evaluation metrics (PR-AUC, F1, Diagnosis Accuracy) filter strictly by `canonical_mask`.
2. **Loss Aggregation:**  
   Collation pads `target_binary` and `target_diagnosis` with `ignore_index = -100`. Both `MaskedBCEWithLogitsLoss` and `MaskedCrossEntropyLoss` aggregate losses strictly over positions where `target != -100`, excluding all canonical padding tokens.

---

## 7. Regression Test Inventory

| Test Function | Location | Tested Guarantee | Status |
| :--- | :--- | :--- | :---: |
| `test_acoustic_padding_alone_vs_batched_invariance` | `tests/test_reconciliation_regression.py` | Alone vs padded batch invariance on custom architecture ($\Delta < 10^{-5}$) | **PASS** |
| `test_pretrained_acoustic_padding_alone_vs_batched_invariance` | `tests/test_reconciliation_regression.py` | Alone vs padded batch invariance on full pretrained backbone ($\Delta < 10^{-5}$) | **PASS** |
| `test_canonical_padding_excluded_from_loss_and_predictions` | `tests/test_reconciliation_regression.py` | Canonical mask propagation in `predict()` | **PASS** |
| `test_mixed_duration_batch_via_padding_mask` | `tests/test_model.py` | Dual interface support (`audio_lengths` and `padding_mask`) | **PASS** |
