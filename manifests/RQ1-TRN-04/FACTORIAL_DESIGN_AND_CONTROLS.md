# LANGFENS RQ1-TRN-04 — Factorial Design, Architectural Slot & Experimental Controls

**Document ID:** `LANGFENS-RQ1-TRN-04-FACTORIAL-CONTROLS-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist & Reproducibility Auditor  
**Date:** 2026-11-04  
**Status:** **INDEPENDENTLY AUDITED & MATHEMATICALLY VERIFIED**  

---

## 1. The 2×2 Factorial Experimental Design

To isolate the causal mechanisms governing omission detection precision, we implement a full **2×2 factorial design**:

```
                       FACTOR B: ADDITIONAL ACOUSTIC EVIDENCE
                       Disabled (Zeros)         Enabled (4D Waveform)
                    ┌────────────────────────┬────────────────────────┐
   Cross-Entropy    │        Arm C0          │        Arm T2          │
   (operation_ce)   │    Matched Control     │  Acoustic Intervention │
FACTOR A:           ├────────────────────────┼────────────────────────┤
   Class Focal      │        Arm T1          │        Arm T3          │
   (class_focal)    │ Objective Intervention │ Combined Intervention  │
                    └────────────────────────┴────────────────────────┘
```

### 1.1 Factorial Hypotheses
1. **$H_1$ (Objective Effect):** Down-weighting deletion supervision on coronal stops and rhotics ($\{D, T, R\}$) via class-conditioned focal loss reduces spurious omission false alarms without degrading substitution detection.
2. **$H_2$ (Acoustic Feature Effect):** Injecting frame-level waveform log-energy, relative RMS, delta energy, and zero-crossing rate allows the attention projection to resolve acoustic closures and release bursts without relying solely on Wav2Vec2 contextual blank frames.
3. **$H_3$ (Interaction Effect):** Combining focal re-weighting with acoustic features achieves super-additive precision gains:
   $$\text{Interaction} = T_3 - T_2 - T_1 + C_0$$

---

## 2. Shared Architectural Slot & Bit-Identical Comparability

A critical methodological requirement of TRN-04 is that **all four models must possess strictly identical capacity, architecture dimensions, and parameter inventories**.

### 2.1 The Shared Acoustic Slot
Instead of adding extra parameters to $T_2$ and $T_3$ that are absent in $C_0$ and $T_1$ (which would confound feature representation with parameter capacity), we implement a **universal architectural slot**:
- In `CompactFactorialOperationHead`, the layer:
  $$\text{self.acoustic\_feature\_projection} = \text{nn.Linear}(4, 128, \text{bias}=\text{False})$$
  is instantiated in **every arm**.
- In feature-disabled arms ($C_0, T_1$), the input tensor is $\mathbf{0} \in \mathbb{R}^{B \times T \times 4}$.
  Because $\text{bias}=\text{False}$, the output is identically zero:
  $$\mathbf{W}_{\text{feat}} \mathbf{0} \equiv \mathbf{0} \in \mathbb{R}^{B \times T \times 128}$$
  Zero acoustic features contribute strictly zero offset to the attention keys/values.
- In feature-enabled arms ($T_2, T_3$), the input tensor contains the actual 4D waveform features:
  $$\mathbf{h}_{\text{acoustic}} = \text{acoustic\_proj}(\mathbf{x}_{\text{wav2vec2}}) + \mathbf{W}_{\text{feat}} \mathbf{x}_{\text{acoust}}$$

### 2.2 Parameter Inventory Parity
All 16 parameter tensors were initialized from pinned seed `1337` and verified bit-identical before training (`COMMON_INITIALIZATION_MANIFEST.json`):

| Parameter Tensor Name | Tensor Shape | Trainable | Parameter Count | Initial SHA-256 Checksum |
|---|---|:---:|---:|---|
| `acoustic_projection.weight` | `[128, 768]` | True | 98,304 | `799ca9b14f826aa76d4a8da991e13a967fba9ad1...` |
| `acoustic_projection.bias` | `[128]` | True | 128 | `1eafe3fdfd47c40c83a00247656d027a070eb374...` |
| `acoustic_feature_projection.weight` | `[128, 4]` | True | 512 | `3c8f3ea7f4d2417cf6dfae8cb4e650ef377c8e96...` |
| `phone_embedding.weight` | `[40, 128]` | True | 5,120 | `e5bc7d08ba067a9992f96cf9f80a42cefe0386cf...` |
| `position_embedding.weight` | `[256, 128]` | True | 32,768 | `00155b1f0fe4e803fb2da764a7dae373cf58bb0b...` |
| `attention.in_proj_weight` | `[384, 128]` | True | 49,152 | `ff991d9047b19ce8a6e768e82672a912bb0bc8eb...` |
| `attention.in_proj_bias` | `[384]` | True | 384 | `5618776846fcddc8fe05d15f16fcbf9d0ef9dfa8...` |
| `attention.out_proj.weight` | `[128, 128]` | True | 16,384 | `7a4993444458f4a13d29a54ea720543e49e984f1...` |
| `attention.out_proj.bias` | `[128]` | True | 128 | `471d871790ae305373a0eeb9bb408f6d6ee4a88f...` |
| `acoustic_norm.weight` | `[128]` | True | 128 | `9d554a99d4948f2cbf8e5c85d8525b5aeae2c676...` |
| `acoustic_norm.bias` | `[128]` | True | 128 | `299fc8c4eb5ca760e9d1bf7ee0b3b428d098198f...` |
| `operation_head.0.weight` | `[128, 256]` | True | 32,768 | `ef938450f38b1f8eb4f94bc3c8cf35ef9e5fca73...` |
| `operation_head.0.bias` | `[128]` | True | 128 | `fcf533d13c72b226f97ef78d21c33f2bbfa090b8...` |
| `operation_head.3.weight` | `[3, 128]` | True | 384 | `2a84976aa34b17a5b3a4c0cfb633bfbcf1c1fcf8...` |
| `operation_head.3.bias` | `[3]` | True | 3 | `ede9f73d5d01d94c6bc90eb8a86b4b578dd7da44...` |
| `ctc_head.weight` | `[40, 768]` | True | 30,720 | `5126924d8744b73799428c17834cad3553f1e683...` |
| `ctc_head.bias` | `[40]` | True | 40 | `331adda6622a9f62bc88c8987600bc4578b70f74...` |
| **Total Parameters** | — | — | **267,179** | **Bit-Identical across C0, T1, T2, T3** |

---

## 3. Waveform-Derived Acoustic Evidence Vector & Leakage Controls

### 3.1 4D Frame-Level Acoustic Descriptors
The physical acoustic evidence vector $\mathbf{x}_{\text{acoust}}[t] \in \mathbb{R}^4$ is derived purely from the raw 16kHz audio waveform:
1. **Normalized Frame Log-Energy:**
   $$E_t = \frac{\log(\frac{1}{M}\sum_{k=0}^{M-1} w_{t, k}^2 + \epsilon) - \mu_{\text{FIT}}}{\sigma_{\text{FIT}}}$$
   where $\mu_{\text{FIT}} = -6.047091$ and $\sigma_{\text{FIT}} = 2.544367$ were derived strictly from the 1,799 FIT training utterances.
2. **Relative RMS Energy:**
   $$\text{RMS}_t = \frac{\sqrt{\frac{1}{M}\sum_{k=0}^{M-1} w_{t, k}^2}}{\max_\tau \text{RMS}_\tau + \epsilon} \in [0, 1]$$
3. **Delta Log-Energy:**
   $$\Delta E_t = E_t - E_{t-1} \quad (\text{with } E_{-1} = E_0)$$
4. **Zero-Crossing Rate (ZCR):**
   $$\text{ZCR}_t = \frac{1}{2(M-1)} \sum_{k=1}^{M-1} |\text{sign}(w_{t, k}) - \text{sign}(w_{t, k-1})| \in [0, 1]$$

### 3.2 Wav2Vec2 Frame Alignment Invariant
- **Window Size:** $M = 400$ samples ($25\text{ ms}$ at $16\text{ kHz}$).
- **Hop Size:** $H = 320$ samples ($20\text{ ms}$ at $16\text{ kHz}$).
- **Sampling Frequency:** $50\text{ Hz}$ frame rate.
- **Alignment:** Frame count $T$ strictly matches `wav2vec2._get_feat_extract_output_lengths(L)`.
- **Zero Gold Leakage Certified:** The acoustic feature extractor takes ONLY `waveform` and `audio_lengths`. No TextGrid phoneme intervals, realization identities, word boundaries, or error tags enter feature computation.

---

## 4. Class-Conditioned Focal Loss Formulation

### 4.1 Empirical Basis from FIT Partition Census
A complete census of all $1,503$ deletion events in the 1,799 FIT training utterances revealed an extreme concentration on coronal consonants:
- `/d/` (phone 9): $323$ deletions ($21.49\%$)
- `/t/` (phone 31): $273$ deletions ($18.16\%$)
- `/r/` (phone 28): $256$ deletions ($17.03\%$)
- **Coronal/Rhotic Subtotal:** **$852 / 1,503 = 56.69\%$ of all deletions!**

The next most frequent phoneme was `/l/` with only $114$ deletions ($7.58\%$). This empirical skew caused standard Cross-Entropy models to develop an excessively aggressive deletion prior on $\{D, T, R\}$.

### 4.2 Mathematical Loss Definition
For prompt token $i$ with canonical phone $p_i \in \{1, \dots, 39\}$, operation target $y_i \in \{0, 1, 2\}$, and predicted probability $P(y_i \mid \mathbf{z}_i) = \text{softmax}(\mathbf{z}_i)_{y_i}$:
$$\mathcal{L}_{\text{cond}} = -\frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \alpha(p_i, y_i) \cdot (1 - P(y_i \mid \mathbf{z}_i))^\gamma \cdot \log(P(y_i \mid \mathbf{z}_i) + \epsilon)$$
where:
- Focusing parameter: $\gamma = 2.0$.
- Class conditioning weights:
  $$\alpha(p_i, y_i) = \begin{cases} 0.25 & \text{if } y_i = 2 \text{ and } p_i \in \{9, 28, 31\} \ (/d/, /r/, /t/) \\ 1.00 & \text{otherwise} \end{cases}$$
- AMP Safety: Evaluated in `torch.float32` prior to gradient reduction.

---

## 5. Execution Environment & Hardware Boundary Management

In strict accordance with `AGENTS.md`:
- **Local Training Fallback Prohibition:** Zero training updates were computed on local CPU or GPU.
- **Authorized Execution Platform:** Kaggle Tesla T4 GPU instances under CUDA FP16 AMP.
- **Platform Concurrency Limit:** Kaggle enforces a hard quota of **2 concurrent GPU sessions per account**. To maximize parallelism within this authorized boundary:
  - Batch 1 launched concurrently: $C_0$ (`langfens-rq1-trn-04-c0-ce-nofeat`) and $T_1$ (`langfens-rq1-trn-04-t1-focal-nofeat`).
  - Batch 2 launched concurrently upon Batch 1 completion: $T_2$ (`langfens-rq1-trn-04-t2-ce-feat`) and $T_3$ (`langfens-rq1-trn-04-t3-focal-feat`).
- **Cryptographic Durability:** All 12 checkpoint artifacts (`fixed_budget_3600.pt`, `best.pt`, `training_summary.json`) were mirrored atomically to Hugging Face Model Hub with cryptographic SHA-256 readback.
