# LANGFENS — RQ1-TRN-01A-R: Alignment & Inference Validity Report

**Execution Gate:** RQ1-TRN-01A-R — Pretrained Backbone & Inference Validity Gate  
**Deliverable Associated Path:** `manifests/RQ1-TRN-01A/reconciliation/ALIGNMENT_AND_INFERENCE_VALIDITY.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/`  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Speech Recognition Architect  
**Gate Evaluation Status:** `ALIGNMENT_AND_INFERENCE_VALIDITY: PASS`  

---

## 1. Executive Summary & Problem Formulation

In a real Computer-Assisted Pronunciation Training (CAPT) deployment, the system receives only:
1. User speech waveform: $x \in \mathbb{R}^{T_{\text{audio}}}$
2. Canonical prompt text or phoneme sequence: $p = (p_1, p_2, \dots, p_N)$

It does **not** receive manual Praat TextGrid interval boundaries $[t_{\text{start}}, t_{\text{end}}]$.

The earlier prototype in `RQ1-TRN-01A` required external `phone_intervals` to perform frame pooling (`pool_frames_to_canonical_phones`), conflating oracle-aligned supervision with deployable inference. This violated CAPT deployment validity.

---

## 2. Evaluation of Candidate Alignment Mechanisms

We conducted a comparative architectural analysis of three candidate mechanisms:

| Approach | Alignment Mechanism | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Learned Monotonic Attention** | Constrained step-wise left-to-right attention | Enforces strictly monotonic progression | Hard step functions are non-differentiable; fails when complete phonetic deletions occur ($D=3,420$). | **Rejected** |
| **2. CTC-Based Forced Alignment** | Viterbi trellis dynamic programming over CTC logits | Uses existing CTC head emissions | Non-differentiable backward pass; GPU-to-CPU synchronization bottleneck; misaligns heavily distorted non-native speech. | **Rejected for Inference Head** |
| **3. Differentiable Multihead Cross-Attention** | Canonical phone tokens act as queries attending to continuous acoustic frames | **100% Differentiable**; zero test-time alignment tool dependency; naturally models substitutions and deletions; GPU batched. | Requires learned sequence positional embeddings (implemented via `CanonicalAcousticCrossAttention`). | **SELECTED FOR DEPLOYMENT** |

---

## 3. Selected Architecture: Differentiable Cross-Attention

The deployable alignment module `CanonicalAcousticCrossAttention` operates as follows:

1. **Canonical Phone Embedding:**
   $$\mathbf{E}_{\text{phone}} = \text{Embedding}(p_i) \in \mathbb{R}^{B \times N \times D_{\text{embed}}}$$
2. **Positional Sequence Encoding:**
   $$\mathbf{Q} = \mathbf{W}_{\text{proj}} \mathbf{E}_{\text{phone}} + \mathbf{E}_{\text{pos}}(1..N) \in \mathbb{R}^{B \times N \times D_{\text{hidden}}}$$
3. **Acoustic Frame Representation:**
   $$\mathbf{K} = \mathbf{V} = \mathbf{H}_{\text{acoustic}} \in \mathbb{R}^{B \times T_{\text{frames}} \times D_{\text{hidden}}}$$
4. **Multihead Cross-Attention with Acoustic Padding Mask:**
   $$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{acoustic}} \right) \in \mathbb{R}^{B \times N \times T_{\text{frames}}}$$
   $$\mathbf{H}_{\text{canonical}} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{B \times N \times D_{\text{hidden}}}$$

### Handling Phonetic Error Categories:
- **Substitutions ($S$):** The canonical phone query $p_i$ attends to the acoustic time segment where the substitute was spoken. The acoustic-canonical embedding discrepancy informs the binary error head, while $\mathbf{H}_{\text{canonical}}$ directly predicts the substitute class in the diagnosis head.
- **Deletions ($D$):** When a sound is omitted, the cross-attention module aggregates diffuse background/surrounding context, producing an anomalous representation that triggers the binary error head ($b_i = 1$) while the diagnosis head target is masked ($-100$).
- **Insertions ($I$):** Unprompted epenthetic additions are captured chronologically by the independent multi-task CTC head without corrupting the canonical sequence order or token count $N$.

---

## 4. Separation of Oracle Baseline and Deployable Inference

The model cleanly distinguishes two operating modes:

1. **Deployable Mode (`alignment_mode="cross_attention"` — DEFAULT):**
   - Inputs: `waveforms` and `canonical_phone_indices`.
   - Any passed `phone_intervals` are strictly ignored.
   - Predictions are 100% alignment-free.
2. **Oracle Baseline (`alignment_mode="oracle"`):**
   - Requires explicit `phone_intervals` from gold TextGrids.
   - Retained solely as a scientific reference ablation labeled **ORACLE-ALIGNED BASELINE**.

---

## 5. Mandatory Negative Invariance Test

To guarantee that deployment predictions cannot secretly leak or depend on gold interval boundaries, we implemented the mandatory negative test in `tests/test_reconciliation_regression.py::test_deployment_mode_invariance_to_gold_intervals`:

1. **Run 1:** Model evaluated with raw audio and canonical tokens (no intervals).
2. **Run 2:** Model evaluated with identical audio, identical tokens, plus true gold TextGrid boundaries.
3. **Run 3:** Model evaluated with identical audio, identical tokens, plus completely scrambled / erroneous boundaries.

### Empirical Test Result:
$$\text{Binary Logits: } \mathbf{z}_{\text{clean}} \equiv \mathbf{z}_{\text{gold}} \equiv \mathbf{z}_{\text{scrambled}} \quad (\text{Exact bit-level equality})$$
$$\text{Diagnosis Logits: } \mathbf{y}_{\text{clean}} \equiv \mathbf{y}_{\text{gold}} \equiv \mathbf{y}_{\text{scrambled}} \quad (\text{Exact bit-level equality})$$

**Verdict:** Test-time ground-truth boundaries cannot influence deployment predictions. Zero leakage confirmed.
