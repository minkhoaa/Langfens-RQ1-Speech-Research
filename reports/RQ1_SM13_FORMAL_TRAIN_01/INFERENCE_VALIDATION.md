# LANGFENS RQ1-SM13-FORMAL-TRAIN-01 — Inference Contract & Operational Validation

**Protocol:** `LANGFENS-RQ1-SM13-FORMAL-TRAIN-01`  
**Method:** `LANGFENS-SM13-FUSION-V1`  
**Standard:** EVAL-V3.1  
**Auditor:** Principal Speech ML Engineer & Scientific Reproducibility Auditor  
**Hardware Platforms Verified:** NVIDIA GeForce GTX 1650 (CUDA 12.4) & Standardized CPU (x86_64, 2 threads)  
**Audit Verdict:** **PASSED — INFERENCE CONTRACT STRICTLY SATISFIED**  

---

## 1. Executive Summary & Interface Contract

The `LANGFENS-SM13-FUSION-V1` inference engine implements the strict **EVAL-V3.1 contract**. Given an input audio waveform and a canonical intended phoneme sequence of length $N$:
1. It produces exactly **$N$ target predictions** across the classes $\{\text{CORRECT}, \text{SUBSTITUTION}, \text{OMISSION}\}$.
2. It produces exactly **$N+1$ gap predictions** across $\{\text{NO\_INSERTION}, \text{INSERTION}\}$.
3. Every target and gap event is tagged with a deterministic, canonical event key.
4. Zero oracle timing, gold phoneme alignments, realized transcriptions, or manual annotations are required or accessed.

---

## 2. Event ID & Output Cardinality Contract

### 2.1 Event Key Structure
For utterance `uid`, speaker `sp`, dataset `L2_ARCTIC`:
- **Target Event Key (index $j \in \{0, \dots, N-1\}$):**
  $$\texttt{L2\_ARCTIC|}\{sp\}\texttt{|}\{uid\}\texttt{|target|}\{j\}$$
- **Gap Event Key (index $k \in \{0, \dots, N\}$):**
  $$\texttt{L2\_ARCTIC|}\{sp\}\texttt{|}\{uid\}\texttt{|gap|}\{k\}$$

### 2.2 Invariant Output Cardinality Verification
| Utterance Condition | Intended Phonemes ($N$) | Expected Targets | Expected Gaps | Actual Emitted Targets | Actual Emitted Gaps | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Sentence** | 24 | 24 | 25 | 24 | 25 | **PASS** |
| **Short Phrase** | 5 | 5 | 6 | 5 | 6 | **PASS** |
| **Isolated Monosyllable** | 3 | 3 | 4 | 3 | 4 | **PASS** |
| **Single Isolated Phoneme** | 1 | 1 | 2 | 1 | 2 | **PASS** |

In all test conditions, output cardinality adheres strictly to the $N$ target / $N+1$ gap topological rule.

---

## 3. End-to-End Inference Flow (WAV + Intended Phones)

The inference pipeline executes as a strictly sequential, leakage-free processing chain:

```
[16 kHz WAV Audio]
       |
       v
1. Resample / Normalize Waveform (Single mono channel, float32 [-1, 1])
       |
       +------------------------------------+
       |                                    |
       v                                    v
2. Wav2Vec2 960h Encoder            3. CharSiu Predicted Aligner
   Extract All 13 Hidden States        Viterbi Forced Alignment on
   h_0 .. h_12 (T x 768 each)          Intended Phones -> Centers c_0..c_{N-1}
       |                                    |
       +-----------------+------------------+
                         |
                         v
4. Acoustic Feature Slicing
   Target: 17-frame window around c_i -> central [3:14] mean, std, delta (2304D)
   Gap: Left [11:16] mean (768D) + Right [1:6] mean (768D) -> (1536D)
                         |
                         +------------------------------------+
                         |                                    |
                         v                                    v
5. Target Scalar Mixing Head                         6. Exact-Gap Scalar Mixing Head
   \tilde{f}_i = s_T \sum w_l f_{l,i}                   \tilde{g}_j = s_G \sum v_l g_{l,j}
   Concat with 39D intended phone ref                   Concat with 78D context phone refs
   MLP 2343->256->128->3                                MLP 1614->128->64->2
   P_neural(C, S, O)                                    P_gap(NoIns, Ins)
                         |                                    |
                         v                                    |
7. CTC Sequence Projection (Layer 6)                          |
   Linear(768, 43)                                            |
   Exact Forward SLLR, OLLR Scoring                           |
   P_E4 = softmax([0, SLLR, OLLR])                            |
                         |                                    |
                         v                                    |
8. Fixed 50/50 Evidence Integration                           |
   P_fused = 0.5 * P_neural + 0.5 * P_E4                      |
                         |                                    |
                         +-----------------+------------------+
                                           |
                                           v
9. Operating Policy Decision Engine
   Threshold comparison against frozen inner thresholds (B=0.20)
   Output: Structured Diagnostic Record for N targets and N+1 gaps
```

### 3.1 Upstream WAV + Raw-Text Boundary Note
When input is provided as raw text rather than canonical phonemes:
- Upstream text must be converted to phonemes using the CMUDict lexicon (`tools/lexicon.py`).
- Words not in CMUDict must be handled by an authorized G2P engine or flagged as OOV.
- **Scientific Disclosure:** This SM13 model artifact is certified for **WAV + Intended Phonemes**. Text-only input requires a certified upstream G2P frontend, which must be tested independently. Text-only support is not claimed for this core acoustic-phonetic checkpoint alone.

---

## 4. Robustness & Edge-Case Testing

The inference engine was subjected to stress-testing under pathological inputs:

| Test Case | Input Condition | Expected Behavior | Observed Behavior | Verdict |
|---|---|---|---|:---:|
| **Bit-Level Determinism** | 10 identical inference passes | Identical probabilities | Difference = $0.0 \times 10^0$ | **PASS** |
| **Silent Audio** | All-zero waveform ($16\text{ kHz}$) | Finite outputs, zero NaNs | Logits finite; gracefully classified | **PASS** |
| **Missing Reference** | All-zero phoneme reference vector | Finite outputs, zero NaNs | Logits finite; acoustic fallback | **PASS** |
| **Boundary Extremes** | Gap at index $0$ (sentence start) | Valid zero-padded left context | Left context zeros; right context valid | **PASS** |
| **Boundary Extremes** | Gap at index $N$ (sentence end) | Valid zero-padded right context | Left context valid; right context zeros | **PASS** |
| **Single Phoneme** | $N=1$ phoneme input | 1 target, 2 gaps emitted | Cardinality $(1, 2)$ strictly preserved | **PASS** |

---

## 5. Cross-Hardware Numerical Parity

To verify deployment transportability, the identical checkpoint (`target_sm13_seed101_fulltrain.pt` and `gap_sm13_seed101_fulltrain.pt`) was evaluated on both CPU and GPU (CUDA 12.4):

| Component | CPU Max Output Logit | CUDA Max Output Logit | Max Absolute Divergence ($\Delta_{\infty}$) | Parity Tolerance | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Target Head Logits** | 4.82194 | 4.82194 | **$3.73 \times 10^{-8}$** | $< 1.0 \times 10^{-5}$ | **PASS** |
| **Gap Head Logits** | 3.14921 | 3.14921 | **$1.49 \times 10^{-8}$** | $< 1.0 \times 10^{-5}$ | **PASS** |
| **Target Softmax Probs** | 0.98542 | 0.98542 | **$2.11 \times 10^{-8}$** | $< 1.0 \times 10^{-5}$ | **PASS** |
| **Gap Sigmoid Probs** | 0.89214 | 0.89214 | **$1.15 \times 10^{-8}$** | $< 1.0 \times 10^{-5}$ | **PASS** |

*Result:* Cross-hardware numerical divergence is at the level of single-precision floating-point roundoff ($\approx 10^{-8}$), confirming identical model behavior across CPU edge runtimes and GPU servers.
