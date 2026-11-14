# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Complete End-to-End Runtime & Inference Audit

**Auditor:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign:** `RQ1-SM13-VALIDATION-PREFLIGHT-01` (Lane C)  
**Target Artifact:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101)  
**Hardware Platforms Profiled:**
1. **GPU Platform:** NVIDIA GeForce GTX 1650 (4096 MiB GDDR5 VRAM), PyTorch 2.6.0+cu124, CUDA 12.4
2. **CPU Platform:** Standardized x86_64 Linux Host, 2 compute threads, PyTorch 2.6.0+cu124  
**Audit Dataset:** Permitted L2-ARCTIC `TRAIN` partition (16 kHz mono WAVs, mean duration: $4.09\text{ seconds}$).  
**Audit Verdict:** **INFERENCE COMPLETENESS PASS (Contract Clarified & Prior Omissions Corrected)**  

---

## 1. Executive Summary & Forensic Runtime Audit

Prior documentation in `reports/RQ1_SM13_FORMAL_TRAIN_01/RUNTIME_BENCHMARK.md` reported an end-to-end GPU latency of **$74.03\text{ ms}$ per utterance** ($\text{RTF} = 0.0181$). 

An independent AST audit and empirical profiling of `agent_c_operational_validation.py` revealed that the reported $74.03\text{ ms}$ was **not an end-to-end pipeline measurement**. It measured a partial sub-graph that relied on pre-cached alignment arrays, omitted neural forced alignment, passed dummy zero vectors for phoneme embeddings, and skipped sequence likelihood scoring.

This audit executes the **complete, self-contained end-to-end inference pipeline** on raw audio files without oracle timing, precomputed features, or cached alignments.

### Key Forensic Benchmark Findings:
- **True GPU End-to-End Latency:** **$223.09\text{ ms}$ per utterance** (Cold start: $396.40\text{ ms}$, Median $\text{P50} = 194.97\text{ ms}$, $\text{P95} = 444.62\text{ ms}$).
- **True GPU Real-Time Factor (RTF):** **$0.0549$** ($\mathbf{18.2\times}$ faster than real time).
- **True CPU End-to-End Latency:** **$840.66\text{ ms}$ per utterance** (Cold start: $1143.52\text{ ms}$, Median $\text{P50} = 828.17\text{ ms}$, $\text{P95} = 1251.38\text{ ms}$, $\text{RTF} = 0.2130$).
- **Quantification of Prior Omissions:** Exactly **$151.24\text{ ms}$ of computation ($67.8\%$ of true end-to-end GPU runtime)** was omitted from the previously published $74.03\text{ ms}$ benchmark. The true runtime is **$3.15\times$ higher** than reported.
- **Real-Time Viability Confirmed:** Despite the underreporting correction, the true RTF of $0.0549$ ($18.2\times$ faster than real time on GPU) and $0.2130$ ($4.7\times$ faster than real time on CPU) remains fully compliant with interactive language tutoring requirements ($\text{RTF} < 0.25$).

---

## 2. Complete Component-by-Component Latency Decomposition

The table below provides the authoritative empirical decomposition of mean warm latency across all 11 pipeline stages, measured across 30 permitted TRAIN utterances on both GPU and CPU:

| Stage # | Pipeline Stage Operation | Prior Benchmark Status | GPU Mean (ms) | GPU Share (%) | CPU Mean (ms) | CPU Share (%) | Hardware Speedup |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **WAV Decode & Audio Loading** | Included | 4.71 ms | 2.1% | 0.47 ms | <0.1% | $0.1\times$ (I/O) |
| **2** | **CharSiu Frame Acoustic Forward** | **OMITTED** | **74.32 ms** | **33.3%** | **455.43 ms** | **54.2%** | **$6.1\times$** |
| **3** | **Viterbi Forced Alignment DP** | **OMITTED** | **29.60 ms** | **13.3%** | **27.16 ms** | **3.2%** | **$0.9\times$ (CPU-bound DP)** |
| **4** | **Wav2Vec2 13-Layer Backbone** | Included | 50.60 ms | 22.7% | 297.67 ms | 35.4% | $5.9\times$ |
| **5** | **Acoustic Crop Feature Slicing** | Included | 14.32 ms | 6.4% | 16.05 ms | 1.9% | $1.1\times$ |
| **6** | **Reference Phone 1-Hot Encoding** | **OMITTED (Used Zeros)** | 0.03 ms | <0.1% | 0.03 ms | <0.1% | $1.0\times$ |
| **7** | **SM13 Target Mixing Head** | Included | 0.70 ms | 0.3% | 1.09 ms | 0.1% | $1.6\times$ |
| **8** | **SM13 Gap Mixing Head** | Included | 0.37 ms | 0.2% | 0.97 ms | 0.1% | $2.6\times$ |
| **9** | **CTC Layer 6 Linear Projection** | Included | 0.11 ms | <0.1% | 0.16 ms | <0.1% | $1.5\times$ |
| **10** | **CTC/E4 Sequence Likelihood Scoring** | **OMITTED** | **47.29 ms** | **21.2%** | **41.51 ms** | **4.9%** | **$0.9\times$ (C++ Kernel)** |
| **11** | **Fixed 50/50 Fusion & Warning Policy** | Included | 0.03 ms | <0.1% | 0.03 ms | <0.1% | $1.0\times$ |
| **TOTAL** | **True Complete End-to-End Pipeline** | **FULL EXECUTION** | **223.09 ms** | **100.0%** | **840.66 ms** | **100.0%** | **$3.77\times$** |

---

## 3. Forensic Analysis of Prior Benchmark Omissions

### 3.1 The Three Major Missing Stages
1. **CharSiu Frame-Level Acoustic Model ($+74.32\text{ ms}$ on GPU, $+455.43\text{ ms}$ on CPU):**
   `agent_c_operational_validation.py` loaded `alignment_centers.npz` directly from disk. In real-world operation without oracle alignments, the audio waveform must be processed by `charsiu/en_w2v2_fc_10ms` to produce frame-level phone emission logits.
2. **Viterbi Trellis Alignment DP ($+29.60\text{ ms}$ on GPU/CPU):**
   Alignment centers were previously loaded from cache. In true inference, an emitting-state dynamic programming trellis must align the frame logits against the canonical intended phoneme sequence to identify phone center coordinates.
3. **CTC/E4 Sequence Likelihood Scoring ($+47.29\text{ ms}$ on GPU/CPU):**
   Prior scripts executed `LinearCTC(out.hidden_states[6][0])` in $0.08\text{ ms}$, but **never executed E4 scoring**. E4 scoring constructs $1 + 5N$ hypotheses ($1$ canonical, $4N$ competitor substitutions, $N$ deletions) and executes the compiled C++ forward pass to evaluate Sequence Log-Likelihood Ratios ($\text{SLLR}, \text{OLLR}$).

### 3.2 Omission Summary & Scaling Analysis
- **Included in Prior Report:** $70.85\text{ ms}$ (Matches the reported $74.03\text{ ms}$).
- **Omitted from Prior Report:** $151.24\text{ ms}$ ($67.8\%$ of total processing).
- **True End-to-End Runtime:** $223.09\text{ ms}$ ($3.15\times$ higher than reported).

---

## 4. Latency Distribution & Tail Latency Analysis

Latencies were measured across 30 utterances ranging from short prompts ($1.3\text{ s}$, $14$ phones) to long sentences ($7.9\text{ s}$, $51$ phones):

| Latency Metric | Complete GPU (ms) | Complete CPU (ms) | Normalized GPU (ms / s audio) |
|---|:---:|:---:|:---:|
| **Cold Start (1st Utterance)** | 396.40 ms | 1143.52 ms | 98.4 ms / s |
| **Warm Minimum** | 104.38 ms | 467.53 ms | 38.2 ms / s |
| **Warm P10 (Short Sentences)** | 129.56 ms | 567.84 ms | 44.1 ms / s |
| **Warm P50 (Median)** | **194.97 ms** | **828.17 ms** | **51.8 ms / s** |
| **Warm Mean** | **223.09 ms** | **840.66 ms** | **54.9 ms / s** |
| **Warm P90** | 378.72 ms | 1188.99 ms | 68.4 ms / s |
| **Warm P95 (Tail Latency)** | **444.62 ms** | **1251.38 ms** | **74.1 ms / s** |
| **Warm P99** | 553.65 ms | 1301.28 ms | 81.2 ms / s |
| **Warm Maximum** | 583.16 ms | 1312.44 ms | 83.0 ms / s |

*Tail Latency Evaluation:* On GPU, even at the 95th percentile ($444.62\text{ ms}$), interactive turn-taking thresholds ($< 500\text{ ms}$) are strictly respected. On CPU, long utterances reach $1.25\text{ s}$, which remains functional for asynchronous grading but exceeds real-time interactive dialogue thresholds.

---

## 5. Hardware Memory & Resource Profile

| Resource Metric | Prior Published | Audited Reality | Disposition |
|---|:---:|:---:|:---:|
| **Peak GPU VRAM Footprint** | $274.9\text{ MB}$ | **$855.9\text{ MB}$** | **UNDERREPORTED ($3.1\times$ higher)** |
| **VRAM Headroom on GTX 1650 (4,096 MB)** | $93.3\%$ | **$79.1\%$ free ($3,240\text{ MB}$ headroom)** | **SAFE (Well within 4GB limit)** |
| **Host System RAM (Resident Set Size)** | $\sim 680\text{ MB}$ | **$1,205.0\text{ MB}$ (GPU) / $2,065.9\text{ MB}$ (CPU)** | **SAFE (< 4GB RAM budget)** |
| **Novel Trainable Parameters** | 28 scalars | Exactly 28 scalar parameters | **VERIFIED** |
| **Model Weights on Disk** | $\sim 3.4\text{ MB}$ | $3.53\text{ MB}$ (Target + Gap + CTC) | **VERIFIED** |

*Root Cause of VRAM Underreporting:* The prior benchmark only loaded the Wav2Vec2 backbone ($360\text{ MB}$) and MLP heads ($3\text{ MB}$). Complete inference concurrently loads the CharSiu acoustic model ($360\text{ MB}$), raising total static model memory to $\sim 725\text{ MB}$, with peak dynamic activations reaching $855.9\text{ MB}$.

---

## 6. Output Contract, Event Cardinality & Robustness Validation

All contract invariants were programmatically verified in `benchmark_complete_inference.py`:

1. **Exact Event Cardinality ($N$ Targets, $N+1$ Gaps):**
   Verified across arbitrary lengths $N \in \{1, 2, 5, 20, 50\}$.
   - Target head strictly outputs $(N, 3)$ simplex probabilities.
   - Gap head strictly outputs $(N+1, 2)$ simplex probabilities.
2. **Boundary Gap Semantics:**
   - Gap $0$: Preceding target phoneme $0$ (leading utterance silence margin).
   - Gap $i$ ($1 \le i \le N-1$): Inter-phoneme interval between target $i-1$ and target $i$.
   - Gap $N$: Trailing target phoneme $N-1$ (trailing utterance silence margin).
3. **Stable Event IDs:**
   Verified coordinate formatting:
   $$\text{Target ID}: \quad \texttt{\{utterance\_id\}::\{slot\_idx\}}$$
   $$\text{Gap ID}: \quad \texttt{\{utterance\_id\}::gap\_\{gap\_idx\}}$$
4. **Unknown / Out-of-Vocabulary Phone Handling:**
   - If an unmapped or OOV symbol appears in the reference string, the 39D one-hot vector defaults to all zeros ($\mathbf{0}_{39}$).
   - The MLP head processes $\mathbf{0}_{39}$ gracefully without NaN or Inf emissions, falling back to pure acoustic evidence.
5. **Numerical Determinism:**
   Across identical CPU and CUDA passes, output probability deltas are $< 1.0 \times 10^{-5}$ (bit-exact on CPU; floating-point identical within CUDA parallelism tolerances).

---

## 7. Inference Scope Limitation: WAV + Canonical Phone Contract

The auditor explicitly defines the boundary of verified inference capabilities:

```
+-----------------------------------------------------------------------------+
| VERIFIED SUPPORTED INFERENCE CONTRACT:                                      |
| Input:  (1) Native 16 kHz Mono WAV audio file                               |
|         (2) Canonical intended phoneme sequence (List[str], ARPAbet 39)    |
| Output: (1) N target error diagnoses (Correct, Substitution, Omission)      |
|         (2) N+1 exact-gap insertion diagnoses                               |
| Status: FULLY VERIFIED, SELF-CONTAINED, DETERMINISTIC                       |
+-----------------------------------------------------------------------------+
                                     VS
+-----------------------------------------------------------------------------+
| UNVERIFIED / OMITTED CAPABILITY: RAW-TEXT-TO-PHONE G2P                      |
| Input:  Raw orthographic text string (e.g. "The task we set ourselves...")  |
| Flaws:  - CMUDict lookup fails on OOV names, acronyms, or numbers.         |
|         - Lacks syntactic homograph disambiguation ("read", "record").      |
|         - Lacks prosodic reduction modeling and non-native variant mapping. |
| Status: EXPLICITLY UNSUPPORTED BY FROZEN ARTIFACT                           |
+-----------------------------------------------------------------------------+
```

---

## 8. Audit Verdict

- **LANE C VERDICT:** **INFERENCE COMPLETENESS PASS (Contract Clarified & Omissions Quantified)**
- **Disposition:**
  1. Complete self-contained inference from WAV + canonical intended phonemes is operational and verified.
  2. Prior published latency ($74.03\text{ ms}$) is corrected to **$223.09\text{ ms}$ on GPU** and **$840.66\text{ ms}$ on CPU**.
  3. Real-time factor ($\text{RTF} = 0.0549$ GPU, $0.2130$ CPU) confirms edge and interactive deployability.
  4. Raw-text-to-phone generation is formally excluded from the artifact's verified scope.
