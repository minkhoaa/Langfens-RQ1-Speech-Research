# LANGFENS RQ1-SM13-FORMAL-TRAIN-01 — Master Formal Training & Model Readiness Report

**Role:** Principal Speech ML Engineer, Training Lead & Scientific Reproducibility Auditor  
**Repository:** `/home/khoa/Projects/New-Method-Research`  
**Execution Standard:** EVAL-V3.1 (Exact $N$ targets, $N+1$ gaps)  
**Selected Method:** `LANGFENS-SM13-FUSION-V1`  
**Hardware Lease Profile:** Exclusive NVIDIA GeForce GTX 1650 (4096 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4  
**Git Parent Revision:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Final Scientific Verdict:** **`AUTHORIZED_FOR_INDEPENDENT_VALIDATION_HANDOFF`**  
**Acceptance Gates:**
- `TRAINING_COMPLETE`: **PASSED**
- `ARTIFACT_READY`: **PASSED**
- `POLICY_READY`: **PASSED (Operational FWR Control Unconfirmed Pending Validation)**
- `BLOCKED`: **FALSE**

---

## 1. Executive Summary & Training Certification

The formal training campaign for **`LANGFENS-SM13-FUSION-V1`** has been executed to completion under strict scientific governance. All exploration, backbone benchmarking, and architecture sweeps were closed prior to launch. Full-TRAIN model artifacts have been reproducibly generated across all three registered seeds (`101`, `202`, `303`) on all 1,800 utterances (12 speakers) of the L2-ARCTIC TRAIN partition.

### Key Campaign Milestones:
1. **Preflight Authorization Gate Passed:** All 12 scientific and engineering prerequisites in Phase 0 passed without failure. Checkpoint integrity, dataset population, dependency versions, baseline parity, and exclusive GPU lease acquisition were independently certified.
2. **Deterministic Full-TRAIN Fitting Complete:** Models were trained for all 3 seeds under locked recipes. Seed 101 was designated *ex ante* as the predetermined primary production artifact. Zero in-sample AP cherry-picking was performed.
3. **Scientific Disclosures & Generalization Boundaries:** In-sample diagnostic metrics (Macro-3 AP $0.5806$ for Seed 101) are strictly demarcated from held-out development generalization ($0.327300 \pm 0.0083$). Operational FWR control is explicitly marked **UNCONFIRMED** on external test data due to the development FWR overshoot ($0.235370$ vs nominal $0.200$).
4. **Ultra-Low Latency Certified:** End-to-end inference completes in **$74.03\text{ ms}$** per utterance on GPU ($\text{RTF} = 0.0181$, $55\times$ real-time) and **$388.22\text{ ms}$** on CPU ($\text{RTF} = 0.0960$, $10.4\times$ real-time).
5. **Protected Partitions Sealed:** `DEV_SEARCH`, `DEV_CAL`, and `TEST_SEALED` remain 100% untouched and sealed. Training strictly halted at artifact readiness.

---

## 2. Phase 0 — Source Freeze & Authorization Gate Audit

Prior to formal training, Phase 0 executed a 12-point automated verification gate (`preflight_gate.py`):

| # | Check Description | Requirement / Specification | Observed Result | Status |
|---|---|---|---|:---:|
| 1 | **Git HEAD & Dependencies** | Clean revision, PyTorch $\ge 2.6$, CUDA 12.4 | Commit `aa4e6ea2...`, PyTorch 2.6.0+cu124, GTX 1650 | **PASS** |
| 2 | **Dataset Manifest Integrity** | 1,800 utts, 12 spks, 59,562 targets, 61,362 gaps | Exact count match; protected splits isolated | **PASS** |
| 3 | **Historical Lineage Recovery** | 24 checkpoints from METHOD-SELECT-02R verified | All 24 checkpoints loadable with exact key matches | **PASS** |
| 4 | **Baseline & SM13 Dev Parity** | Baseline AP 0.304154, SM13 Dev AP 0.327300 | Replicated to $< 10^{-4}$ tolerance ($+7.61\%$ gain) | **PASS** |
| 5 | **Matched-FWR Audit** | FWR overshoot documented, matched gain $+9.22\%$ | Dev FWR 0.235370, Matched F1 0.173022 at FWR 0.2044 | **PASS** |
| 6 | **Feature Invariants Audit** | 13 layers of Target (2304D) & Gap (1536D) | All 26 `.npy` memmaps valid, shape verified | **PASS** |
| 7 | **Model Architecture & Autograd** | Gradient flow on $\boldsymbol{\alpha}, \boldsymbol{\beta}, s_T, s_G$; eval() determinism | Gradients non-zero, eval() disables dropout | **PASS** |
| 8 | **Checkpoint Resume Safety** | Atomic write, bit-exact parameter reload | Simulated crash recovered parameter-identical state | **PASS** |
| 9 | **Audited CTC Kernel** | Shared object `ctc_batch.so` loadable | Symbol `score_ctc_batch` verified | **PASS** |
| 10 | **Exclusive GPU Lease** | Lockfile `locks/gpu.lock` acquire & release | Acquired exclusively without contention | **PASS** |
| 11 | **Compute Profiling** | Peak VRAM $< 4096\text{ MB}$, runtime $< 300\text{ s}$ | Peak VRAM 274.9 MB, runtime $\sim 140\text{ s}$ | **PASS** |
| 12 | **Preflight Gate Verdict** | All checks pass without blockers | **AUTHORIZED_FOR_FORMAL_TRAINING** | **PASS** |

---

## 3. Agent A — Final SM13 Implementation & Inference Contract

The adopted architecture, **`LANGFENS-SM13-FUSION-V1`**, combines:
1. **Differentiable 13-Layer Multi-Layer Neural Trunk:**
   - Differentiable scalar mixing across all 13 hidden layers ($0 \dots 12$) of `facebook/wav2vec2-base-960h`.
   - Target head input: $2,304\text{D}$ acoustic crop ($[-8, +8]$ frames, central $[3:14]$ mean, std, boundary delta) $+ 39\text{D}$ intended phone one-hot $= 2,343\text{D}$.
   - Gap head input: $1,536\text{D}$ acoustic context (left $[11:16]$ mean $+$ right $[1:6]$ mean) $+ 78\text{D}$ context refs $= 1,614\text{D}$.
   - Learnable layer attention weights ($\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^{13}$) and scalar scales ($s_T, s_G \in \mathbb{R}$).
2. **Frozen CTC Sequence Branch & Fixed 50/50 Evidence Fusion:**
   - `Linear(768, 43)` projection on layer 6 sequence representations.
   - Exact hypothesis scoring: $\text{SLLR} = \log P(\text{candidate substitution}) - \log P(\text{canonical})$, $\text{OLLR} = \log P(\text{canonical deletion}) - \log P(\text{canonical})$.
   - Sequence simplex: $P_{\text{E4}} = \text{softmax}([0, \text{SLLR}, \text{OLLR}])$.
   - Fused Target Probability: $P_{\text{fused}} = 0.5 \cdot P_{\text{neural}} + 0.5 \cdot P_{\text{E4}}$.
3. **Exact Cardinality Contract (EVAL-V3.1):**
   - Emits exactly $N$ target predictions and $N+1$ gap predictions per utterance.
   - Stable canonical event IDs: `L2_ARCTIC|{sp}|{uid}|target|{j}` and `L2_ARCTIC|{sp}|{uid}|gap|{k}`.
4. **WAV + Intended Phone Inference:**
   - Evaluated without gold timing, realized phone inputs, or manual alignments.
   - Text-only input requires separate certification of an upstream G2P frontend.

---

## 4. Agent B — Formal Full-TRAIN Fitting Results

Training was executed across all 12 eligible TRAIN speakers for seeds 101, 202, 303.

### 4.1 Summary Multi-Seed Comparison Table

| Attribute / Metric | Seed 101 (Primary) | Seed 202 (Replication) | Seed 303 (Sensitivity) | Multi-Seed Summary |
|---|:---:|:---:|:---:|:---:|
| **Designation** | **PRIMARY_PREDETERMINED** | **CONFIRMATORY** | **CONFIRMATORY** | **All 3 Seeds Completed** |
| **Target Epochs** | 12 | 12 | 12 | 12 (Fixed Terminal) |
| **Gap Epochs** | 5 | 5 | 5 | 5 (Fixed Terminal) |
| **Final Target Loss** | 0.3833 | 0.3782 | 0.3853 | $0.3823 \pm 0.0037$ |
| **Final Gap Loss** | 0.1620 | 0.1597 | 0.1630 | $0.1616 \pm 0.0017$ |
| **Learned Target Scale ($s_T$)** | 0.8813 | 0.8667 | 0.8911 | $0.8797 \pm 0.0123$ |
| **Learned Gap Scale ($s_G$)** | 1.0315 | 1.0291 | 1.0278 | $1.0295 \pm 0.0019$ |
| **Peak Target Attention Layer** | **Layer 6 (0.1105)** | **Layer 6 (0.1113)** | **Layer 6 (0.1116)** | **Layer 6 Invariant** |
| **Peak Gap Attention Layer** | Layer 7 (0.0790) | Layer 9 (0.0790) | Layer 7 (0.0789) | Layers 7–9 Invariant |
| **In-Sample Macro-3 AP** *(TRAIN)* | 0.580630 | 0.593743 | 0.589308 | $0.587894 \pm 0.0067$ |
| **In-Sample AP-Substitution** | 0.605548 | 0.604322 | 0.599183 | $0.603018 \pm 0.0034$ |
| **In-Sample AP-Omission** | 0.741877 | 0.760571 | 0.768703 | $0.757050 \pm 0.0138$ |
| **In-Sample AP-Insertion** | 0.394465 | 0.416336 | 0.400038 | $0.403613 \pm 0.0114$ |
| **In-Sample Macro-3 F1 ($B=0.2$)** | 0.417067 | 0.459214 | 0.449221 | $0.441834 \pm 0.0220$ |
| **In-Sample Realized FWR** | 0.776111 | 0.496667 | 0.550000 | $0.607593 \pm 0.1484$ |
| **In-Sample Insertion P@560** | 0.444643 | 0.442857 | 0.430357 | $0.439286 \pm 0.0078$ |
| **Historical Dev Macro-3 AP** *(Held-Out)* | **0.324357** | **0.320912** | **0.336630** | **$0.327300 \pm 0.0083$** |

### 4.2 Scientific Integrity Note: In-Sample vs Generalization Demarcation
- **In-sample metrics on TRAIN are NOT generalization results.** Because the model was fit on these 1,800 utterances, in-sample Macro-3 AP ($\sim 0.588$) reflects training fit and model capacity.
- **The authoritative estimate of held-out performance remains the multi-seed cross-validation figure:**
  $$\textbf{Macro-3 Error AP: } \mathbf{0.327300 \pm 0.0083} \quad (+7.61\% \text{ relative over baseline } 0.304154)$$
- True generalization to new speakers and audio distributions will be established exclusively by the independent validation gate on held-out evaluation partitions.

---

## 5. Agent C — Policy, Robustness & Latency Validation

1. **Warning Policy Adherence:**
   - Reference policy thresholds derived strictly on inner OOF splits of TRAIN.
   - Nominal $B=0.20$ false alarms/utterance ($80\%$ target, $20\%$ gap).
   - On development cross-validation, achieved FWR was **$0.235370$** ($+17.7\%$ overshoot).
   - At strictly matched realized FWR ($0.2044$), SM13 achieves Macro-3 F1 of **$0.173022$** vs baseline $0.158420$ ($+9.22\%$ gain).
   - Operational FWR control is explicitly marked **UNCONFIRMED** until independent evaluation.
2. **Determinism & Parity:**
   - 10 consecutive inference passes yielded identical logits ($\Delta = 0.0 \times 10^0$).
   - Cross-hardware numerical divergence between CPU and CUDA is $< 3.8 \times 10^{-8}$ (pure floating-point precision limit).
3. **Cardinality & Edge Cases:**
   - Correct handling of single-phoneme utterances, silent audio, and missing phoneme references verified.
4. **Latency Benchmark:**
   - GPU mean latency: **$74.03\text{ ms}$** per utterance ($\text{RTF} = 0.0181$).
   - CPU mean latency: **$388.22\text{ ms}$** per utterance ($\text{RTF} = 0.0960$).
   - Peak VRAM allocated: **$274.9\text{ MB}$** on GTX 1650.

---

## 6. Supervisor Acceptance Gate Determinations

| Gate | Disposition | Scientific Verification Evidence |
|---|:---:|---|
| **`TRAINING_COMPLETE`** | **PASSED** | Formal full-TRAIN fitting completed for all 3 registered seeds (101, 202, 303) across all 12 eligible speakers. Seed 101 designated as primary *ex ante*. All 28 novel scalar parameters verified. |
| **`ARTIFACT_READY`** | **PASSED** | Checkpoints load cleanly with zero missing/unexpected keys. Keyed predictions, intermediate epoch archives, and training receipts cryptographically sealed under SHA256 manifest. Determinism verified. |
| **`POLICY_READY`** | **PASSED (UNCONFIRMED)** | Eligible inner-derived warning policy loaded successfully. External operational FWR control is marked **UNCONFIRMED** pending independent validation gate due to development overshoot ($0.2354$ vs $0.200$). |
| **`BLOCKED`** | **FALSE** | Zero unresolved blockers in model, data, lineage, hardware, or scientific integrity. |

---

## 7. Actionable Directives & Validation Handoff

1. **STOP at Artifact Readiness:** Formal training is complete. The model is frozen and sealed.
2. **Do Not Unlock Protected Partitions:** The training team shall NOT open `DEV_SEARCH`, `DEV_CAL`, or `TEST_SEALED`.
3. **Handoff to Independent Validation:** Model artifacts, configurations, receipts, and instructions are transferred to the independent validation team per `INDEPENDENT_VALIDATION_HANDOFF.md`.
4. **Production Deployment Prohibition:** The model artifact must not be deployed to user-facing applications until the independent validation gate certifies external FWR control and generalization.
