# LANGFENS RQ1-SM13-FORMAL-TRAIN-01 — Source & Method Freeze

**Protocol:** `LANGFENS-RQ1-SM13-FORMAL-TRAIN-01`  
**Method Designation:** `LANGFENS-SM13-FUSION-V1`  
**Standard:** EVAL-V3.1 (Exact $N$ targets, $N+1$ gaps)  
**Parent Git Revision:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Hardware Lease Profile:** Exclusive NVIDIA GeForce GTX 1650 (4096 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4  
**Certification Status:** **FROZEN & IMMUTABLE — ALL PREREQUISITES VERIFIED**  

---

## 1. Executive Freeze Declaration

Following the formal completion and statistical certification of the method-selection campaign (`RQ1-METHOD-SELECT-02` and confirmatory replication `RQ1-METHOD-SELECT-02R`), method exploration and architecture benchmarking are officially closed. The winning method, **`LANGFENS-SM13-FUSION-V1`**, is here frozen without modification for formal full-TRAIN fitting.

This document records the exact, byte-level specification of the code, data manifests, acoustic backbone, neural architectures, scalar-mixing mechanisms, loss functions, sequence CTC branch, alignment anchors, and evaluation contracts.

---

## 2. Dataset Scope & Boundary Freeze

### 2.1 Training Partition Definition
- **Manifest:** `data/derived/l2arctic/manifests/train.jsonl`
- **Total Utterances:** Exactly **1,800 utterances**
- **Speaker Inventory (12 speakers):** `['ABA', 'ASI', 'BWC', 'EBVS', 'HJK', 'HKK', 'HQTV', 'LXC', 'NJS', 'PNV', 'SKA', 'SVBI']`
- **Target Event Population ($N$):** Exactly **59,562 phoneme targets**
  - Correct ($C$): 50,356 ($84.54\%$)
  - Substitution ($S$): 7,381 ($12.39\%$)
  - Omission ($O$): 1,825 ($3.06\%$)
- **Gap Event Population ($N+1$):** Exactly **61,362 inter-phoneme gaps**
  - No Insertion: 60,802 ($99.09\%$)
  - Insertion ($I$): 560 ($0.91\%$)

### 2.2 Strict Protection of Evaluation Partitions
The following evaluation splits are strictly locked, sealed, and inaccessible during formal training:
1. `data/derived/l2arctic/manifests/dev_search.jsonl` (599 utterances, 4 speakers: NCC, RRBI, YBAA, YDCK) — **SEALED**
2. `data/derived/l2arctic/manifests/dev_cal.jsonl` (300 utterances, 2 speakers: ERMS, THV) — **SEALED**
3. `data/derived/l2arctic/manifests/test_sealed.jsonl` (600 utterances, 4 speakers: BLC, LXC_test, MBH, TLK) — **SEALED**

Zero gold labels, audio files, or alignments from these partitions were inspected or processed during training.

---

## 3. Acoustic Backbone & Feature Representation Freeze

### 3.1 Frozen Self-Supervised Backbone
- **Model Identifier:** `facebook/wav2vec2-base-960h`
- **HuggingFace Hub Snapshot:** `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- **Status:** **100% Frozen.** Zero weights in the CNN feature encoder or 12 transformer blocks are updated during formal training.
- **Hidden Layers:** All 13 hidden states ($l \in \{0, \dots, 12\}$):
  - Layer 0: Output of CNN feature encoder prior to transformer blocks ($768\text{D}$)
  - Layers 1..12: Outputs of transformer blocks 1 through 12 ($768\text{D}$ each)
- **Frame Rate Clock:** 20 ms per frame ($50\text{ Hz}$).

### 3.2 Forced Alignment Anchors
- **Aligner Model:** `charsiu/en_w2v2_fc_10ms` (pretrained frame classification aligner).
- **Alignment Cache:** `experiments/rq1_encoder_bench_01/features/alignment_centers.npz` (SHA256: `3b25e709a3dc8f6452140a85ecf0a2df46797e889416a9ea1ff2b545d6540c10`).
- **Alignment Method:** Monotonic Viterbi forced alignment using canonical intended phonemes and CharSiu frame logits.
- **Leakage Guard:** Zero gold boundary timestamps, realized phone transcriptions, or manual annotations are used.

### 3.3 Target Acoustic Features ($2,304\text{D}$ per layer)
For target token $i$ with alignment center $c_i$:
- Window: 17 frames centered at $c_i$ ($[-8, +8]$ frames, covering $340\text{ ms}$).
- Extracted Acoustic Slice: Central frames $[3:14]$ (11 frames, $220\text{ ms}$).
- Concatenated Feature Vector $\mathbf{f}_{l, i} \in \mathbb{R}^{2304}$:
  1. Mean over central frames: $\frac{1}{11} \sum_{t=3}^{13} \mathbf{h}_{l, i, t} \in \mathbb{R}^{768}$
  2. Population standard deviation: $\sqrt{\frac{1}{11} \sum_{t=3}^{13} (\mathbf{h}_{l, i, t} - \boldsymbol{\mu})^2} \in \mathbb{R}^{768}$
  3. Boundary spectral delta: $\mathbf{h}_{l, i, 13} - \mathbf{h}_{l, i, 3} \in \mathbb{R}^{768}$

### 3.4 Gap Acoustic Features ($1,536\text{D}$ per layer)
For gap position $j \in \{0, \dots, N\}$:
- Left Context: Mean of frames $[11:16]$ from token $j-1$ (zero-padded if $j=0$), $\mathbb{R}^{768}$.
- Right Context: Mean of frames $[1:6]$ from token $j$ (zero-padded if $j=N$), $\mathbb{R}^{768}$.
- Concatenated Feature Vector $\mathbf{g}_{l, j} \in \mathbb{R}^{1536}$.

### 3.5 Feature Normalization Protocol
- Raw multi-layer acoustic representations are fed unstandardized to the scalar-mixing heads.
- Relative acoustic dispersion and energy differentials across transformer depths are preserved.
- Global feature calibration is learned end-to-end via the differentiable scale parameter ($s_T$ and $s_G$).

---

## 4. Head Architecture & Loss Function Freeze

### 4.1 Target Scalar-Mixing Head (`ScalarMixingTargetHead`)
- **Differentiable Mixing:**
  $$\tilde{\mathbf{f}}_i = s_T \sum_{l=0}^{12} w_l \mathbf{f}_{l, i}, \quad w_l = \frac{\exp(\alpha_l)}{\sum_{k=0}^{12} \exp(\alpha_k)}$$
  - Logits $\boldsymbol{\alpha} \in \mathbb{R}^{13}$ initialized to $\mathbf{0}$ (uniform $w_l = 1/13 \approx 0.0769$).
  - Scale parameter $s_T \in \mathbb{R}$ initialized to $1.0$.
- **Reference Embedding:** 39D one-hot vector $\mathbf{r}_i$ for canonical intended phoneme.
- **Combined Input:** $\mathbf{x}_i = [\tilde{\mathbf{f}}_i, \mathbf{r}_i] \in \mathbb{R}^{2343}$.
- **MLP Architecture:**
  - `Linear(2343, 256)` $\to$ `ReLU()` $\to$ `Dropout(0.25)`
  - `Linear(256, 128)` $\to$ `ReLU()` $\to$ `Dropout(0.20)`
  - `Linear(128, 3)` (Outputs: Correct, Substitution, Omission)
- **Total Parameters:** Exactly **633,361 parameters** (14 scalar mixing + 633,347 MLP).
- **Loss:** Weighted Cross-Entropy with class weights $[1.0, 2.5, 4.0]$ for [CORRECT, SUBSTITUTION, OMISSION].
- **Optimizer:** AdamW, learning rate $0.001$, weight decay $0.01$, batch size $512$.
- **Epochs:** Exactly $12$ epochs. Terminal epoch 12 checkpoint selected without in-sample tuning.

### 4.2 Exact-Gap Scalar-Mixing Head (`ScalarMixingGapHead`)
- **Differentiable Mixing:**
  $$\tilde{\mathbf{g}}_j = s_G \sum_{l=0}^{12} v_l \mathbf{g}_{l, j}, \quad v_l = \frac{\exp(\beta_l)}{\sum_{k=0}^{12} \exp(\beta_k)}$$
  - Logits $\boldsymbol{\beta} \in \mathbb{R}^{13}$ initialized to $\mathbf{0}$.
  - Scale parameter $s_G \in \mathbb{R}$ initialized to $1.0$.
- **Reference Context:** 78D vector (39D left reference + 39D right reference).
- **Combined Input:** $\mathbf{y}_j = [\tilde{\mathbf{g}}_j, \mathbf{r}_{\text{gap}, j}] \in \mathbb{R}^{1614}$.
- **MLP Architecture:**
  - `Linear(1614, 128)` $\to$ `GELU()` $\to$ `Dropout(0.15)`
  - `Linear(128, 64)` $\to$ `GELU()`
  - `Linear(64, 2)` (Outputs: No Insertion, Insertion)
- **Total Parameters:** Exactly **215,120 parameters** (14 scalar mixing + 215,106 MLP).
- **Loss:** Weighted Cross-Entropy with inverse square-root frequency weights:
  $$w_c = \sqrt{\frac{N_{\text{total}}}{2 \cdot N_c}}$$
- **Optimizer:** AdamW, learning rate $0.0008$, weight decay $0.0001$, batch size $1024$.
- **Epochs:** Exactly $5$ epochs. Terminal epoch 5 checkpoint selected without in-sample tuning.

---

## 5. CTC Sequence Likelihood Branch & Evidence Integration Freeze

### 5.1 CTC Projection Branch
- **Architecture:** `Linear(768, 43)` projection on frozen Wav2Vec2 layer 6 hidden states.
- **Classes:** 42 phoneme classes + 1 blank token (blank ID = 42).
- **Scoring Engine:** Audited C++ forward-pass batch kernel (`experiments/rq1_method_bench_04/agent_a/ctc_batch.so`).
- **Hypothesis Scoring:** Exact sequence likelihood ratios:
  $$\text{SLLR} = \log P(\text{candidate substitution}) - \log P(\text{canonical})$$
  $$\text{OLLR} = \log P(\text{canonical deletion}) - \log P(\text{canonical})$$
- **Sequence Simplex Probabilities:**
  $$P_{\text{E4}} = \text{softmax}([0, \text{SLLR}, \text{OLLR}])$$

### 5.2 Fixed 50/50 Evidence Integration Rule
The target prediction probability simplex is computed by the fixed, unlearned linear blend:
$$P_{\text{target}}(c) = 0.5 \cdot P_{\text{neural}}(c) + 0.5 \cdot P_{\text{E4}}(c), \quad c \in \{\text{CORRECT}, \text{SUBSTITUTION}, \text{OMISSION}\}$$

---

## 6. Seed Schedule & Primary Artifact Designation

- **Registered Seeds:**
  - `Seed 101`: **Predetermined Primary Production Artifact**
  - `Seed 202`: Confirmatory Multi-Seed Replication
  - `Seed 303`: Confirmatory Multi-Seed Sensitivity
- **Cherry-Picking Prohibition:**
  No seed selection or hyperparameter adjustments were performed based on full-TRAIN in-sample metrics. Seed 101 was designated as primary *ex ante* in `PREREGISTRATION.json`.
