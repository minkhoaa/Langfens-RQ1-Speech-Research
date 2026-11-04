# LANGFENS RQ1 TRN-04: Agent D Verification Report
## T3 Combined Intervention Specification, Factorial Interaction Algebra, and Execution Audit

**Agent:** Agent D — T3 Combined Intervention Lead  
**Workstream Directory:** `/home/khoa/Projects/KLTN/Dataset/workstreams/trn04_agent_d`  
**Target Arm:** T3 (Combined Treatment: Class-Conditioned Focal Loss + 4D Waveform Acoustic Evidence Enabled)  
**Remote Execution Slug:** `minkhoaa/langfens-rq1-trn-04-t3-focal-feat`  
**Execution Platform:** Kaggle Tesla T4 GPU (CUDA FP16 AMP)  
**Verification Date:** November 4, 2026  
**Status:** **AUDIT PASSED / REMOTE TRAINING COMPLETE / BENCHMARKED & CRYPTOGRAPHICALLY VERIFIED / REJECTED AS WINNER (FLOOR EFFECT ARTIFACT IDENTIFIED)**

---

## 1. Executive Summary & Experimental Role

### 1.1 The Factorial Context of Arm T3
LANGFENS RQ1 TRN-04 implements a controlled $2 \times 2$ factorial experiment designed to isolate the individual main effects and interactive contributions of two distinct architectural and supervision interventions for Mispronunciation Detection and Diagnosis (MDD) on the L2-ARCTIC corpus:
1. **Factor A (Supervision Objective):** Standard 3-Way Cross-Entropy (`operation_ce`) versus Class-Conditioned Focal Loss (`class_focal`).
2. **Factor B (Acoustic Evidence Injection):** Additional acoustic features disabled (`features_enabled = False`, zero-valued architectural slot) versus 4D frame-level waveform acoustic features enabled (`features_enabled = True`).

Within this factorial matrix, **Arm T3** represents the **full combined intervention arm** where both treatment factors are simultaneously active:

| Arm | Objective Function | Additional Acoustic Features | Experimental Designation | Role / Lead |
| :---: | :---: | :---: | :---: | :---: |
| **C0** | Uniform Cross-Entropy (`operation_ce`) | Disabled (Zero Slot) | Matched Control Baseline | Agent A |
| **T1** | Class-Conditioned Focal (`class_focal`) | Disabled (Zero Slot) | Objective Treatment Only | Agent B |
| **T2** | Uniform Cross-Entropy (`operation_ce`) | Enabled (4D Waveform Features) | Acoustic Treatment Only | Agent C |
| **T3** | **Class-Conditioned Focal (`class_focal`)** | **Enabled (4D Waveform Features)** | **Full Combined Treatment** | **Agent D (Lead)** |

### 1.2 Core Audit Findings
- **Exact Capacity Matching:** Arm T3 instantiates the common `CompactFactorialOperationHead(FactorialHeadConfig())` yielding **exactly 267,179 trainable parameters** across 17 parameter tensors, matched bit-for-bit to C0, T1, and T2 at initialization (seed 1337). All 17 tensors match `COMMON_INITIALIZATION_MANIFEST.json` SHA-256 hashes bit-for-bit.
- **Simultaneous Activation:** Both interventions operate concurrently. In the forward pass, 4D waveform acoustic descriptors are projected via `acoustic_feature_projection` (`nn.Linear(4, 128, bias=False)`) into the cross-attention key/value streams. In the backward pass, gradients are scaled by the class-conditioned focal loss with focusing parameter $\gamma = 2.0$ and attenuation $\alpha = 0.25$ on coronal stop and rhotic deletions ($\{/d/, /r/, /t/\}$; phone IDs $\{9, 28, 31\}$).
- **Kaggle Deployment Package:** Verified authentic at `deploy/kaggle_trn04_t3/` with target slug `minkhoaa/langfens-rq1-trn-04-t3-focal-feat`, machine shape `NvidiaTeslaT4`, dataset source `minkhoaa/langfens-rq1-fold1-staging` (archive SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`), and private execution.
- **Remote Training Execution:** Kaggle kernel execution completed all 3,600 successful updates across 9 epochs in 493.40 seconds on a Tesla T4 GPU under CUDA FP16 AMP (with 3 skipped AMP gradient overflow attempts during warmup). Checkpoints (`fixed_budget_3600.pt`, `best.pt`, `training_summary.json`) are cryptographically verified and benchmarked in `manifests/RQ1-TRN-04/ALL_EVALUATION_METRICS.json`.
- **Factorial Interaction Formulation:** Under the linear factorial model:
  $$\text{Interaction } (A \times B) = T_3 - T_2 - T_1 + C_0$$
  The computed algebraic interaction on Dedicated Omission AP is $+0.017077$, and on Deployed Precision is $+0.45\text{ pp}$.
- **Floor Effect Artifact Identified:** Rigorous empirical analysis reveals that this positive algebraic interaction is **NOT** a manifestation of constructive super-additive synergy. Instead, it is a **mathematical artifact of a severe floor effect**: both T1 and T3 collapsed to a degenerate decision state (deployed precision $1.46\%$, all-gold recall $100.0\%$, and $985.44$ false alarms per 1,000 canonical tokens). Because T1 was already completely degraded, adding acoustic features in T3 could produce no further collapse.
- **Strict Prohibition of Premature Winner Declaration:** Arm T3 failed both Level A (Pilot Feasibility) and Level B (Research Target) acceptance gates. T3 is **not** a winner and must not be advanced as a viable MDD production candidate.

---

## 2. Hypothesis H3 Formulation & Mathematical Basis

### 2.1 Formal Statement of Hypothesis H3
> **Hypothesis H3 (Factorial Super-Additive Synergy):**  
> *Concurrently modulating the loss objective via Class-Conditioned Focal Loss ($\gamma = 2.0, \alpha = 0.25$ on $\{/d/, /r/, /t/\}$ deletions) and providing synchronized 4D frame-level waveform acoustic evidence will yield a super-additive interaction ($\text{Interaction} = T_3 - T_2 - T_1 + C_0 > 0$) on omission detection precision and dedicated omission average precision (AP), wherein physical acoustic evidence provides the necessary localized boundary closure information that enables the focal loss to recalibrate the decision threshold without collapsing into pathological false-alarm explosion.*

### 2.2 Theoretical & Linguistic Motivation
In second-language pronunciation evaluation, phonetic omission detection suffers from two distinct and interacting failure modes:
1. **Statistical Shortcut Learning (Prior Skew):** In the L2-ARCTIC training set (FIT partition, 1,799 utterances), deletions are heavily concentrated on coronal stops and rhotics: $\{/d/, /r/, /t/\}$ account for **852 of 1,503 total deletions (56.69%)**. Uniform cross-entropy training causes the operation head to exploit canonical prompt phoneme embeddings as a static predictive shortcut, predicting "deletion" whenever a coronal prompt is observed regardless of the acoustic signal.
2. **Acoustic Ambiguity (Spectral Smoothing):** Unreleased coda stops ([t̚, d̚]) and brief alveolar taps ([ɾ], duration 20–40 ms) possess minimal acoustic energy. Pretrained Wav2Vec2 transformer self-attention pools representations broadly across context, smoothing out localized acoustic dips and leading the model to confound unreleased closures or weak taps with genuine omissions.

Hypothesis H3 hypothesized a dual-remediation mechanism:
- **Acoustic Evidence Injection:** Providing instantaneous, un-smoothed waveform physical metrics (normalized log energy, relative RMS, delta log energy, zero-crossing rate) injects physical closure dynamics directly into the attention stream.
- **Focal Loss Modulation:** Reducing the supervision weight ($\alpha = 0.25$) on $\{/d/, /r/, /t/\}$ deletions suppresses the static prompt shortcut, compelling the attention layers to actively query the acoustic stream.
- **Expected Synergy:** Neither intervention was expected to succeed fully in isolation:
  - Acoustic features alone (T2) might be ignored if the network continues relying on the static prompt shortcut.
  - Focal loss alone (T1) might suppress the deletion prior without giving the network the physical evidence needed to distinguish low-energy realizations from true omissions.
  - Together (T3), the two interventions were hypothesized to act in concert: the focal loss forces reliance on acoustic evidence, and the 4D acoustic vector supplies that exact evidence, producing a super-additive performance boost.

---

## 3. Mathematical Formulation of the Factorial Interaction Model

### 3.1 The $2 \times 2$ Factorial Design Matrix
Let $y$ denote an evaluation metric of interest (e.g., Deployed Precision, Dedicated Omission AP, Binary MDD AP). The $2 \times 2$ factorial structure is indexed by two binary treatment factors:
- Factor $A$ (Supervision Objective): $A \in \{0, 1\}$, where $0 = \text{operation\_ce}$ and $1 = \text{class\_focal}$.
- Factor $B$ (Acoustic Evidence): $B \in \{0, 1\}$, where $0 = \text{disabled}$ and $1 = \text{enabled}$.

The four experimental arms correspond to the vertices of this design space:
- $C_0 = y(0, 0)$: Control baseline (Standard Cross-Entropy, Acoustic Features Disabled).
- $T_1 = y(1, 0)$: Objective intervention only (Class Focal Loss, Acoustic Features Disabled).
- $T_2 = y(0, 1)$: Acoustic intervention only (Standard Cross-Entropy, Acoustic Features Enabled).
- $T_3 = y(1, 1)$: Combined intervention (Class Focal Loss, Acoustic Features Enabled).

### 3.2 Main Effects and Interaction Decomposition
The main effects of Factors $A$ and $B$, and their interaction $A \times B$, are mathematically defined as:

$$\text{Main Effect of Objective } (A) = \frac{(T_1 - C_0) + (T_3 - T_2)}{2}$$

$$\text{Main Effect of Acoustic Features } (B) = \frac{(T_2 - C_0) + (T_3 - T_1)}{2}$$

$$\text{Factorial Interaction } (A \times B) = T_3 - T_2 - T_1 + C_0$$

### 3.3 Derivation of the Interaction Term
To understand the scientific meaning of the interaction term, express $T_3$ relative to an additive baseline:
1. Define the isolated treatment effect of $T_1$ over control:
   $$\Delta_1 = T_1 - C_0$$
2. Define the isolated treatment effect of $T_2$ over control:
   $$\Delta_2 = T_2 - C_0$$
3. Under the **linear additive null hypothesis** ($H_0: \text{Interaction} = 0$), the two interventions operate through independent, non-interacting channels. The expected combined performance is strictly the sum of individual gains:
   $$T_3^{\text{additive}} = C_0 + \Delta_1 + \Delta_2 = C_0 + (T_1 - C_0) + (T_2 - C_0) = T_1 + T_2 - C_0$$
4. The interaction is precisely the discrepancy between observed combined performance and the additive expectation:
   $$\text{Interaction } (A \times B) = T_3 - T_3^{\text{additive}} = T_3 - (T_1 + T_2 - C_0) = T_3 - T_2 - T_1 + C_0$$

Equivalently, the interaction measures how the effect of the acoustic intervention changes when the objective intervention is present:
$$\text{Interaction } (A \times B) = (T_3 - T_1) - (T_2 - C_0) = (T_3 - T_2) - (T_1 - C_0)$$

---

## 4. Scientific Taxonomy of Factorial Interactions

In speech ML and experimental phonetics, the sign and magnitude of the interaction term represent distinct causal regimes:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FACTORIAL INTERACTION REGIMES                                  │
├───────────────────┬──────────────────────────────┬───────────────────────────────────────────────┤
│ Regime            │ Algebraic Condition          │ Scientific Interpretation                     │
├───────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ **Super-Additive**│ Interaction > 0              │ **Constructive Synergy:** The presence of one │
│                   │ (for benefit metrics like AP)│ factor unlocks or amplifies the mechanism of  │
│                   │                              │ the other. Combined gain exceeds sum of parts.│
├───────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ **Null (Additive)**│ Interaction ≈ 0              │ **Orthogonal Mechanisms:** Factors operate via│
│                   │                              │ independent channels with zero cross-talk.    │
├───────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ **Sub-Additive**  │ Interaction < 0              │ **Antagonism or Saturation:** Diminishing     │
│                   │                              │ returns, redundant overlap, or destructive    │
│                   │                              │ interference between mechanisms.              │
├───────────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ **Floor / Ceiling**│ Spurious algebraic sign      │ **Pathology Artifact:** Severe performance    │
│ **Artifact**      │ driven by one-sided collapse │ collapse at boundary prevents expression of   │
│                   │                              │ second factor, creating false synergy.        │
└───────────────────┴──────────────────────────────┴───────────────────────────────────────────────┘
```

### 4.1 Super-Additive Interaction ($\text{Interaction} > 0$)
- **Scientific Meaning:** Positive synergy. Factor $B$ (acoustic evidence) supplies the physical grounding that Factor $A$ (focal loss) requires to recalibrate decision boundaries. The combined intervention achieves a breakthrough that neither intervention could approach alone.
- **Algebraic Requirement:** $T_3 - C_0 > (T_1 - C_0) + (T_2 - C_0)$.

### 4.2 Sub-Additive Interaction ($\text{Interaction} < 0$)
- **Scientific Meaning:** Antagonistic interference or functional redundancy.
  - *Redundancy:* Both interventions provide the same correction. Once focal loss suppresses coronal false alarms, acoustic closure evidence provides no additional discriminant information.
  - *Antagonism:* One intervention actively impairs the operating conditions of the other. For example, aggressive focal down-weighting might warp representation space such that acoustic feature projections cannot be effectively integrated by cross-attention.
- **Algebraic Requirement:** $T_3 - C_0 < (T_1 - C_0) + (T_2 - C_0)$.

### 4.3 Null Interaction ($\text{Interaction} = 0$)
- **Scientific Meaning:** Strict linear additivity. The loss modification operates strictly at the global decision threshold level, while the acoustic feature projection operates strictly at the local frame representation level, with no coupling between them.
- **Algebraic Requirement:** $T_3 = T_1 + T_2 - C_0$.

---

## 5. Architectural Alignment & Gate 0 Invariant Audit

To ensure that factorial interaction calculations are unconfounded by architectural capacity, optimization hyperparameters, or data leakage, all four arms strictly adhere to the frozen Gate 0 experimental contracts.

### 5.1 Common Parameter Inventory Parity (267,179 Parameters)
All four arms instantiate `CompactFactorialOperationHead(FactorialHeadConfig())` with **exactly 267,179 trainable parameters** across 17 tensors. No parameters were added to T2/T3 or removed from C0/T1.

```
Universal Architectural Slot:
CompactFactorialOperationHead(
  acoustic_projection:         Linear(768 -> 128, bias=True)     [98,432 params]
  acoustic_feature_projection: Linear(4 -> 128,   bias=False)    [   512 params]  <-- Universal Slot
  phone_embedding:             Embedding(40 -> 128)              [ 5,120 params]
  position_embedding:          Embedding(256 -> 128)             [32,768 params]
  cross_attention:             MultiheadAttention(embed_dim=128) [65,920 params]
  acoustic_norm:               LayerNorm(128)                    [   256 params]
  operation_head:              Sequential(Linear, GeLU, Linear)  [33,283 params]
  ctc_head:                    Linear(768 -> 40, bias=True)      [30,760 params]
)
Total Trainable Head Parameters: 267,179 (Bit-identical at init, seed 1337)
```

### 5.2 Universal Acoustic Slot Mechanics
The acoustic projection layer `self.acoustic_feature_projection = nn.Linear(4, 128, bias=False)` is instantiated in every arm:
- In feature-disabled arms ($C_0, T_1$): input is $\mathbf{0} \in \mathbb{R}^{B \times T \times 4}$. Since `bias=False`:
  $$W_{\text{acoust}} \cdot \mathbf{0} \equiv \mathbf{0} \in \mathbb{R}^{B \times T \times 128}$$
  Zero extra representation is injected into attention keys/values, and backpropagation yields $\nabla_{W_{\text{acoust}}} \mathcal{L} \equiv \mathbf{0}$.
- In feature-enabled arms ($T_2, T_3$): input is the discrete 4D waveform acoustic vector $\mathbf{x}_{\text{acoust}}[t]$. The projection $W_{\text{acoust}} \mathbf{x}_{\text{acoust}}[t]$ is active and receives continuous gradient updates ($\nabla_{W_{\text{acoust}}} \mathcal{L} \neq \mathbf{0}$) throughout training.

### 5.3 Bit-Identical Quad Initialization Audit
Bit-for-bit identity across all 17 parameter tensors was verified against `COMMON_INITIALIZATION_MANIFEST.json` before training:

| Tensor Index | Parameter Tensor Name | Tensor Shape | Trainable | Trainable Numel | Pinned SHA-256 Checksum | Init Match |
|:---:|---|---|:---:|---:|---|:---:|
| 0 | `acoustic_projection.weight` | `[128, 768]` | True | 98,304 | `799ca9b14f826aa76d4a8da991e13a967fba9ad1...` | **MATCH** |
| 1 | `acoustic_projection.bias` | `[128]` | True | 128 | `1eafe3fdfd47c40c83a00247656d027a070eb374...` | **MATCH** |
| 2 | `acoustic_feature_projection.weight` | `[128, 4]` | True | 512 | `3c8f3ea7f4d2417cf6dfae8cb4e650ef377c8e96...` | **MATCH** |
| 3 | `phone_embedding.weight` | `[40, 128]` | True | 5,120 | `e5bc7d08ba067a9992f96cf9f80a42cefe0386cf...` | **MATCH** |
| 4 | `position_embedding.weight` | `[256, 128]` | True | 32,768 | `00155b1f0fe4e803fb2da764a7dae373cf58bb0b...` | **MATCH** |
| 5 | `attention.in_proj_weight` | `[384, 128]` | True | 49,152 | `ff991d9047b19ce8a6e768e82672a912bb0bc8eb...` | **MATCH** |
| 6 | `attention.in_proj_bias` | `[384]` | True | 384 | `5618776846fcddc8fe05d15f16fcbf9d0ef9dfa8...` | **MATCH** |
| 7 | `attention.out_proj.weight` | `[128, 128]` | True | 16,384 | `7a4993444458f4a13d29a54ea720543e49e984f1...` | **MATCH** |
| 8 | `attention.out_proj.bias` | `[128]` | True | 128 | `471d871790ae305373a0eeb9bb408f6d6ee4a88f...` | **MATCH** |
| 9 | `acoustic_norm.weight` | `[128]` | True | 128 | `9d554a99d4948f2cbf8e5c85d8525b5aeae2c676...` | **MATCH** |
| 10 | `acoustic_norm.bias` | `[128]` | True | 128 | `299fc8c4eb5ca760e9d1bf7ee0b3b428d098198f...` | **MATCH** |
| 11 | `operation_head.0.weight` | `[128, 256]` | True | 32,768 | `ef938450f38b1f8eb4f94bc3c8cf35ef9e5fca73...` | **MATCH** |
| 12 | `operation_head.0.bias` | `[128]` | True | 128 | `fcf533d13c72b226f97ef78d21c33f2bbfa090b8...` | **MATCH** |
| 13 | `operation_head.3.weight` | `[3, 128]` | True | 384 | `2a84976aa34b17a5b3a4c0cfb633bfbcf1c1fcf8...` | **MATCH** |
| 14 | `operation_head.3.bias` | `[3]` | True | 3 | `ede9f73d5d01d94c6bc90eb8a86b4b578dd7da44...` | **MATCH** |
| 15 | `ctc_head.weight` | `[40, 768]` | True | 30,720 | `5126924d8744b73799428c17834cad3553f1e683...` | **MATCH** |
| 16 | `ctc_head.bias` | `[40]` | True | 40 | `331adda6622a9f62bc88c8987600bc4578b70f74...` | **MATCH** |
| **Quad** | **Total Trainable Parameters** | — | — | **267,179** | **Verified Bit-Identical across C0, T1, T2, T3** | **PASS** |

### 5.4 Acoustic Feature Invariants & Leakage Controls
- **4D Vector:** $\mathbf{x}_{\text{acoust}}[t] = [f_1(t), f_2(t), f_3(t), f_4(t)]^T$:
  - $f_1$: Log frame energy normalized by FIT training set constants: $\mu_{\text{FIT}} = -6.047091$, $\sigma_{\text{FIT}} = 2.544367$.
  - $f_2$: Relative RMS energy normalized to utterance peak: $\text{RMS}_t / (\max_\tau \text{RMS}_\tau + \epsilon) \in [0, 1]$.
  - $f_3$: Delta log frame energy: $\Delta E_t = E_t - E_{t-1}$.
  - $f_4$: Frame zero-crossing rate: $\text{ZCR}_t \in [0, 1]$.
- **Wav2Vec2 Alignment:** Window $M=400$ ($25\text{ ms}$), Hop $H=320$ ($20\text{ ms}$), 50 Hz frame rate. Output frame count strictly matches `wav2vec2._get_feat_extract_output_lengths(L)`.
- **Zero Leakage:** Feature extraction operates solely on raw audio arrays. No TextGrid intervals, realization alignments, phone boundaries, or error annotations are referenced.

### 5.5 Class-Conditioned Focal Loss Formulation
For prompt token $i$ with canonical phone $p_i$, target operation $y_i \in \{0, 1, 2\}$ (0=Correct, 1=Substitution, 2=Deletion), and predicted probability $P(y_i \mid \mathbf{z}_i)$:
$$\mathcal{L}_{\text{cond}} = -\frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \alpha(p_i, y_i) \cdot (1 - P(y_i \mid \mathbf{z}_i))^\gamma \cdot \log(P(y_i \mid \mathbf{z}_i) + \epsilon)$$
where:
- Focusing parameter: $\gamma = 2.0$.
- Class conditioning weights:
  $$\alpha(p_i, y_i) = \begin{cases} 0.25 & \text{if } y_i = 2 \text{ and } p_i \in \{9, 28, 31\} \ (/d/, /r/, /t/) \\ 1.00 & \text{otherwise} \end{cases}$$
- AMP Safety: Evaluated in `torch.float32` before loss scaling.

---

## 6. Preflight Validation & Kaggle Deployment Package Verification

### 6.1 Kaggle Kernel Deployment Specification
The deployment package for Arm T3 was verified at `deploy/kaggle_trn04_t3/`:
- **Kernel Metadata (`kernel-metadata.json`):**
  - Slug: `minkhoaa/langfens-rq1-trn-04-t3-focal-feat`
  - Accelerator: `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`
  - Privacy: `is_private: true`
  - Dataset Sources: `minkhoaa/langfens-rq1-fold1-staging`
  - SHA-256 Checksum: `e80188d008f398f313e3ecfdae8290b81a14b13514a6b0dc5848116860aa325a`
- **Execution Notebook (`kaggle_train.ipynb`):**
  - SHA-256 Checksum: `bda1793c75a857103904378cb5d32dbd02f0dad60cbbe0837b786c516f8cd28c`
  - Phase 1: Hardware verification (`nvidia-smi`, Tesla T4 detection, CUDA FP16 AMP assert).
  - Phase 2: Credentials resolution via `UserSecretsClient` or staged `/kaggle/input` token.
  - Phase 3: Git checkout of branch `review/rq1-trn-04-factorial-training`.
  - Phase 4: Authorized dataset staging archive verification (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
  - Phase 5: Invocation of official training entrypoint:
    ```bash
    python audit_scripts/train_trn04_factorial.py \
      --arm T3 \
      --split-manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
      --zip-path /kaggle/input/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.bin \
      --output-dir /kaggle/working/checkpoints/RQ1-TRN-04/t3 \
      --seed 1337 \
      --batch-size 2 \
      --gradient-accumulation-steps 2 \
      --warmup-updates 100 \
      --max-updates 3600 \
      --lr 0.001 \
      --weight-decay 0.01 \
      --gradient-clip 1.0 \
      --max-epochs 25 \
      --remote-mirror-target minkhoaa/langfens-rq1-cross-fold-checkpoints \
      --remote-mirror-backend huggingface
    ```

### 6.2 Preflight Forward/Backward Gradient Sanity
Preflight verification script `audit_t3_combined_intervention.py` confirmed that under synthetic input, the combined forward and backward step executes without numerical overflow or NaN values, propagating active gradients simultaneously into:
- The acoustic projection weight: $\nabla_{W_{\text{acoust}}} \mathcal{L} \neq \mathbf{0}$ ($\|\nabla_{W_{\text{acoust}}}\|_2 > 0$).
- The operation classification head: $\nabla_{W_{\text{op}}} \mathcal{L} \neq \mathbf{0}$.
- The CTC phone classification head: $\nabla_{W_{\text{ctc}}} \mathcal{L} \neq \mathbf{0}$.

---

## 7. Remote Training Execution & Checkpoint Durability Audit

### 7.1 Remote Execution Summary
Remote training executed on Kaggle Tesla T4 GPU under CUDA FP16 AMP. The training summary receipt (`training_summary.json`, SHA-256 `247bcaeb32fd44b6ff6c42790728faa626d5c7c259f09fa3479d11e87a0355f8`) confirms:
- **Total Successful Updates:** Exactly **3,600 updates** (matching fixed budget).
- **Total Epochs:** 9 epochs completed (447 updates/epoch).
- **Total Micro-Steps:** 7,206 steps (effective batch size 4).
- **Execution Duration:** 493.40 seconds (~8.2 minutes).
- **Skipped AMP Attempts:** Exactly 3 attempts skipped during early warmup (Epoch 1) due to dynamic GradScaler inf/NaN protection, after which scaler stabilized at $16,384.0$.
- **Best Validation Selection AP:** $0.467060$ achieved at Epoch 5 (Update 2,247).
- **Final Learning Rate:** $0.0008699$ (cosine decay from $0.001$).
- **Completion Timestamp:** `2026-11-04T07:02:37.490077+00:00`.

### 7.2 Epoch-by-Epoch Convergence Trajectory
The convergence progression of Arm T3 across its 9 epochs:

| Epoch | Successful Updates | Cum. Attempts | Skipped AMP | SELECT AP | Blank Fraction | Greedy PER | Task Loss | CTC Loss | Learning Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 447 | 450 | 3 | 0.451975 | 0.839009 | 0.329384 | 0.143683 | 3.501207 | 0.000999 |
| 2 | 897 | 900 | 3 | 0.456350 | 0.838120 | 0.287912 | 0.128323 | 1.047663 | 0.000993 |
| 3 | 1,347 | 1,350 | 3 | 0.455068 | 0.839177 | 0.271905 | 0.128099 | 0.960233 | 0.000983 |
| 4 | 1,797 | 1,800 | 3 | 0.459286 | 0.838258 | 0.257562 | 0.127022 | 0.907706 | 0.000968 |
| **5 (Best)** | **2,247** | **2,250** | **3** | **0.467060** | **0.837906** | **0.254963** | **0.125698** | **0.871514** | **0.000950** |
| 6 | 2,697 | 2,700 | 3 | 0.444533 | 0.838396 | 0.249558 | 0.125883 | 0.843619 | 0.000927 |
| 7 | 3,147 | 3,150 | 3 | 0.439591 | 0.836374 | 0.246544 | 0.124698 | 0.822563 | 0.000900 |
| 8 | 3,597 | 3,600 | 3 | 0.455163 | 0.836956 | 0.239892 | 0.124302 | 0.805840 | 0.000870 |
| **9 (3,600)** | **3,600** | **3,603** | **3** | **0.457589** | **0.837783** | **0.240619** | **0.132413** | **0.787973** | **0.000870** |

### 7.3 Cryptographic Checkpoint Durability
Both official dual checkpoints were downloaded and cryptographically verified:
1. **Fixed-Budget 3,600 Updates Checkpoint (`fixed_budget_3600.pt`):**
   - File Size: $3,243,408\text{ bytes}$
   - SHA-256 Checksum: `ca32da07eac20946af125539ca1ef7ca3835a35ce05e1ec4897d3a790e90eeb9`
   - State Content: Verified complete with `head` state dict (17 parameter tensors), AdamW `optimizer` state (step 3600), cosine `scheduler` state, GradScaler state (`scale: 16384.0`), full RNG state dictionaries (Python, NumPy, PyTorch CPU, PyTorch CUDA), and metadata tagging.
2. **Validation-Selected Checkpoint (`best.pt`):**
   - File Size: $3,239,079\text{ bytes}$
   - SHA-256 Checksum: `1feb311ec9f88569915fa43aefaa923e24c55f7fadb77a821a2b2c2c8dcf6107`
   - Captured at: Epoch 5 (Update 2,247, Selection AP $0.467060$).

---

## 8. Empirical Factorial Effects Decomposition & Interaction Analysis

### 8.1 Primary Factorial Decomposition Table
Evaluation on the unseen Fold-1 VAL partition (600 utterances: NJS, RRBI; 9,956 supervised canonical tokens; 145 gold deletions) yields the complete factorial matrix:

| Metric Channel | Arm C0 (Control) | Arm T1 (Objective) | Arm T2 (Acoustic) | Arm T3 (Combined) | Main Effect: Objective ($A$) | Main Effect: Acoustic ($B$) | Factorial Interaction ($A \times B$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Strict Deployed Precision** | 7.55% | 1.46% | 7.09% | 1.46% | **-5.86 pp** | **-0.23 pp** | **+0.45 pp** |
| **All-Gold Omission Recall** | 60.00% | 100.00% | 51.03% | 100.00% | **+44.48 pp** | **-4.48 pp** | **+8.97 pp** |
| **Dedicated Omission AP** | 0.113331 | 0.027572 | 0.096293 | 0.027611 | **-0.077220** | **-0.008500** | **+0.017077** |
| **False Alarms per 1k Phones**| 107.07 | 985.44 | 97.33 | 985.44 | **+883.24** | **-4.87** | **+9.74** |
| **Binary MDD AP** | 0.321619 | 0.315171 | 0.321573 | 0.315847 | **-0.006087** | **+0.000315** | **+0.000721** |
| **Legacy Substitution AP** | 0.310022 | 0.311767 | 0.312209 | 0.312187 | **+0.000861** | **+0.001304** | **-0.001767** |
| **Greedy CTC PER** | 0.238943 | 0.239045 | 0.239045 | 0.239045 | **+0.000051** | **+0.000051** | **-0.000101** |
| **Acoustic Reliance Gap** | 0.019901 | 0.014102 | 0.022486 | 0.017230 | **-0.005527** | **+0.002856** | **+0.000544** |

### 8.2 Detailed Metric-by-Metric Breakdown

#### 1. Strict Deployed Precision
- $C_0 = 7.5455\%$, $T_1 = 1.4564\%$, $T_2 = 7.0949\%$, $T_3 = 1.4564\%$.
- Main Objective Effect: $\frac{(1.46 - 7.55) + (1.46 - 7.09)}{2} = -5.86\text{ pp}$.
- Main Acoustic Effect: $\frac{(7.09 - 7.55) + (1.46 - 1.46)}{2} = -0.23\text{ pp}$.
- Factorial Interaction: $1.4564 - 7.0949 - 1.4564 + 7.5455 = \mathbf{+0.4506\text{ pp}}$.

#### 2. Dedicated Omission AP
- $C_0 = 0.113331$, $T_1 = 0.027572$, $T_2 = 0.096293$, $T_3 = 0.027611$.
- Main Objective Effect: $\frac{(0.027572 - 0.113331) + (0.027611 - 0.096293)}{2} = \mathbf{-0.077220}$.
- Main Acoustic Effect: $\frac{(0.096293 - 0.113331) + (0.027611 - 0.027572)}{2} = \mathbf{-0.008500}$.
- Factorial Interaction: $0.027611 - 0.096293 - 0.027572 + 0.113331 = \mathbf{+0.017077}$.

#### 3. False Alarm Explosion
- $C_0 = 107.07$, $T_1 = 985.44$, $T_2 = 97.33$, $T_3 = 985.44$ false alarms per 1,000 canonical tokens.
- Both T1 and T3 trigger alarms on **9,811 non-omission tokens out of 9,956 canonical tokens (98.54%)**!
- Main Objective Effect: $\mathbf{+883.24\text{ false alarms / 1,000 tokens}}$.
- Factorial Interaction: $\mathbf{+9.74\text{ false alarms / 1,000 tokens}}$.

---

## 9. Scientific Interpretation: The Floor Effect Artifact

### 9.1 The Apparent Paradox of the Positive Interaction Term
A superficial, uncritical reading of the interaction calculation might conclude:
> *"The interaction term for Dedicated Omission AP is positive ($+0.017077$), and the interaction for Deployed Precision is positive ($+0.45\text{ pp}$). Therefore, combining the objective with acoustic features demonstrates super-additive synergy ($H_3$ supported)."*

**This conclusion is completely false and unscientific.**

### 9.2 Mathematical Anatomy of the Floor Effect
To demonstrate why $+0.017077$ is an artifact, rewrite the interaction formula by grouping terms:
$$\text{Interaction } (A \times B) = (T_3 - T_1) + (C_0 - T_2)$$

Now evaluate each term with the empirical numbers from the evaluation matrix:
1. Difference between T3 and T1 (the marginal effect of acoustic features under focal loss):
   $$T_3 - T_1 = 0.027611 - 0.027572 = \mathbf{+0.000039} \approx 0$$
   The acoustic features produced virtually zero change in Dedicated Omission AP when applied on top of focal loss.
2. Difference between C0 and T2 (the negative marginal effect of acoustic features under cross-entropy):
   $$C_0 - T_2 = 0.113331 - 0.096293 = \mathbf{+0.017038}$$
3. Summing the two components:
   $$\text{Interaction} = 0.000039 + 0.017038 = \mathbf{+0.017077}$$

**Notice that:**
$$\frac{C_0 - T_2}{\text{Interaction}} = \frac{0.017038}{0.017077} = \mathbf{99.77\%}$$

Over **99.7% of the positive interaction term** comes entirely from the fact that acoustic features caused a small drop from $C_0$ to $T_2$ ($-0.017038$), but could cause **no further drop** from $T_1$ to $T_3$ because $T_1$ was already completely collapsed at the empirical floor ($0.0276$).

### 9.3 The Floor Mechanism in Deployed Precision
The identical artifact appears in strict deployed precision:
- $T_3 - T_1 = 1.4564\% - 1.4564\% = \mathbf{0.0000\text{ pp}}$.
- $C_0 - T_2 = 7.5455\% - 7.0949\% = \mathbf{+0.4506\text{ pp}}$.
- $\text{Interaction} = (T_3 - T_1) + (C_0 - T_2) = 0.0000 + 0.4506 = \mathbf{+0.4506\text{ pp}}$.

Because the focal loss drove precision down to the absolute floor of random guessing ($1.46\%$, which is simply the prevalence of true omissions in the VAL set: $145 / 9,956 = 1.4564\%$), performance could not drop below $1.46\%$. The positive interaction $+0.45\text{ pp}$ is purely the drop in T2 that was bounded from repeating in T3.

### 9.4 Physical Reality: Total Logit Displacement
Why did T1 and T3 collapse to the exact prevalence floor?
- In the class-conditioned focal loss, attenuating coronal and rhotic deletions by $\alpha = 0.25$ inadvertently distorted the inter-class margin between the deletion class ($y=2$) and the correct class ($y=0$).
- Rather than suppressing deletion false alarms on coronals, the loss landscape experienced severe gradient imbalance, driving the model's uncalibrated deletion logit upwards across the entire phoneme inventory.
- As a result, the model fired "deletion" on almost every prompt token in VAL (9,956 alarms out of 9,956 tokens; 145 true positives, 9,811 false alarms).
- In this degenerate state, the network acts as a trivial detector ($f(x) \equiv 1$). Recall is trivially $100.0\%$, precision is trivially equal to the dataset prior ($1.46\%$), and the presence of 4D waveform acoustic features is completely bypassed by the saturated logits.

Therefore, **Hypothesis H3 is decisively REJECTED**. There is zero constructive synergy between the class-conditioned focal loss and acoustic features.

---

## 10. Strict Acceptance Gate Evaluation & Non-Winner Status

### 10.1 Acceptance Gate Audit
Both Level A (Pilot Feasibility) and Level B (Research Target) acceptance criteria were evaluated for Arm T3 against the matched control baseline C0:

| Gate Identifier | Acceptance Criterion | Threshold | Arm C0 (Control) | Arm T3 (Combined) | Gate Status |
|:---:|---|:---:|:---:|:---:|:---:|
| **Gate A.1** | Strict Deployed Precision | $\ge 25.0\%$ | 7.55% | **1.46%** | **FAILED** |
| **Gate A.2** | All-Gold Omission Recall | $\ge 30.0\%$ | 60.00% | 100.00%* | *Trivial |
| **Gate A.3** | False Alarms per 1,000 Phones | $\le 30.0$ | 107.07 | **985.44** | **FAILED** |
| **Gate A.4** | Binary MDD AP Non-Inferiority | $\ge \text{C0} - 0.02$ | 0.3216 | 0.3158 | **PASSED** |
| **Gate A.5** | Substitution AP Non-Inferiority | $\ge \text{C0} - 0.02$ | 0.3100 | 0.3122 | **PASSED** |
| **Gate A.6** | Greedy CTC PER Non-Inferiority | $\le \text{C0} + 0.02$ | 0.2389 | 0.2390 | **PASSED** |
| **Level A** | **Overall Pilot Feasibility** | **All A Gates** | **FAILED** | **FAILED** | **OVERALL FAILED** |
| **Gate B.1** | Strict Deployed Precision | $\ge 30.0\%$ | 7.55% | **1.46%** | **FAILED** |
| **Gate B.2** | All-Gold Omission Recall | $\ge 40.0\%$ | 60.00% | 100.00%* | *Trivial |
| **Gate B.3** | False Alarms per 1,000 Phones | $\le 15.0$ | 107.07 | **985.44** | **FAILED** |
| **Gate B.4** | Acoustic Reliance Gap Positive | $> 0$ | +0.0199 | +0.0172 | **PASSED** |
| **Level B** | **Overall Research Target** | **All B Gates** | **FAILED** | **FAILED** | **OVERALL FAILED** |

### 10.2 Explicit Confirmation: T3 Is Not Declared a Winner
Agent D hereby issues an explicit, formal scientific determination:
1. **Arm T3 is NOT a winner.**
2. Arm T3 exhibits severe, unacceptable false alarm explosion ($985.44$ false alarms per 1,000 canonical tokens; 98.54% of all tokens flagged as deleted).
3. Arm T3's deployed precision ($1.46\%$) is 5.2x worse than the baseline control C0 ($7.55\%$) and 4.9x worse than T2 ($7.09\%$).
4. Arm T3's dedicated omission AP ($0.0276$) is 4.1x worse than C0 ($0.1133$) and 3.5x worse than T2 ($0.0963$).
5. The apparent algebraic interaction ($+0.017077$) is a mathematically proven floor effect artifact and must never be cited as evidence of model superiority or synergistic efficacy.

---

## 11. Cross-Arm Comparative Synthesis (C0 vs T1 vs T2 vs T3)

To contextualize Arm T3's findings, the complete comparative synthesis across all four arms at the fixed budget of 3,600 updates is summarized below:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  CROSS-ARM COMPARATIVE PERFORMANCE (3,600 UPDATES)                             │
├──────────────────────┬──────────────────┬──────────────────┬──────────────────┬────────────────────────────────┤
│ Evaluation Dimension │ Control (C0)     │ Focal Only (T1)  │ Acoustic (T2)    │ Combined (T3)                  │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────────────────────────┤
│ Objective            │ operation_ce     │ class_focal      │ operation_ce     │ class_focal                    │
│ Acoustic Features    │ Disabled (Zeros) │ Disabled (Zeros) │ Enabled (4D Wave)│ Enabled (4D Wave)              │
│ Total Parameters     │ 267,179          │ 267,179          │ 267,179          │ 267,179                        │
│ Deployed Precision   │ 7.55%            │ 1.46% (Floor)    │ 7.09%            │ 1.46% (Floor)                  │
│ Omission Recall      │ 60.00%           │ 100.00% (Trivial)│ 51.03%           │ 100.00% (Trivial)              │
│ Dedicated Omission AP│ 0.113331         │ 0.027572         │ 0.096293         │ 0.027611                       │
│ False Alarms / 1k    │ 107.07           │ 985.44           │ 97.33 (Best)     │ 985.44                         │
│ Binary MDD AP        │ 0.321619         │ 0.315171         │ 0.321573         │ 0.315847                       │
│ Substitution AP      │ 0.310022         │ 0.311767         │ 0.312209 (Best)  │ 0.312187                       │
│ Greedy CTC PER       │ 0.238943 (Best)  │ 0.239045         │ 0.239045         │ 0.239045                       │
│ Acoustic Reliance Gap│ +0.019901        │ +0.014102        │ +0.022486 (Best) │ +0.017230                      │
│ Acceptance Status    │ FAILED           │ FAILED           │ FAILED           │ FAILED                         │
└──────────────────────┴──────────────────┴──────────────────┴──────────────────┴────────────────────────────────┤
```

### Key Comparative Insights:
1. **The Objective Factor Dominates:** Whenever `class_focal` is active (T1 and T3), omission precision collapses to $1.46\%$ and false alarms explode to $985.44/1\text{k}$. The objective factor completely overwhelms the acoustic factor.
2. **The Acoustic Factor Provides Genuine Grounding Only Under Standard CE:** In T2 (where `operation_ce` was held constant), acoustic features successfully reduced false alarms by $-9.1\%$ relative ($97.33$ vs $107.07/1\text{k}$) and increased acoustic reliance by $+13.0\%$ relative ($+0.022486$ vs $+0.019901$). However, this acoustic grounding was accompanied by an $8.97\text{ pp}$ drop in omission recall ($51.03\%$ vs $60.00\%$), leading to a net drop in omission AP ($0.0963$ vs $0.1133$).
3. **No Arm Met the Pilot Feasibility Standard:** Even the best control arm (C0) achieved only $7.55\%$ precision against a target of $\ge 25\%$. The fundamental bottleneck in automated omission detection on L2 English speech remains unresolved by single-head local thresholding or localized frame feature concatenation.

---

## 12. Verification Artifacts & Cryptographic Audit Manifest

The following official verification artifacts have been authored, executed, and validated in `/home/khoa/Projects/KLTN/Dataset/workstreams/trn04_agent_d/`:

| Artifact Path | Format | Role | Checksum (SHA-256) / Status |
|---|---|---|---|
| `audit_t3_combined_intervention.py` | Python Script | Executable audit of parameters, quad identity, gradients, deployment, and algebra | Executed, Exit Code 0 |
| `test_t3_combined_intervention.py` | Pytest Suite | 7 comprehensive unit/regression tests validating all T3 contracts & interaction | 7 Passed in 2.55s |
| `t3_combined_audit_receipt.json` | JSON Receipt | Cryptographic audit receipt with full metric decomposition & status | Authentic, Generated UTC |
| `AGENT_D_REPORT.md` | Markdown Report | Authoritative scientific report on T3 intervention & factorial algebra | Complete |

### Remote Deployment & Checkpoint Hashes:
- Kaggle Metadata (`deploy/kaggle_trn04_t3/kernel-metadata.json`): `e80188d008f398f313e3ecfdae8290b81a14b13514a6b0dc5848116860aa325a`
- Kaggle Notebook (`deploy/kaggle_trn04_t3/kaggle_train.ipynb`): `bda1793c75a857103904378cb5d32dbd02f0dad60cbbe0837b786c516f8cd28c`
- Training Summary (`checkpoints/RQ1-TRN-04/t3/training_summary.json`): `247bcaeb32fd44b6ff6c42790728faa626d5c7c259f09fa3479d11e87a0355f8`
- Fixed-Budget Checkpoint (`checkpoints/RQ1-TRN-04/t3/fixed_budget_3600.pt`): `ca32da07eac20946af125539ca1ef7ca3835a35ce05e1ec4897d3a790e90eeb9`
- Best Checkpoint (`checkpoints/RQ1-TRN-04/t3/best.pt`): `1feb311ec9f88569915fa43aefaa923e24c55f7fadb77a821a2b2c2c8dcf6107`

---

## 13. Conclusions & Recommended Next Steps

### 13.1 Scientific Conclusions
1. **Hypothesis H3 is Conclusively Rejected:** Combining class-conditioned focal loss with 4D waveform acoustic features fails to yield constructive super-additive synergy. The positive algebraic interaction ($+0.017077$) is rigorously proven to be a mathematical artifact of a floor effect.
2. **Pathology of Targeted Loss Attenuation:** Selectively attenuating deletion loss weights ($\alpha=0.25$) on frequent deletion phonemes $(\{/d/, /r/, /t/\})$ destabilizes logit calibration, producing indiscriminate false-alarm triggering across all phoneme classes.
3. **Acoustic Evidence Inefficacy under Saturated Logits:** Acoustic feature representations cannot overcome severe logit displacement. When the classification head defaults to predicting omissions globally, downstream attention representations are ignored.

### 13.2 Recommended Next Steps for LANGFENS Research
1. **Halt Exploration of Static Class-Conditioned Loss Modulations:** Do not pursue further heuristic down-weighting ($\alpha$) or focusing ($\gamma$) modifications on the operation classification head. Static loss re-weighting fails to address the underlying conditional probability calibration problem.
2. **Preserve Arm T2 Acoustic Representations for Downstream Architectures:** While T2 failed Level A precision gates, its acoustic reliance gap was the highest of all arms ($+0.022486$, $+13\%$ over C0) and its false alarms were the lowest ($97.33/1\text{k}$). The 4D acoustic vector provides genuine physical closure information that should be integrated into higher-capacity dual-stream or multi-stage architectures rather than a single linear projection.
3. **Shift to Two-Stage Detection / Adjudication:** Because the single-stage detector is overwhelmed by the 90.7% prior of canonical correct speech, adopt a hierarchical architecture:
   - Stage 1: Broad mispronunciation candidate proposer (high recall).
   - Stage 2: Specialized acoustic boundary adjudicator focused purely on closure duration, burst energy, and voicing continuity to confirm or reject proposed omissions.
4. **Advance to Cross-Arm Synthesis & Handoff to Agent E:** Integrate Agent D's factorial findings into the project-wide factorial effect synthesis (`FACTORIAL_EFFECT_ANALYSIS.md`) and support the Lead Orchestrator in finalizing the RQ1-TRN-04 research milestone.
