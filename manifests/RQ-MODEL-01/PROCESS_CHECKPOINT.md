# RQ-MODEL-01 Process Checkpoint: E0 vs E1 Fold 1 Kaggle GPU Benchmark

**Document ID:** `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md`  
**Phase:** `RQ-MODEL-01`  
**Status:** `COMPLETED`  
**Execution Platform:** Remote Kaggle GPU (`machine_shape: NvidiaTeslaT4`, Tesla T4, PyTorch 2.11.0+cu128, FP32 inference)  
**Execution Kernel:** `minkhoaa/langfens-e1-fold1-validation-benchmark` (Version 4)  
**Kernel URL:** https://www.kaggle.com/code/minkhoaa/langfens-e1-fold1-validation-benchmark  
**Source Commit Pinned & Verified:** `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`  
**Date:** 2026-11-03  
**Policy Authority:** `AGENTS.md` (Strict Remote Execution Boundary & Mandatory Process Checkpoint Policy)

---

## 1. Executive Summary & Recommendation

This checkpoint records the successful remote execution of the **E0 (Baseline R) vs E1 (Alignment-Aware Skip MDD) Fold 1 Validation Benchmark** on Kaggle GPU.

### Final Scientific Recommendation: STRICT NO-GO for Cross-Fold Rollout (Folds 2–4)

1. **Primary Target Metric Degraded:** Omission AP degraded from **0.035020** (E0) to **0.032252** (E1) ($\Delta = -0.002768$, **-7.90% relative**).
2. **Diagnostic Operational Metric Degraded:** Omission Recall at 15% FPR degraded from **0.400000 (40.00%)** to **0.365517 (36.55%)** ($\Delta = -0.034483$, **-8.62% relative**, -3.45 percentage points).
3. **Overall Detection Accuracy Degraded:** Full validation Binary AP degraded from **0.317021** to **0.309293** ($\Delta = -0.007728$, **-2.44% relative**).
4. **Architectural Inductive Bias Failure (Skip Mass Collapse):** The mean skip mass on deletions in E1 is **0.000000** ($0.0\%$), and on correct tokens is **0.000031** ($0.0031\%$). The learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ was completely ignored by the cross-attention mechanism during omission phonemes. Zero acoustic skip mass separation emerged.
5. **Auxiliary Metric:** Auxiliary CTC PER improved marginally from **23.68%** to **23.24%** ($\Delta = -0.45$ percentage points), but this did not translate into any omission or mispronunciation detection benefit.
6. **Decision:** Do NOT commit GPU budget or train Folds 2–4 with E1. The skip-token cross-attention formulation does not resolve phoneme omission blindness in MDD.

---

## 2. Checkpoint Provenance & Execution Invariants

| Invariant | Specification / Registered Value | Verification Status |
|---|---|---|
| **Kaggle Kernel ID** | `minkhoaa/langfens-e1-fold1-validation-benchmark` | **PASS** (Version 4, Status: `COMPLETE`) |
| **Hardware** | Nvidia Tesla T4 GPU (16 GB VRAM) | **PASS** (`Tesla T4`, `CUDA:0`) |
| **Precision** | PyTorch FP32 inference (`allow_tf32=False`) | **PASS** (Bit-exact FP32 evaluation) |
| **Execution Scope** | Inference only (0 training steps, 0 backward, 0 optimizer updates) | **PASS** |
| **Sealed Test Isolation** | Active test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) strictly unread | **PASS** (Physical check passed) |
| **Validation Utterances** | Full Fold 1 validation partition: 300 utterances (speakers `NJS`, `RRBI`) | **PASS** (300 / 300 evaluated) |
| **Canonical Tokens** | 9,956 phonemes (8,799 correct, 1,012 substitutions, 145 deletions) | **PASS** (Exact match) |
| **Source Commit** | `baa4989243d5ff46b8b3bf5a3c280cdf2857382b` | **PASS** (Checked out & verified via rev-parse) |
| **E0 Model Weights** | `best.pt` (Baseline R Fold 1, 3,600 updates) | **PASS** (SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`) |
| **E1 Model Weights** | `fixed_budget_checkpoint.pt` (E1 Fold 1, 3,600 updates) | **PASS** (SHA-256: `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`) |
| **Staging Package** | `authorized_RQ1-TRN_fold1_train_val_only.zip` | **PASS** (SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) |

---

## 3. Benchmark Results: E0 vs E1 Head-to-Head Comparison

### 3.1 Primary Registered Metrics

| Metric | E0 (Baseline R) | E1 (Skip MDD) | Absolute Delta ($\Delta$) | Relative Delta (%) | Result |
|---|:---:|:---:|:---:|:---:|:---:|
| **Omission AP** | **0.035020** | 0.032252 | **-0.002768** | **-7.90%** | **Degraded** |
| **Omission Recall @ 15% FPR** | **0.400000** | 0.365517 | **-0.034483** | **-8.62%** | **Degraded** |
| **Binary MDD AP** | **0.317021** | 0.309293 | **-0.007728** | **-2.44%** | **Degraded** |
| **Greedy CTC PER** | 0.236818 | **0.232365** | **-0.004453** | **-1.88%** | Marginal Gain |

### 3.2 Attention Localization & Skip Mass Behavior

| Attention Feature | E0 (Baseline R) | E1 (Skip MDD) | Delta ($\Delta$) | Interpretation |
|---|:---:|:---:|:---:|---|
| **Mean In-Window Conditional Mass** | 0.032023 | 0.035854 | +0.003831 | Modest +11.96% gain; remains diffuse near uniform (0.030883) |
| **Mean Peak Temporal Offset** | 1.191963 s | 1.197316 s | +0.005353 s | Temporal alignment unchanged |
| **Mean Skip Mass on Deletions** | 0.000000 | **0.000000** | +0.000000 | **Total failure to activate on omissions** |
| **Mean Skip Mass on Correct** | 0.000000 | 0.000031 | +0.000031 | Near-zero background activation |
| **Mean Skip Mass on All Tokens** | 0.000000 | 0.000028 | +0.000028 | Skip token collapsed ($< 0.003\%$) |

### 3.3 Secondary & Disaggregated Metrics

| Metric Category | Metric | E0 (Baseline R) | E1 (Skip MDD) | Delta ($\Delta$) |
|---|---|:---:|:---:|:---:|
| **ROC-AUC** | Binary ROC-AUC | 0.751401 | 0.749722 | -0.001679 |
| | Omission ROC-AUC | 0.751803 | 0.730946 | -0.020857 |
| | Substitution ROC-AUC | 0.751343 | 0.752412 | +0.001069 |
| **Protocol Z** | Aligned Tokens (99.63% Coverage) | 9,919 / 9,956 | 9,919 / 9,956 | 0 |
| | Protocol Z Aligned AP | 0.313529 | 0.305736 | -0.007793 |
| **L1: Hindi (`NJS`)** | Binary AP | 0.251518 | 0.245874 | -0.005644 |
| | Omission AP | 0.036553 | 0.034216 | -0.002337 |
| **L1: Spanish (`RRBI`)** | Binary AP | 0.394931 | 0.383344 | -0.011587 |
| | Omission AP | 0.035561 | 0.031820 | -0.003741 |

---

## 4. Root-Cause Analysis of E1 Inductive Bias Collapse

1. **Attention Unconstrained Competition:** The acoustic skip token was parameterized as a single learnable embedding $\mathbf{s} \in \mathbb{R}^{768}$ prepended to the frame sequence $\mathbf{K}, \mathbf{V}$. Because training was supervised by standard multi-task CTC + Binary cross-entropy with whole-utterance audio inputs, the cross-attention queries found stronger local correlation with speech frames (even diffuse frames) than routing mass to the ungrounded static vector $\mathbf{s}$.
2. **Lack of Direct Skip Token Supervision:** While the model had access to an auxiliary skip weight feature in the classification head, the cross-attention weights themselves received no direct loss penalty or contrastive alignment objective forcing $\alpha_{i, \text{skip}} \to 1$ on deleted intervals. Without explicit attention loss, gradient backpropagation through the binary classification head was insufficient to overcome the local acoustic query bias.
3. **Monotonic Temporal Prior Insufficiency:** The Gaussian monotonic distance bias $-\gamma (t/T - i/N)^2$ enforced rough chronological ordering along the diagonal, but did not resolve the fundamental ambiguity between silence/unvoiced segments and phoneme deletions.

---

## 5. Artifact Inventory & Cryptographic Checksums

| Artifact | Location | File Size (Bytes) | SHA-256 Digest |
|---|---|---|---|
| `E1_FOLD1_BENCHMARK_SUMMARY.json` | `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json` | 1,222 | `006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3` |
| `E0 E1_EVALUATION_REPORT.json` | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E0_FOLD1_EVAL/E1_EVALUATION_REPORT.json` | 1,845 | `b2315fef480daa37df2e69f60d6a3594d3bfafbb5589bb68bf00b00476685e3e` |
| `E0 primary_metrics_summary.csv` | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E0_FOLD1_EVAL/primary_metrics_summary.csv` | 425 | `68724a360b7aab2dee74a6974b779ae3ee1e6afd38f7c598de279054d3972eba` |
| `E1 E1_EVALUATION_REPORT.json` | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E1_FOLD1_EVAL/E1_EVALUATION_REPORT.json` | 1,930 | `0d9c6e6a3d63d26f8f3bcc4780389e96494f9ba33a6dccc05c425136c449b0d4` |
| `E1 primary_metrics_summary.csv` | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E1_FOLD1_EVAL/primary_metrics_summary.csv` | 425 | `95e4aaf5cac27b331874a6477b168ce9ebba57bb1d1cb5e4374ebe0758b8270e` |
| `kernel-metadata.json` | `/tmp/langfens-e1-launch/kernel-metadata.json` | 561 | `8ea678736a30c51010372df03dcae3c983d92fe72948c26bbef6ea18685e2978` |
| `langfens-e1-fold1-validation-benchmark.ipynb` | `/tmp/langfens-e1-launch/langfens-e1-fold1-validation-benchmark.ipynb` | 9,339 | `8db8cb0f8982a5a5462cf83fc70977469a538a329fa03aeec9556fb966f97c55` |
| Remote Execution Log | `/tmp/langfens-e1-launch/output/langfens-e1-fold1-validation-benchmark.log` | 74,484 | `8106a72e816a757ea1ffb851b272f2d96c91a0c4f8f01b97bfb9ef617a2dc34a` |

---

## 6. Restart & Safe Continuance Instructions

1. **Preserve Scientific Baseline:** Baseline R remains the authoritative production anchor across all research questions. Under no circumstances should E1 checkpoints replace Baseline R models.
2. **Folds 2–4 Scope:** Folds 2–4 must remain at their Baseline R models; cross-fold rollout of E1 is officially rejected.
3. **Future Research Directions for Omission Modeling:**
   - Explicit alignment-level supervision (e.g., CTC alignment loss or forced alignment boundaries) rather than unconstrained cross-attention.
   - Contrastive unvoiced/silence penalty to prevent acoustic hallucination on deleted phonemes.
   - Moving from unconstrained attention to segment-level or frame-synchronous lattice decoding.
