# LANGFENS RQ1-TRN-04 — Experiment Registration & Controlled Factorial Protocol

**Document ID:** `LANGFENS-RQ1-TRN-04-REGISTRATION-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist, Experimental Design Lead, Reproducibility Auditor & Coordinator  
**Date:** 2026-11-04  
**Status:** **REGISTERED, FROZEN & INDEPENDENTLY EXECUTED**  
**Execution Platform:** Authorized Remote GPU Hardware (Kaggle Nvidia Tesla T4, CUDA FP16 AMP)  
**Persistent Storage:** Hugging Face Model Hub (`minkhoaa/langfens-rq1-cross-fold-checkpoints` under prefix `trn04/`)  
**Evaluation Standard:** Evaluation Contract V1 Protocol Z (Fold-1 VAL, 300 utterances, 9,956 gold canonical tokens)  
**Policy Authority:** Repository `AGENTS.md` (Strict Remote GPU Execution Boundaries; Zero Local Training Invariant)  

---

## 1. Executive Mission & Scientific Scope

This investigation prospectively specifies, executes, and evaluates a controlled **2×2 factorial training experiment** to determine whether:
1. **Class-Conditioned Focal Supervision ($\text{Factor } A$):** Penalizing omission alarms on high-frequency deletion phonemes ($\alpha=0.25$ on coronal stops and rhotics $\{D, T, R\}$, accounting for $56.69\%$ of FIT deletions) resolves the model's exaggerated deletion prior without sacrificing substitution detection.
2. **Frame-Level Waveform Acoustic Evidence ($\text{Factor } B$):** Direct injection of a 4-dimensional physical acoustic feature vector (normalized log energy, relative RMS energy, delta log energy, zero-crossing rate) into a dedicated architectural slot resolves frame-level blank collapse on unreleased closures and taps.
3. **Factorial Interaction ($A \times B$):** Combining class-conditioned focal supervision with physical acoustic evidence produces super-additive precision recovery beyond independent marginal effects.

**Primary Empirical Invariant:** No outcome is assumed. Performance claims are governed strictly by measured test results across pinned checkpoints on the frozen Fold-1 VAL partition.

---

## 2. Recovered Research State & Historical Reference Benchmarks

### 2.1 Pinned Commit & Evaluation Contract
- **Original Training Base Commit:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`
- **Evaluation Contract:** `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` (Contract SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`).
- **Partitions:**
  - **FIT:** 1,799 utterances across 12 speakers (`BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA`) — training gradient updates only.
  - **SELECT:** 300 utterances across 2 speakers (`MBMPS, TLV`) — checkpoint selection and validation AP tracking only.
  - **CAL:** 300 utterances across 2 speakers (`NCC, THV`) — threshold calibration only.
  - **VAL:** 300 utterances across 2 speakers (`NJS, RRBI`) — exploratory Protocol Z evaluation only.
  - **TEST:** 900 utterances across 6 speakers (`ABA, ASI, EBVS, HJK, LXC, PNV`) — strictly sealed and prohibited.

### 2.2 Historical Frozen T-OP Reference Metrics (Protocol Z, Fold-1 VAL)
Under genuine Protocol Z inference (9,956 gold canonical positions, 9,919 aligned tokens, 37 unaligned tokens):

| Reference Metric Channel | Historical P-SUP T-OP (Fixed Budget 3600) |
|---|---:|
| Strict Deployed Omission Precision | **0.067825 (6.78%)** |
| All-Gold Omission Recall | **0.565517 (56.55%)** |
| Dedicated Omission AP ($s_{\text{omit}}$) | **0.098560** |
| Binary MDD AP | **0.318619** |
| Legacy Substitution AP ($s_{\text{error}}$) | **0.308974** |
| Greedy CTC Phone Error Rate (PER) | **0.241980 (24.20%)** |
| False Alarms per 1,000 Canonical Phones | **113.20** |

*Note:* These represent external historical reference metrics from P-SUP V2. Because TRN-04 introduces a shared architectural feature slot (`CompactFactorialOperationHead`), newly trained C0 serves as the matched internal control.

---

## 3. Mandatory 2×2 Factorial Experimental Design

### 3.1 Experimental Arms Matrix
All four arms possess **100% bit-identical parameter counts (267,179 parameters)**, identical layer dimensions, and identical initial tensor weights from pinned seed `1337`:

| Arm | Operation Classification Objective | Additional Frame-Level Acoustic Evidence | Input Slot Behavior | Remote Kernel Slug |
|---|---|---|---|---|
| **C0** | Original 3-way Cross-Entropy (`operation_ce`) | **Disabled** | Zero-valued tensor $(B, T, 4)$; projection bias=False | `langfens-rq1-trn-04-c0-ce-nofeat` |
| **T1** | Class-Conditioned Focal Loss (`class_focal`) | **Disabled** | Zero-valued tensor $(B, T, 4)$; projection bias=False | `langfens-rq1-trn-04-t1-focal-nofeat` |
| **T2** | Original 3-way Cross-Entropy (`operation_ce`) | **Enabled** | Waveform-derived 4D features $(B, T, 4)$ | `langfens-rq1-trn-04-t2-ce-feat` |
| **T3** | Class-Conditioned Focal Loss (`class_focal`) | **Enabled** | Waveform-derived 4D features $(B, T, 4)$ | `langfens-rq1-trn-04-t3-focal-feat` |

### 3.2 Common Architectural Slot Specification
- **Backbone:** Pinned `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`), strictly frozen during training and evaluation.
- **Head Architecture:** `CompactFactorialOperationHead`:
  - `acoustic_projection`: `Linear(768, 128)` (98,432 params)
  - `acoustic_feature_projection`: `Linear(4, 128, bias=False)` (512 params) — shared slot.
  - `phone_embedding`: `Embedding(40, 128)` (5,120 params)
  - `position_embedding`: `Embedding(256, 128)` (32,768 params)
  - `attention`: `MultiheadAttention(128, heads=4)` (66,048 params)
  - `acoustic_norm`: `LayerNorm(128)` (256 params)
  - `operation_head`: `Sequential(Linear(256, 128), GELU, Dropout(0.1), Linear(128, 3))` (33,283 params)
  - `ctc_head`: `Linear(768, 40)` (30,760 params)
  - **Total Trainable Parameters:** Exactly **267,179** in all 4 models.
- **Zero-Input Invariance:** When acoustic features are disabled (C0, T1), input is $\mathbf{0} \in \mathbb{R}^{B \times T \times 4}$. Because `bias=False`, $\mathbf{W}_{\text{feat}} \mathbf{0} \equiv \mathbf{0}$. Feature-disabled models receive strictly zero extra projection signal.

---

## 4. Prospective Acceptance Gates & Rejection Thresholds

Evaluation occurs on Fold-1 VAL under Protocol Z at the nominal 15% CAL FPR ceiling ($\tau_{\text{CAL}}$ selected on NCC/THV using correct tokens only):

### Level A — Pilot Feasibility Gates (Mandatory Practical Viability)
1. Strict deployed omission precision $\ge 25.0\%$.
2. All-gold omission recall $\ge 30.0\%$.
3. False alarms $\le 30.0$ per 1,000 gold canonical positions.
4. Binary AP non-inferiority: $\ge 0.313619$ (margin $-0.005$ vs historical T-OP $0.318619$).
5. Substitution AP non-inferiority: $\ge 0.303974$ (margin $-0.005$ vs historical $0.308974$).
6. Greedy CTC PER non-inferiority: $\le 0.246980$ (margin $+0.005$ vs historical $0.241980$).

### Level B — Research Improvement Target Gates (Superiority Target)
1. Strict deployed omission precision $\ge 30.0\%$.
2. All-gold omission recall $\ge 40.0\%$.
3. False alarms $\le 15.0$ per 1,000 canonical positions.
4. Binary and Substitution AP non-inferiority preserved.
5. Upstream ASR non-inferiority preserved ($\text{PER} \le 0.246980$).
6. Acoustic Grounding Reliance: Mismatched audio gap $\text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}} \ge 0.010$.

---

## 5. Provenance & Execution Sign-Off
- **Training Code:** `audit_scripts/train_trn04_factorial.py`
- **Evaluation Code:** `src/langfens_prep/trn04/evaluator.py`
- **Initial Hashes:** `manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json`
- **Gate 0 Automated Tests:** `tests/trn04/test_gate0_factorial_contracts.py` (7/7 passed).
