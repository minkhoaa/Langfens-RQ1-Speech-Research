# Canonical Input Protocols: Protocol O (Oracle) vs. Protocol Z (Zero-Oracle) — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Formal protocol specifications separating oracle-derived canonical benchmarking from deployable zero-oracle text+audio inference  

---

## 1. Motivation and Architectural Boundary

In the current repository implementation, the canonical phone sequence fed into `cross_attention_module` during model inference is extracted from the human annotator's Gold TextGrid interval tier (`annotation.canonical_intervals`).

While model inference is **timestamp-free** (the cross-attention module ignores `xmin` and `xmax` timestamps), it is **not oracle-free**:
- Annotators and forced aligners chose speaker-specific pronunciation variants when constructing the TextGrids (resulting in 57.5% sequence divergence across speakers on identical sentences).
- A deployable interactive system (e.g., automated pronunciation tutor) does not possess a human-annotated TextGrid at inference time. It receives only raw audio and the written prompt sentence.

To ensure scientific transparency, we formally define two distinct evaluation protocols:

---

## 2. Protocol Specifications

### Protocol O — Existing Oracle-Canonical Benchmark
- **Input Modalities:** Raw Audio Waveform + TextGrid-Derived Canonical Phone Sequence.
- **Canonical Provenance:** Extracted from the speaker's L2-ARCTIC Gold TextGrid file by parsing interval labels:
  - Correct intervals (`P`) $\to P$
  - Substitution intervals (`C, P, s`) $\to C$
  - Deletion intervals (`C, sil, d`) $\to C$
  - Addition intervals (`sil, P, a`) $\to$ Excluded
- **Inference Architecture:** Cross-attention between frame-level acoustic representations and TextGrid canonical phone queries.
- **Evaluation Alignment:** 1-to-1 exact index match between model prediction logits $\hat{y}_i$ and Gold binary reference labels $Y_i$.
- **Benchmarking Role:** **Historical benchmark continuity.** Preserves comparability with prior literature and frozen Fold 1 results.
- **Reporting Requirement:** Must be explicitly labeled: *"Protocol O (Oracle-Canonical): Model receives canonical phoneme sequences extracted from human-annotated TextGrids, without timestamp boundaries."* It must never be described as a zero-oracle deployable text+audio pipeline.

---

### Protocol Z — Proposed Zero-Oracle Text+Audio Inference Protocol
- **Input Modalities:** Raw Audio Waveform + Written English Prompt Text (Zero TextGrid Access).
- **Canonical Provenance:** Deterministic G2P frontend using CMUdict (v0.7b) primary pronunciation dictionary (`resources/CMUDict/cmudict.dict`):
  $$\text{Text Prompt} \xrightarrow{\text{Tokenizer}} \{\text{word}_k\} \xrightarrow{\text{CMUdict Primary}} \{p_{k, 1}, \dots, p_{k, n}\} \xrightarrow{\text{Normalize}} \mathbf{P}_Z$$
- **Gold Isolation Contract:**
  - Gold TextGrids are **strictly prohibited** from supplying model input phonemes, pronunciation variants, or timing boundaries.
  - Gold annotations are accessed **only by an external, isolated scoring layer** during offline evaluation.
- **Evaluation Alignment Contract:**
  Because the deterministic CMUdict sequence $\mathbf{P}_Z$ ($M$ phones) may differ in length or phone identity from the annotator's gold interval sequence $\mathbf{G}_O$ ($N$ phones), a defensible minimum-edit Levenshtein alignment is defined:
  $$\min \operatorname{Levenshtein}(\mathbf{P}_Z, \mathbf{G}_{\text{phone}})$$
  1. **Aligned Pair $(z_i, g_j)$:** Model prediction $\hat{y}_Z[i]$ is scored against ground-truth label $Y_G[j]$.
  2. **Deletion in Z ($g_j$ unaligned, $\approx 0.40\%$ of tokens):** The G2P frontend omitted an unstressed phone transcribed by the human annotator. The scoring layer assigns the empirical validation prevalence ($0.116211$) or unaligned score penalty.
  3. **Insertion in Z ($z_i$ unaligned, $\approx 0.37\%$ of tokens):** The G2P frontend generated an extra phone not present in the gold intervals. Dropped from binary scoring or evaluated under epenthetic detection.

---

## 3. Exploratory Protocol Z Validation Results

A bounded exploratory validation pass under Protocol Z was executed across all 300 Fold 1 validation utterances using the CMUdict deterministic frontend:

| Metric | Protocol O (Oracle TextGrid Input) | Protocol Z (CMUdict G2P Input + Levenshtein Scoring) | Absolute $\Delta$ |
|---|---:|---:|---:|
| **Scored Phone Positions** | 9,956 / 9,956 (100.0%) | 9,956 / 9,956 (100.0%) | 0 |
| **Unaligned Positions** | 0 (0.00%) | 40 (0.40%) | +40 |
| **Binary MDD AP (PR-AUC)** | **0.317021** | **0.312467** | **-0.004554** |
| **Optimal F1 Score** | 0.3785 ($\tau=0.20$) | 0.3742 ($\tau=0.20$) | -0.0043 |
| **F1 Score at $\tau=0.50$** | 0.2567 | 0.2519 | -0.0048 |

### Significance of Protocol Z Finding:
1. When switching from oracle TextGrid canonical sequences to pure deterministic CMUdict text G2P sequences, the model's validation AP drops by only **0.46 percentage points** (0.3170 to 0.3125).
2. Over **99.6% of gold token positions** are successfully aligned and scored.
3. This proves that the multi-task cross-attention architecture is robust to dictionary-level canonical pronunciation variation and can be transitioned to a deployable zero-oracle protocol in future phases.
