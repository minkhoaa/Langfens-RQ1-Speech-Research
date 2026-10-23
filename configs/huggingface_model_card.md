---
language:
- en
license: cc-by-nc-4.0
tags:
- speech
- wav2vec2
- mispronunciation-detection
- mdd
- capt
- phoneme-recognition
- cross-attention
- l2-arctic
- speechocean762
datasets:
- L2-ARCTIC
- speechocean762
metrics:
- pr_auc
- f1
- precision
- recall
- accuracy
model-index:
- name: langfens-wav2vec2-mdd
  results:
  - task:
      type: audio-classification
      name: Mispronunciation Detection & Diagnosis
    dataset:
      type: L2-ARCTIC
      name: L2-ARCTIC v5.0 (4-Fold Cross Validation)
    metrics:
    - type: pr_auc
      name: Strict Binary PR-AUC
    - type: f1
      name: Strict Binary F1-Score
    - type: accuracy
      name: Substitution Diagnosis Accuracy
    - type: recall
      name: Deletion Recall
---

# Langfens: Pretrained Wav2Vec 2.0 Multi-Task Phoneme Mispronunciation Detection & Diagnosis (MDD)

**Status:** ARCHITECTURE & INFERENCE VALIDATED / PRE-TRAINING GATE CLEARED (`RQ1-TRN-01A-R`)  
**Real-Data Fine-Tuning Status:** UNAUTHORIZED IN THIS GATE (Deferred to Authorized Execution Phase `RQ1-TRN-01B`)  
**Model Identifier:** `langfens-wav2vec2-mdd`  
**Visibility:** Private  
**Pretrained Backbone:** `facebook/wav2vec2-base-960h` (94.37M parameters, Apache-2.0)  
**Custom MDD Heads:** Multihead Cross-Attention, Realized CTC (40 classes), Binary MDD (Masked BCE), Substitution Diagnosis (39 ARPAbet classes, Masked CE) (~3.03M parameters)  
**Total Architecture Parameters:** 97,405,648 (~97.41M parameters)  
**Research Framework:** Project Langfens (Phoneme-level Computer-Assisted Pronunciation Training)  
**Target Evaluation Corpora:** L2-ARCTIC v5.0 (TAMU PSI Lab) & SpeechOcean762  
**Downstream Dataset License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  
**Source Code License:** MIT License  

---

## Model Description

`langfens-wav2vec2-mdd` integrates a genuine Hugging Face `facebook/wav2vec2-base-960h` acoustic backbone with multi-task CAPT prediction heads.

### Core Architectural Guarantees:
1. **Alignment-Free Deployable Inference:**
   - Unlike legacy systems that require manual gold TextGrid timestamps, the deployable architecture maps continuous acoustic frame representations to canonical prompt phonemes via **differentiable Multihead Cross-Attention**.
   - Test-time inference accepts strictly the raw speech waveform ($x$) and canonical prompt tokens ($p_{1..N}$), requiring zero external alignment files.
   - An optional oracle TextGrid pooling baseline is retained strictly for ablation and scientific comparison (`alignment_mode="oracle"`).
2. **Binary Mispronunciation Detection:**
   - Detects whether a canonical phoneme was produced correctly ($0$) or mispronounced ($1$, substitution or deletion).
   - Primary evaluation uses **Strict Policy**; sensitivity analysis reports **Lenient Policy** (reclassifying $1,112$ base-preserving allophonic variations `CPL,CPL*,s`).
3. **Phoneme Substitution Diagnosis:**
   - Evaluated strictly over the $S_{\text{eval}} = \mathbf{13,840}$ clean non-native substitutions (masking the 258 unalignable `'err'` tokens from the 39-class Cross-Entropy loss).
4. **Realized-Phone CTC Recognition Head:**
   - $40\text{ classes}$ ($0 = \text{<BLANK>}$, $1..39 = \text{standard ARPAbet}$).
   - Supervised using chronological realized phone sequences reconstructing valid insertions (epenthetic additions) and clean substitutions, while strictly excluding acoustic deletions and non-speech silence pauses.

---

## Partition Protocols & Evaluation Denominators

All experiments strictly consume the frozen split manifests in `manifests/RQ1-SPLIT-01/`:

1. **Primary Benchmark (4-Fold Speaker-Independent Cross-Validation):**
   - 24 speakers (balanced 3M/3F per test fold, 1 speaker per L1).
   - Total corpus canonical denominator: $N_{\text{canonical}} = \mathbf{119,151}$ tokens ($C = 101,633$, $S = 14,098$, $D = 3,420$, $E = 17,518$).
   - **Fold-Specific Test Denominators (Must Not Be Conflated with Aggregate):**
     - Fold 1 Test: $N = \mathbf{29,769}$ ($C = 25,249$, $S = 3,744$, $D = 776$, $E = 4,520$).
     - Fold 2 Test: $N = \mathbf{29,798}$ ($C = 25,160$, $S = 3,649$, $D = 989$, $E = 4,638$).
     - Fold 3 Test: $N = \mathbf{29,798}$ ($C = 25,578$, $S = 3,467$, $D = 753$, $E = 4,220$).
     - Fold 4 Test: $N = \mathbf{29,786}$ ($C = 25,646$, $S = 3,238$, $D = 902$, $E = 4,140$).
2. **Vietnamese In-Domain Adaptation (LOSO Regime A):**
   - 4 rounds across Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`).
   - Per-round partition: **21 train / 2 validation / 1 held-out test speaker**.
3. **Vietnamese Zero-Shot Generalization (Regime B):**
   - Partition: **18 non-Vietnamese train / 2 validation / 4 held-out Vietnamese test speakers**.
4. **Confidence Intervals & Statistical Rigor:**
   - 95% Confidence Intervals computed via **Speaker-Clustered Bootstrap** ($B = 1,000$ resamples clustered at the speaker level) to account for within-speaker phonetic correlation.
   - Operating threshold $\theta^*$ calibrated strictly on validation splits.

---

## Hardware & Deployment Feasibility

- **Reference Verification GPU:** NVIDIA GeForce GTX 1650 Mobile (3.73 GB usable VRAM, 4.0 GB physical).
- **Execution Limits:** Supports micro-batches $B \in [1, 16]$ with AMP fp16, gradient checkpointing, and frozen feature encoder (peak reserved VRAM $< 2.0\text{ GB}$).
- **OOM Boundary:** Local GPU triggers out-of-memory at $B \ge 48$ on 4.0s speech; cloud GPUs ($\ge 16\text{ GB}$) recommended for full multi-fold training throughput.
