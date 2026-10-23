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
- l2-arctic
- speechocean762
datasets:
- L2-ARCTIC
- speechocean762
metrics:
- f1
- precision
- recall
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

# Langfens: Wav2Vec 2.0 for Phoneme-Level Mispronunciation Detection & Diagnosis (MDD)

**Status:** PLANNED / NOT TRAINED (Architecture & Contract Smoke Testing Phase `RQ1-TRN-01A`)  
**Model Identifier:** `langfens-wav2vec2-mdd`  
**Visibility:** Private  
**Base Backbone:** `facebook/wav2vec2-base-960h` (95M parameters)  
**Research Framework:** Project Langfens (Phoneme-level Computer-Assisted Pronunciation Training)  
**Evaluation Corpora:** L2-ARCTIC v5.0 & SpeechOcean762  
**License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

---

## Model Description

`langfens-wav2vec2-mdd` is a fine-tuned Wav2Vec 2.0 acoustic model equipped with multi-task prediction heads designed for:
1. **Binary Mispronunciation Detection:** Identifying whether a learner substituted, distorted, or deleted a canonical prompt phoneme (Strict binary error policy over $N = 119,151$ canonical tokens).
2. **Phoneme Substitution Diagnosis:** Classifying the exact non-native substitution across 39 ARPAbet phoneme classes.
3. **Phonemic Deletion Detection:** Detecting elisions and omissions, with specialized tracking on high-deletion accents (e.g. Vietnamese L1 speakers).

---

## Architectural Details

- **Waveform Input:** 16 kHz Mono Audio (linear PCM / float32)
- **Feature Encoder:** 7-layer Convolutional Feature Extractor ($320\times$ temporal downsampling, $50\text{ Hz}$ frame rate)
- **Transformer Encoder:** 12 Transformer blocks, 768 hidden dimension, 8 attention heads
- **Supervision Heads:**
  - Realized Phoneme CTC Head ($40\text{ classes}$: 39 ARPAbet + 1 `<BLANK>`)
  - Canonical Token Binary Detection Head (Masked BCE with Logits)
  - Phoneme Diagnosis Head ($39\text{ classes}$, Masked Cross-Entropy)

---

## Scientific Evaluation Protocol

Evaluated on L2-ARCTIC v5.0 across 24 non-native speakers (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese) under:
- **Primary Benchmark:** 4-Fold Balanced Speaker-Independent Cross-Validation
- **Vietnamese Adaptation:** Leave-One-Vietnamese-Speaker-Out (LOSO Regime A)
- **Vietnamese Zero-Shot:** Cross-lingual transfer from 20 non-Vietnamese speakers (Regime B)
- **Phonemic Error Inventory:**
  - Substitutions ($S = 14,098$; evaluable $S_{\text{eval}} = \mathbf{13,840}$, masking $258$ unalignable `'err'` tokens)
  - Deletions ($D = 3,420$; Vietnamese cohort $D_{\text{Viet}} = \mathbf{1,566}$)
  - Additions ($I = 1,092$, 50ms tolerance window matching)
- **Evaluation Denominator:** $N_{\text{canonical}} = \mathbf{119,151}$ strictly invariant tokens ($C = 101,633$, $E_{\text{strict}} = 17,518$)

---

## Ethical & Non-Commercial Compliance

This model and derived weights are released strictly for **non-commercial educational and academic research** in accordance with the L2-ARCTIC CC BY-NC 4.0 license.
