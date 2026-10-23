# MRPL2 Dataset Research & Provenance Documentation

## 1. Overview
- **Dataset Title:** The MRPL2 Non-native Speech Corpus
- **Zenodo Record ID:** `22690685` (Concept ID: `20365864`)
- **DOI:** https://doi.org/10.5281/zenodo.22690685
- **URL:** https://zenodo.org/records/22690685
- **Version:** `1.1`
- **Publication Date:** 2026-09-10 (Last modified: 2026-09-18)
- **Creators:** Daniel Tweddle, Eugenio Donati (University of West London)
- **License:** Creative Commons Attribution 4.0 International (`cc-by-4.0`)
- **Access Status:** `ACCESS_RESTRICTED` (Zenodo Restricted Access — Requires formal access application; no files downloaded per acquisition safety rules; non-blocking).

## 2. Dataset Scope & Composition
- **Target Task:** Mispronunciation Detection & Diagnosis (MDD), Automatic Speech Recognition (ASR), Phonetic Analysis.
- **Reference Dialect:** British English — Modern Received Pronunciation (MRP).
- **Scale:** 4,500 utterances across 23 non-native speakers.
- **L1 Coverage:** 14 native language backgrounds.
- **Audio Specifications:** Provided in dual sampling rates: 48 kHz (high fidelity) and 16 kHz (standard ASR/MDD rate).
- **Annotations:**
  - Manually annotated Praat TextGrids per utterance.
  - Word-level orthographic transcripts.
  - Two master corpus files: corpus-wide expected (`exp`) and ground-truth (`gt`) phoneme sequences per utterance.

## 3. Suitability for RQ1 & Evaluation Viability
- **External Evaluation:** Excellent candidate for cross-dialect, cross-L1 out-of-domain evaluation of MDD models trained on American English corpora (L2-ARCTIC, SpeechOcean762).
- **Dialect Difference:** Tests model generalization across American English (General American / CMUdict ARPAbet) vs. British English (MRP).
- **Phase RQ1 Status:** Optional candidate only; access is restricted and not required for `REFERENCE_PACK_COMPLETE`.
