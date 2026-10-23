# KoelLabs/L2Arctic Dataset Card & Provenance Documentation

## 1. Overview
- **Dataset Repository:** `KoelLabs/L2Arctic`
- **URL:** https://huggingface.co/datasets/KoelLabs/L2Arctic
- **Revision / SHA:** `e9f2aaca153989a45e86bc2d981ba03c4879476d`
- **Created Date:** 2025-05-27T10:33:37.000Z
- **Last Modified:** 2026-02-19T19:27:22.000Z
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (`cc-by-nc-4.0`)
- **Access Status:** `GATED (auto)` — Access requires requesting access via Hugging Face website and accepting terms; gated read token required for direct Parquet download. Full dataset download is NOT required for RQ1-ACQ-01.

## 2. Description & Lineage
`KoelLabs/L2Arctic` is a processed, Hugging Face `datasets`-compatible derivative of the original L2-ARCTIC speech corpus (Zhao et al., 2018).
Original source: Texas A&M University (PSI Lab) and Iowa State University (https://psi.engr.tamu.edu/l2-arctic-corpus/).

## 3. Preprocessing & Normalization Changes
As documented by KoelLabs (data processing script: `https://github.com/KoelLabs/ML/blob/main/scripts/data_loaders/L2ARCTIC.py`):
1. **Audio Normalization:**
   - Audio is converted to `float32` in the `[-1.0, 1.0]` range.
   - Resampled to 16 kHz sampling rate.
2. **Phonetic Representation (IPA Conversion):**
   - Maps raw ARPAbet phoneme annotations to International Phonetic Alphabet (IPA).
   - Compatible with `ipapy` and `panphon` libraries.
3. **Annotation Cleaning & Filtering:**
   - Corrected typos and malformatted annotation files present in raw distributions.
   - Filtered out non-expert verified samples, retaining only validated utterances.

## 4. Splits & Dataset Statistics
- **Features Schema:**
  - `audio`: Audio feature (decoded waveform at 16 kHz, float32)
  - `ipa`: string (IPA phoneme annotation string)
  - `text`: string (Prompt orthographic text)
  - `g2p`: string (Grapheme-to-phoneme output)
  - `speaker_code`: string (e.g., `ABA`, `BWC`, etc.)
  - `speaker_gender`: string (`M` or `F`)
  - `speaker_native_language`: string (`Vietnamese`, `Korean`, `Mandarin`, `Spanish`, `Hindi`, `Arabic`)
- **Splits:**
  - `scripted`: 3,599 utterances (~220 minutes), corresponding to the 3,599 manually verified utterances from CMU ARCTIC prompts across 24 speakers. (Size: ~423 MB).
  - `spontaneous`: 22 samples (~26 minutes), corresponding to spontaneous speech recordings from the 'suitcase' task across 22 speakers. (Size: ~50.2 MB).
- **Storage & Download Size:**
  - Download size: ~472.5 MB (Parquet format)
  - Total dataset size: ~473.2 MB

## 5. Usage & Access Requirements
- Authenticated Hugging Face CLI login (`huggingface-cli login`) with gated read permission.
- Does not replace raw L2-ARCTIC for phonetic alignment provenance or gold Praat TextGrids, but serves as a standardized reference for preprocessed IPA baselines if authorized in future phases.
