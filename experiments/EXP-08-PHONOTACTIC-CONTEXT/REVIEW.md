# Experiment Review: EXP-08-PHONOTACTIC-CONTEXT

## 1. Objective and Setup
Investigated the impact of utterance-level sequential context using a Bidirectional LSTM network (`BiLSTMContextModel`) on `DEV_SEARCH` (4 speakers: YBAA, NCC, RRBI, YDCK; 599 utterances).
The architecture explicitly conditioned inter-phoneme gap representations on the left and right phonetic hidden states $[\mathbf{h}_{\text{left}} ; \mathbf{h}_{\text{right}}]$.

## 2. Quantitative Breakthroughs
- **Breakthrough on INSERTION Detection:**
  Conditioning gap features on surrounding phonetic context caused a dramatic leap in insertion metrics:
  - `INSERTION` AP increased from 0.0332 to **0.2016** (+507% relative gain over EXP-07, +3,183% over prior chance baseline 0.0061).
  - `INSERTION` F1 increased from 0.0290 to **0.2605** (+798% relative gain).
  - Precision reached **0.2500**, Recall reached **27.20%**.
- **All-Time Campaign High on OMISSION:**
  - `OMISSION` AP increased to **0.1725** (highest all-time).
  - `OMISSION` F1 reached **0.2323**, Recall reached **30.62%**.
- **Campaign Record Weighted Utility:**
  - Weighted utility rose to **0.3610**, surpassing all previous methods (EXP-06 at 0.3404, EXP-01 at 0.3183, EXP-07 at 0.3057).
- **Macro Performance:**
  - Macro F1 reached **0.5126** (highest all-time).
  - Macro AP reached **0.4991** (highest all-time).
- **Model Calibration:**
  - Overall ECE dropped to **0.0720** (Gap ECE is **0.0091**).

## 3. Failure Mode Diagnostics & Next Research Direction
While `CORRECT` (0.977 AP, 0.938 F1) is fully confirmed above the 0.80 gate:
- `SUBSTITUTION` F1 is currently 0.3674 (Precision = 0.4462). The primary false substitutions stem from non-native vowel length and formant shifts that do not change phoneme identity in the speaker's L1.
- `OMISSION` F1 is currently 0.2323. The false omissions occur primarily on unstressed vowels (/AH/, /IH/) and word-final stops (/t/, /d/) where acoustic energy is weak but not fully absent.
- **Direct Next Action (EXP-09-CROSS-CORPUS-SPEECHOX):**
  Evaluate cross-dataset generalization on the independently audited `SpeechOcean762` corpus to assess domain robustness and phoneme accuracy correlation.
