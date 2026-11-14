# Hypothesis H01: Self-Supervised Acoustic Posterior (GOP) for Phone-Level Diagnosis

## 1. Proposition
Self-supervised speech foundation models fine-tuned on phoneme or character CTC (such as `facebook/wav2vec2-base-960h` or `facebook/wav2vec2-lv-60-espeak-cv-ft` or `charsiu`) provide dense frame-level posterior representations capable of separating correctly articulated phonemes from substitutions and omissions in non-native L2 speech.
When combined with dynamic time warping or CTC forced alignment against the canonical target prompt:
1. Target phoneme slots whose time-averaged canonical posterior is high correspond to `CORRECT`.
2. Target phoneme slots where canonical posterior is depressed while an alternative phonetic posterior is elevated correspond to `SUBSTITUTION`.
3. Target phoneme slots where canonical posterior is depressed, duration is near zero, and energy/silence posterior is elevated correspond to `OMISSION`.
4. Inter-phoneme gaps with non-blank, non-silence phoneme peaks correspond to `INSERTION`.

## 2. Experimental Interventions
- **Run EXP-01-PRIOR:** Negative control (Class Prior / Text-Only baseline). No audio input.
- **Run EXP-02-MISMATCH:** Negative control (Mismatched audio). Utterance audio paired with random prompts.
- **Run EXP-03-SSL-GOP:** Base SSL CTC alignment and frame-level GOP diagnostic scoring on `DEV_SEARCH`.

## 3. Measurable Targets
- Outperform Prior and Mismatched-Audio controls on all classes.
- Achieve measurable AP and F1 across all four core classes simultaneously.
- Identify the exact failure modes on L2-ARCTIC `DEV_SEARCH` (4 speakers: YBAA, NCC, RRBI, YDCK; 599 utterances).
