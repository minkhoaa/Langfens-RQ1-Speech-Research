# Experiment Review: EXP-03-SSL-GOP

## 1. Objective and Setup
First deployment-feasible acoustic phoneme assessment method using a Wav2Vec2-backed 10ms frame classifier (`charsiu/en_w2v2_fc_10ms`) on L2-ARCTIC `DEV_SEARCH` (4 speakers: YBAA, NCC, RRBI, YDCK; 599 utterances).
Inference executed strictly on raw 16kHz audio and canonical prompt text without oracle TextGrids or timestamps.

## 2. Key Findings and Quantitative Gains
- **Acoustic Signal Demonstrated Across All Classes:**
  - `CORRECT` AP reached 0.9307 and F1 reached 0.9185 (exceeding the 0.80 gate).
  - `SUBSTITUTION` AP reached 0.1828 (+104% relative gain over prior chance baseline 0.0895). F1 = 0.2444.
  - `OMISSION` AP reached 0.0427 (+303% relative gain over prior chance baseline 0.0106).
  - `INSERTION` AP reached 0.0144 (+136% relative gain over prior chance baseline 0.0061).
- **Acoustic Calibration:** ECE is low (0.0523), well inside the $\le 0.15$ guardrail.

## 3. Scientific Failure Diagnostics (Root Cause Analysis)
1. **The Alignment-Forcing Deletion Blindspot:**
   Standard forced alignment forcibly aligns frames to omitted phones, preventing duration from dropping to 0ms. Out of 209 true omissions in `DEV_SEARCH`, only 2 were recalled (Recall = 0.0096).
   *Required Fix:* Implement explicit skip transitions in Viterbi alignment or sequence-level CTC loss differencing ($\Delta \mathcal{L}_{\text{omission}}$ per Hypothesis H1).
2. **False Warning Penalty Triggered:**
   Because 1,429 false substitutions and 300 false insertions occurred (3.12 false alarms/utterance), the coverage risk penalty zeroed the coverage/risk utility component.
   *Required Fix:* Operating policy calibration on `DEV_CAL`. Raw uncalibrated probabilities place 0.50 threshold on noisy frames. We need margin logit calibration and temperature scaling.
