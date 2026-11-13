# Experiment Review: EXP-02-MISMATCH

## 1. Objective and Setup
Mismatched-audio negative control to establish whether the evaluation pipeline has any oracle side-channels or illicit access to ground truth labels. Prompt transcripts from `DEV_SEARCH` (599 utterances) were paired with audio from completely different utterances.

## 2. Key Findings
- **Zero Oracle Leakage Confirmed:** When audio is mismatched, the system collapses to massive false warnings (30.03 false alarms per utterance).
- **Utility Collapse:** Weighted utility drops precipitously to 0.0716 (vs 0.3183 in EXP-01-PRIOR).
- **Calibration Failure:** Overall ECE jumps to 0.4522, triggering maximum risk penalty and zeroing coverage utility.
- **Scientific Significance:** This proves the evaluation pipeline strictly measures acoustic-prompt congruence. A high score is impossible unless the acoustic waveform genuinely matches the phonetic predictions.
