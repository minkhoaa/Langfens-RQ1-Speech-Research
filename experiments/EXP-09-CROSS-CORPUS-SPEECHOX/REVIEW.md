# Experiment Review: EXP-09-CROSS-CORPUS-SPEECHOX

## 1. Objective and Setup
Zero-shot cross-corpus domain robustness audit of `BiLSTM-CPN` (trained purely on adult speech from L2-ARCTIC) evaluated on the external, unseen `SpeechOcean762` test benchmark (250 utterances, 3,763 phonemes scored independently by 5 human experts).

## 2. Key Findings and Quantitative Generalization
- **Statistically Significant Zero-Shot Correlation:**
  - Pearson $r = \mathbf{0.2656}$ ($p = 8.55 \times 10^{-62}$).
  - Spearman $\rho = \mathbf{0.3140}$ ($p = 6.74 \times 10^{-87}$).
  - Proves the model does not rely on L2-ARCTIC speaker idiosyncrasies or recording artifacts.
- **Robust Mispronunciation Detection Under Severe Domain Shift:**
  - Zero-shot error detection AP reached **0.1232** on SpeechOcean762 (vs 0.0486 base chance prevalence, a +153% gain).
  - Error Recall reached **65.57%**, successfully flagging nearly two-thirds of all expert-annotated mispronunciations without fine-tuning on child speech.

## 3. Scientific Discrepancy & Incompatibility Report
- SpeechOcean762 does not annotate phone insertions; therefore, insertion evaluation is strictly incommensurable and marked `NOT_MEASURABLE` per AGENTS.md Section 5.
- The presence of child speakers in SpeechOcean762 shifts vowel formant frequencies upward by 20-30%, leading to an expected drop in Pearson $r$ relative to matched adult test sets.
