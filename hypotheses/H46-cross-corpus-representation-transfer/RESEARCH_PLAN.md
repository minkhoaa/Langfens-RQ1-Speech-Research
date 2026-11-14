# Research Plan: Experiment EXP-51 (Cross-Corpus Transfer Benchmark on SpeechOcean762)

## 1. Objective & Benchmark Configuration
Perform zero-shot cross-corpus benchmarking on **SpeechOcean762** test set (300 utterances, 4,536 phonemes, expert human pronunciation scores $\in [0, 10]$):
- **Models Evaluated:**
  1. **HRTD-Net (EXP-48):** Champion 5ms sub-frame micro-dynamics model.
  2. **DMCT-Net (EXP-44):** Dynamic sequence token masking model.
  3. **Decimal-Pareto (EXP-45):** 10-Model consensus mixture.
  4. **BiLSTM Context Model (EXP-08):** Baseline reference comparator.
- **Evaluation Criteria:**
  - Pearson correlation $r$ with human phonetic scores.
  - Spearman rank correlation $\rho$.
  - Binary mispronunciation classification: human score $< 1.5$ defines ground-truth mispronunciation.
  - Precision, Recall, F1, and Average Precision (AP) for error detection.

## 2. Experimental Steps
1. Parse `speechocean762.tar.gz` test partition in memory (audio waveforms and phonetic labels).
2. Align acoustic features using Wav2Vec2 CTC alignment (`charsiu/en_w2v2_fc_10ms`).
3. Extract 5ms sub-frame temporal dynamics (burst onset ratios, $dE/dt$, crest factor) directly from the raw 16kHz audio waveforms.
4. Pass preprocessed sequences through HRTD-Net, DMCT-Net, BiLSTM, and Decimal-Pareto.
5. Compute Pearson $r$, Spearman $\rho$, and binary mispronunciation metrics.
6. Compare results against EXP-41 baselines and audit against H46 falsification criteria.
