# Hypothesis H36: Zero-Shot Cross-Corpus Transfer on SpeechOcean762 Benchmark

## 1. Context and Problem Statement
In computer-assisted pronunciation training, single-corpus models often overfit to corpus-specific recording acoustics, sentence structures, or annotation quirks.
Under `AGENTS.md` Section 5 and 14, demonstrating cross-corpus external validity is mandatory.
In EXP-09, an early baseline on SpeechOcean762 established $r = 0.2656$ and $\text{Error AP} = 0.1232$.
Since EXP-09, the laboratory developed:
1. Physical waveform prosodic energy dynamics (APED-Net, EXP-28).
2. Lexical phonotactic n-gram prior conditioning (PNC-Net, EXP-34).
3. Hierarchical broad-category vowel/consonant cascaded routing (HTC-Net, EXP-38).
4. Octad-Pareto super-ensemble (EXP-39).

Hypothesis H36 tests whether these advanced representations transfer zero-shot to an entirely independent publisher corpus—SpeechOcean762 (comprising Chinese children and adult L2 learners, non-overlapping acoustic environment, and human continuous accuracy scores [0-10]).

## 2. Formal Hypothesis
Evaluating Octad-Pareto and HTC-Net zero-shot on the complete SpeechOcean762 test set (3,763 phones across 500 utterances) without any corpus-specific fine-tuning will demonstrate:
1. **Correlation Elevation:** Pearson correlation between predicted phone correctness probabilities and human scores will increase from $r = 0.2656$ to $r \ge 0.3500$ ($p < 10^{-100}$).
2. **Error Detection Elevation:** Error AP will increase from $0.1232$ to $\ge 0.1800$.
3. **High Error Recall:** Under frozen operating policies, zero-shot Error Recall will exceed $70.0\%$.

## 3. Predicted Evidence & Significance
If validated, this establishes definitive proof of domain invariance, demonstrating that physical acoustic energy dynamics and phonological tree structures capture universal phonetic invariants across divergent recording conditions and age demographics.
