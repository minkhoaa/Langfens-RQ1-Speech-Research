# Hypothesis H05: Cross-Corpus Generalization and Domain Shift (SpeechOcean762)

## 1. Proposition
A scientifically valid pronunciation assessment model must demonstrate genuine generalization beyond its source dataset.
The `BiLSTM-CPN` model trained exclusively on L2-ARCTIC will transfer zero-shot to SpeechOcean762 (Mandarin L1 speakers, mixture of adult and child speech, recorded in varied acoustic environments), achieving:
1. Statistically significant positive Pearson correlation ($r \ge 0.50$, $p < 10^{-5}$) between predicted $P(\text{CORRECT})$ and human expert phoneme accuracy scores.
2. Binary Mispronunciation Detection AP $\ge 0.45$ on unseen Mandarin adult and child speakers without any corpus-specific retraining.

## 2. Incompatibility and Ontology Mapping
- **Phonetic Inventory:** Both corpora employ the ARPAbet 39-phoneme convention. Lexical stress markers (0, 1, 2) in SpeechOcean762 are stripped to match ARPAbet.
- **Label Mapping:**
  - SpeechOcean762 phone accuracy is scored 0.0 to 2.0 by 5 human experts.
  - Correct speech: Score $\ge 1.5$.
  - Mispronunciation: Score $< 1.5$ (encompassing substitutions and omissions).
  - Insertions are NOT annotated in SpeechOcean762; therefore, cross-dataset evaluation reports target-phone assessment and marks insertion as `NOT_MEASURABLE_ON_SPEECHOCEAN762` per AGENTS.md Section 5.
