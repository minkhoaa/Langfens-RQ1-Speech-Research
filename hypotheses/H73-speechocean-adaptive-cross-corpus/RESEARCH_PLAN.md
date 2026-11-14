# Research Plan: Experiment EXP-78 (SpeechOcean762 Cross-Corpus Transfer Benchmark)

## 1. Objective & Target Dataset
Evaluate zero-shot transfer performance across corpora, accents, age groups, and rating rubrics by deploying models trained strictly on adult L2-ARCTIC directly onto **SpeechOcean762** (4,536 phonemes across 300 test utterances spoken by Chinese children).

## 2. Models Benchmarked
1. **BiLSTM Context Model (EXP-08 baseline)**
2. **DMCT-Net (EXP-44 champion)**
3. **HRTD-Net (EXP-48 5ms dynamics champion)**
4. **DSCA-Net (EXP-55 dual-stream cross-attention champion)**
5. **IPTM-Net (EXP-69 interlanguage prior champion)**
6. **TPD-Net (EXP-77 temporal pyramid diffusion champion)**
7. **Septendecim-Pareto (EXP-76 17-model super-ensemble champion)**

## 3. Evaluation Protocol & Metrics
- Alignment: Charsiu Wav2Vec2 CTC forced alignment to canonical prompt phones.
- Prediction: Model predicted probability of correct pronunciation ($P(\text{CORRECT})$).
- Evaluation Metrics:
  * Pearson linear correlation $r$ with human rater scores $s \in \{0, 1, 2\}$.
  * Spearman rank correlation $\rho$ with human rater scores.
  * Binary error detection Average Precision (AP) and F1 for scores $< 1.5$.
  * Statistical significance ($p$-values).
- Verification: Zero child audio in training sets; 100% zero-shot cross-corpus domain transfer.
