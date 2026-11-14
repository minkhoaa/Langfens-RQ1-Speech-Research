# Research Plan: Hypothesis H05 (Cross-Corpus SpeechOcean762)

1. **Ingestion & Mapping:**
   - Extract test audio and scores from `speechocean762.tar.gz`.
   - Parse word and phoneme sequences, strip vowel stress markers (e.g. `EH0` -> `EH`).
   - Extract 250 held-out test utterances from `speechocean762/test/wav.scp` (~5,000 phone tokens).
2. **Inference with Pinned Model:**
   - Pass 16kHz audio through frozen `charsiu/en_w2v2_fc_10ms` backbone.
   - Align target phones via Viterbi alignment.
   - Extract acoustic-phonetic feature sequences normalized by L2-ARCTIC train statistics.
   - Run forward pass of trained `BiLSTMContextModel` from `EXP-08-PHONOTACTIC-CONTEXT`.
3. **Statistical Metrics:**
   - Compute Pearson Correlation ($r$) and Spearman Rank Correlation ($\rho$) vs expert scores.
   - Compute Average Precision (AP), Precision, Recall, and F1 for Mispronunciation Detection (score $< 1.5$).
4. **Artifacts & Review:**
   - Save complete prediction tables, metrics, logs, and hashes in `experiments/EXP-09-CROSS-CORPUS-SPEECHOX/`.
