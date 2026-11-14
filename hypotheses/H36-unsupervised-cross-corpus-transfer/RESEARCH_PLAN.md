# Research Plan: Hypothesis H36 / Experiment EXP-41

## 1. Objective
Benchmark zero-shot cross-corpus transfer of champion Octad-Pareto (EXP-39), HTC-Net (EXP-38), and baseline SSL-GOP (EXP-03) on the independent SpeechOcean762 test set (3,763 phones across 500 utterances).

## 2. Experimental Steps
1. **SpeechOcean762 Ingestion:**
   - Ingest raw audio and human phone-level scores [0-10] from `data/original_sources/speechocean762/` or `data/derived/speechocean762/`.
   - Binarize human accuracy: Score $\le 6.0 \to$ Pronunciation Error (`has_error=1`), Score $> 6.0 \to$ Correct (`has_error=0`).
2. **Feature Extraction:**
   - Run deployment pipeline (Wav2Vec2 frame classifier + Viterbi alignment + physical waveform prosodic features).
3. **Inference & Scoring:**
   - Evaluate Octad-Pareto, HTC-Net, and SSL-GOP zero-shot.
   - Compute Pearson correlation $r$, Spearman correlation $\rho$, Error AP, Error Precision, Error Recall, and Error F1.
4. **Ledger & Artifact Audit:**
   - Save comparison table, PR curves, and SHA-256 manifests under `experiments/EXP-41-ZERO-SHOT-L2-ARCTIC-TRANSFER-BENCHMARK/`.
