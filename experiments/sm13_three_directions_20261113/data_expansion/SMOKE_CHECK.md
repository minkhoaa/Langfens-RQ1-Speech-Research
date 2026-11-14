# Extraction parity check

The corrected extractor was checked on annotated TRAIN utterance `ABA_arctic_a0003` only to validate waveform and feature implementation. The archive waveform is 44.1 kHz. The repository's `torchaudio.transforms.Resample(44100, 16000)` and PCM16 WAV save/decode path produced an exact 78,720-sample match to the existing local 16 kHz WAV (maximum absolute difference 0, correlation 1.0).

With checkpoint-configured `AutoFeatureExtractor` normalization (`do_normalize=True`), pooled features from the corrected waveform reproduced all 13 historical Wav2Vec2 layer features with cosine similarity 1.0. Layer 6 RMSE was `4.54e-7` and maximum absolute error was `4.89e-6`; layer 12 RMSE was `1.65e-5`. The detailed per-layer comparison is in `INPUT_PARITY_AUDIT.json`. CharSiU continues to receive the resampled PCM16 samples directly, without applying Wav2Vec2 normalization.

The full corrected extra-audio feature extraction and training runs are recorded in `logs/` and `RUN_LOG.md`. The initial unnormalized extraction smoke was retained under `REJECTED_PREPROCESSING_MISMATCH/` and is not considered evidence for the candidate results.
