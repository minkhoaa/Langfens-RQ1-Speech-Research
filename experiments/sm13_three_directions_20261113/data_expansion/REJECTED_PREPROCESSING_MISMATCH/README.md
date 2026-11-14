# Rejected preprocessing lineage

The first exploratory caches and candidate folds 0 / seeds 101 and 202 were generated with direct, unnormalized Wav2Vec2 inputs and a SciPy resampling path. The historical pooled feature contract uses `tools/parse_l2arctic.py` torchaudio resampling with PCM16 WAV quantization, followed by the checkpoint's `AutoFeatureExtractor` normalization (`do_normalize=True`).

An annotated TRAIN probe (`ABA_arctic_a0003`) showed that the corrected path matches the stored waveform exactly and reproduces the historical 13 pooled layers to numerical precision. The old candidate predictions, checkpoints, receipts, and extra-feature caches are retained here for audit only and are not benchmark results. The shared labelled-control artifacts remain in `../control_results` and are unaffected.
