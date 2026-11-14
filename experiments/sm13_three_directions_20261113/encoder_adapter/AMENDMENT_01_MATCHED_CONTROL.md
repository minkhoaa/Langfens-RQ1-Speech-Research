# Preregistered technical amendment: matched prefix precision

Before continuing the remaining comparisons, preflight found that the adapter consumes the shared layer-3 prefix cache in float16 while the original frozen middle-mixture control consumed float32 pooled caches. The measured difference was small (`max abs 9.67e-4` at the cached prefix; suffix max abs `6.46e-4`), but it is an avoidable input precision mismatch.

The primary paired comparison is amended as follows:

- Keep every candidate LoRA checkpoint, prediction, and fixed training schedule unchanged.
- Recompute frozen control layers 5–9 from the exact same cached float16 prefix through frozen Wav2Vec2 blocks 4–8, then pool with the same target/gap feature code.
- Warm-start and scaler remain the original fit-only middle-layer initialization shared by both arms. Retrain each frozen control for the same two paired epochs and per-utterance optimizer schedule, with all encoder blocks frozen.
- Reuse the already computed candidate predictions and the same canonical all-13-layer shared gap probabilities. Do not use held labels for any selection.
- Preserve the original float32-cache control files and pair summaries under `folds/superseded_quantization_mismatch/`; label them superseded and exclude them from the primary matched comparison.

This amendment corrects numerical parity only. The remaining candidate model, hyperparameters, epochs, and evaluation policy remain fixed.
