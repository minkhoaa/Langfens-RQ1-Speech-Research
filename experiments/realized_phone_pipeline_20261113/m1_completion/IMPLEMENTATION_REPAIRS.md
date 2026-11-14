# Implementation repairs and numerical limits

These changes repaired the implementation before accepting a completed run.
Failed/interrupted directories and logs are preserved and are never pooled.

1. The ASR initialization checkpoint omits `masked_spec_embed`, which is needed
   for SpecAugment training but not its previous inference use. Transformers
   4.57.0's low-memory loading left this missing tensor nonfinite. The first
   smoke stopped immediately on nonfinite CTC loss. Initialize only that missing
   parameter with the exact uniform initialization in Wav2Vec2Model.__init__;
   assert the missing-key set and every encoder parameter is finite. The
   regression test checks finiteness, range and seed reproducibility. No batch
   was skipped, and no successful model uses an uninitialized tensor.
2. The first speed implementation inverted the sample-rate ratio and rounded it
   to a nearly coprime frequency. This both differed from SpeechBrain and made
   resampling very expensive. The first full attempt was stopped before its
   first epoch finished. The corrected frequency is `16000*speed//100`, as in
   SpeechBrain SpeedPerturb. Corrected smoke was rerun before full training.
   Functional float32 resampling and SpeechBrain's cached transform differ by
   up to 1.26e-6 on the sine fixture because of kernel computation/casting;
   shape/rate are exact, numerical tolerance is 2e-6. This runtime difference
   is disclosed, not described as bitwise upstream waveform equivalence.
3. SM13 checkpoint tensors reload exactly. Predicting a single utterance rather
   than the original larger batch changes float32 GEMM rounding (observed
   maximum ~1.2e-7). The reload check compares tensors exactly and predictions
   with atol 5e-7/rtol 1e-6. Completed scientific runs use the same batching for
   original prediction and calibration replay.
4. Fresh SM13 cache reconstruction on the same GPU/project runtime is bitwise
   equal for all 52 target/gap/layer checks across two WAVs. The CPU/reference
   Transformers runtime shows last-layer differences up to 2.62e-4, RMSE
   1.75e-5; it does not pass the strict cache guard. Both profiles and failed
   guard logs are retained. SM13 controls consume the same original float32
   cache, not mixed CPU/GPU re-extractions. H6 storage parity is checked with
   its separately registered fp16 tolerance. No model gain is attributed to
   these backend differences.

The source before each repair is preserved in `source_before_smoke_repairs/`
and `source_before_speed_repair/`. The incomplete/earlier smoke directories have
`failed` or `before_speed_repair` in their names. Device ledger durations include
the failed/interrupted CUDA attempts. The original M0/M1/capacity snapshots
remain unchanged. No repair was selected using held effectiveness metrics.
