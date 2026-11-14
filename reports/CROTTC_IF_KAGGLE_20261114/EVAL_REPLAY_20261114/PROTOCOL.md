# Registered paired evaluator replay — CROTTC-EVAL-REPLAY-20261114-v1

User agrees to testing the evaluation pipeline after the confirmed RoPEMHA eval-dropout defect.
**Scope: one bounded private Kaggle CPU-only evaluation, no optimizer/backward,
no main runtime modification, no new production training or test/DEV_CAL access.**

## Frozen design before outcomes

- Fixed independently downloaded/verified AM checkpoint **3150/internalepoch55/cursor480**,
  SHA `47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
  Not selected after looking at this replay's scores; not official released weights.
- Exactly599 authorized frozen DEV_SEARCH utterances, speakers NCC/RRBI/YBAA/YDCK;
  manifest and each audio SHA verified before inference. Reuse private authorized assetsv1/1
  and exact parent kernelwindow001/1; no new WAV/dataset/checkpoint upload.
- Full native315453120WavLM+3822765heads, FP32, eval/frozen, CPU2threads.
  Cache one native unpadded audio-derived WavLM memory per utterance in **RAM only**.
  Both arms consume identical memories; decoding sees only IDs/features, no labels.
- Interleaved fixed trials: legacy3047, repaired3047, legacy3048, repaired3048,
  legacy3049. All outcomes reported, not highest repeat. Repairpredictions must be
  exactly identical across two seeds. Each trial saves/restores Python/NumPy/Torch RNG.
- **Only intervention:** in disposable evaluator memory, set two RoPEMHA dropout attributes
  from.1 to0 during eval control; restore on exit. No change to train-mode implementation,
  weights, package files, main controller, LR, labels, metric policy or historical scores.
- Pin actual remoteattention.pySHA to public/localv1.0.3 source. Fail if it differs.
- Postdecode scoring: pinned MpdStats hierarchical detection counts/F1/P/R/CD/ED,
  old metric separately; production-compatible PER retaining hypothesis sil plus
  explicitly labeled no-silPER. Per-speaker reports and per-utterance alignments saved.
  Undefined source metric reported explicitly, never replaced with zero.
- Main result: repairedF1 minus meanlegacy3 F1, plus legacyrange and all repetitions.
  This is diagnostic repeat variability, **not independent model-seed uncertainty**.
- Backend is CPU/unpadded/shared-memory, whereas productionvalidation is2T4/padded.
  Thus direct comparison to historicalGPUF1 is confounded; use paired within-replay
  comparison only. Backbone feature caching changes RNG consumption relative to
  end-to-end historical passes; fixed head seeds define this replay, not exact replay
  of historical RNG. No guaranteed gain or60% target.

## Resource/profile gate

- Separate **CPU-only** private kernel; must see0CUDAdevices. Never provision a
  competingGPUjob or stop main to admit this diagnostic.
- Platform10800seconds, driver7200seconds,2CPUthreads, initialRAM≥12GiB,
  runtimeavailableRAM≥4GiB. No model/feature checkpoint outputs; only bounded text artifacts.
- Actual native CPUprofile: DEV median/p95/longest by frozen duration (no metric-based
  selection). Conservative featurewallestimate=`max(seconds/audioseconds)*totalduration`;
  full599admitted only when estimate<6000seconds. Otherwise preserve failure/profile,
  no automatic retry or smaller population promoted as full result.
- No batch skip. Interrupted/partial predictions cannot yield COMPLETEfullpopulation result.
- Existing main3150exports remain untouched. Local download only lightweight result
  text into this report directory; no additional weights/features copy on low-disk laptop.

## Evidence and boundaries

Code: `research/crottc_if_official/eval_replay.py`, preparation:
`tools/prepare_crottc_eval_replay.py`, CPU tests:`tests/test_crottc_eval_replay.py`.
Kernelregistration/package source hashes are pinned before launch.
Scope remains local-protocol development, **not paper reproduction or GOAL2 common-F1**.
Production repair/selection/recipe migration would require separate explicit provenance;
this diagnostic does not silently apply such a change.
