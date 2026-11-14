# Resource optimization handover — 2026-11-14 12:24 +07

**Latest status:** legacywindow002 is now **COMPLETE**, remote receipt says update480/epoch8/cursor768/no pending gradients (8completed epochs, partway through9th); controller is downloading full output. Do not certify480locally until download/SHA/load verification completes. Allocatorconfirmation is **RUNNING**, not QUEUED. Optimized production has **not** launched: controller1141950waits for both confirmation and latest verified480parent. Unit/regression count163passed (`OPTIMIZATION_V8_CPU_TESTS.log`). Native migration was deliberately validated only in CPU tests; trying the old125code-v4checkpoint against code-v5strict source guard correctly rejected it—production must use latestwindow002code-v5, never125.

## User mandate and safety interpretation

User explicitly requested maximum resource usage / both GPUs / VRAM/RAM/CPU. Optimize useful native training throughput, not dummy memory pressure or intentional OOM. Full model, 300-total-epoch AM target and full-paper scope are unchanged; effective batch remains32. No local production model training; no cron/timer was created.

## Production progress

Legacy `minkhoaa/langfens-crottc-if-am-window002/1` remains RUNNING; live log at11:56 showed update450/internal epoch7/cursor1632 (nearly finished8th epoch), loss3.7658 and remote checkpoint445. Do not use that old snapshot as current exact progress. The last locally downloaded/verified parent is still125 until window002 exports and is retrieved.

`AM_CAMPAIGN_02/STOP` blocks the next legacy window, not the active one. Controller PID863482 will finish/download/verify window002, then stop. It is not safe to kill the remote job or resume from125. Window002 started around08:12 with14400driver seconds; environment staging is outside that driver clock. Expect terminal status around12:20, not exactly12:12.

## Native benchmark evidence (complete and downloaded)

Primary scratch profile `minkhoaa/langfens-crottc-if-resource-optimization-profile/1`, code-v6, is COMPLETE and independently verified at `OPTIMIZED_CAMPAIGN_01/PROFILE_VERIFICATION.json`. Full JSON/telemetry/logs are at `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_resource_profile_v1/`. Profile SHA:
`115e587afe8469394192d07911c400e1ec60ba6ef279e57ab6e98fe486906484`.

Same96authorizedTRAIN IDs/effective batch32, native315M WavLM/3.8M heads, per-utterance two-view Conformer grouping, FP32 OT/CR loss, two warmups+10measured optimizer updates each, original baseline repeated at end. Loss parity gate passed before timing. All cases really performed12updates; no campaign state mounted.

| Config | Median sec/update | Peak GPU0/GPU1 reserved GiB | Original1.5GiB reserved headroom |
|---|---:|---:|---|
| Legacy FP32 CPU offload mb1 |38.65|3.07/0|PASS|
| Two-GPU resident FP32 mb1 |20.55|8.23/1.61|PASS|
| Two-GPU resident FP16 mb1 |20.39|8.80/2.31|PASS|
| FP16 mb2 |11.53|10.71/4.00|PASS|
| FP16 mb4 |7.36|12.43/7.02|PASS|
| FP16 mb8 |5.74|13.89/12.09|REJECT|
| FP16 mb16 |5.23|13.94/11.77|REJECT|
| FP16 mb32 |5.19|14.32/14.29|REJECT|

Safe original selectionmb4 gives5.25x measured update speedup. This is **scratch update throughput**, not yet deployed production or guaranteed end-to-end speed/accuracy parity. The high reserved VRAM of larger batches is mostly caching/fragmentation: mb32peak actual allocation10.12/7.72GiB, versus14.32/14.29GiB reserved. We did not hide or relax the original safety rule.

## Allocator confirmation and production preparation

A separately registered confirmation tests `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, same baseline+mb4/8/16/32 and same10trials/1.5GiB reserve. Protocol: `RESOURCE_ALLOCATOR_CONFIRMATION_PROTOCOL.json`; private kernel `minkhoaa/langfens-crottc-if-resource-allocator-confirmation/1`, code-v8, currently QUEUED. This may allow a faster physical batch without consuming reserve. No promise of that result. All four CPUs are used for model threads and bounded preprocessing; audio cached in RAM; no fake100%CPU/RAM target.

Source preparation:
- `research/crottc_if_official/kaggle_native_batch.py`: two-GPU WavLM batching, GPU-resident Adam, valid-length masking, per-waveform normalization, original unpadded per-utterance two-view heads. Original loss/backend AST unchanged from primary profile; only diagnostic weight probe changed to the always-connected SSL feature projection (last layer can legitimately LayerDrop).
- `kaggle_optimized_train.py`: full atomic weights/Adam/scheduler/scaler/RNG/cursor/unscaled-pending-gradient continuation, explicit receipt and primary RNG preservation plus declared deterministic secondary CUDA RNG seed. FP16/physical batch/device changes are explicitly non-bitwise-equivalent; no historical driver mutation.
- `tools/kaggle_crottc_optimized_campaign.py`: wait for confirmed profile and **latest verified window002**, then capped2-update optimized smoke; download/hash/load/restore/order audit before long4hwindow. Never rollback to125. Outputs retained, diskfloor14GiB, stop-on-error/noautomatic retry.
- Native WavLM has layerdrop0.1. Original CPU offload effectively updates all initialized Adam master states each step, while native Adam can legitimately leave a skipped layer's state step behind the globalupdate. Do not zero-fill absent gradients or reset/force counters. Native export records all537ordered names, originstep and inactive state IDs perupdate; verifier replays each parameter counter exactly. Pending absence permitted only in original WavLM encoder.layers. All remaining tensors must match shape/dtype.
- Faster checkpoints: at most50updates or300seconds, whichever first, and everyepoch/window, with full atomic reload verification. Avoid serializing3.8GB every5fastupdates; no learning change.

Frozen private code-v8 inventory:
`7d7f78e3d9c5b33b5b8b0de751dba8d15d12f0adfe5c63cb020f4a4c860290ea`.
Runtime datasetversion8 is READY/private. Assets1/state1 unchanged. All five historical driver hashes preserved. Tests: `OPTIMIZATION_V8_CPU_TESTS.log` (CPU-only, native proof still mandatory).

## Live controllers — avoid duplicates

- Legacy PID863482: `AM_CAMPAIGN_02`, STOP blocks next launch, retrieves current output.
- **Optimized waiting controller PID1141950**: `OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json` and `CONTROLLER.log`. It waits for allocator confirmation, then latest verified parent. Args/code frozen-v8, outputbase `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_optimized_campaign_02/`, max64windows plus initial capped2-update smoke. Platform admission/quota not assumed.
- Prior optimized-controller PID1094443 was SIGTERM'd **while only waiting** before any native launch, following code-review findings about allocator/LayerDrop state. `OPTIMIZED_CAMPAIGN_01/STOP` and `PRE_POLICY_REVIEW_STATUS.json` preserve that fact. No remote campaign was killed; original primary profileverification is preserved. Do not restart this superseded waiting controller.

A STOP file in `OPTIMIZED_CAMPAIGN_02` prevents native launch between windows. Keep laptop awake/networked for CLI downloads/submission, not training. Official test remainsSEALED. Full-paper coverage remains work after training; no reproduction-complete claim.
