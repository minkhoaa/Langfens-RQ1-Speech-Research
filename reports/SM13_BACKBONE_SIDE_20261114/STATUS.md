# SM13 backbone side study — 2026-11-14 14:23 +07

**Native side smoke is now COMPLETE, DOWNLOADED/VERIFIED**, not merely platformRUNNING. Receipt: `native_smoke_v1/NATIVE_CPU_FEATURE_VERIFICATION.json`;16featureextractions=8allowlistedTRAIN×2full315Mfrozenbackbones, matchingN/N+1geometries; driver43.91s/bootstrap218.34s/maxsampleRSS2.01GiB. Bothvariants used the same requests and declared normalization. All8adaptedfeature fingerprints differ from native; that is not evidence of better quality. Provisionalparent482is not finalCROTTC. Side retrieval process1310766endednormally. Maintraining/controller1291861stillRUNNING, uninterrupted. Fullheadfitting/developmentmetrics remainpending.

The14:18running/retrieval descriptions below preserve the launch snapshot.

Decision/user authorization is in root `SIDE_GOAL.md`. This is bounded parallel work, not a replacement for full-paper CROTTC-IF training or permission for a historical sweep. No gain or new-method novelty is claimed.

## Work executed

1. CPU-only real inventory audit (`inventory_v1/AUDIT.json`): all1800authorizedTRAIN WAV SHA256 verified,26originalW2V2feature files fully hashed and geometry checked,1800frozen predicted-alignment entries checked,59562targets/61362gaps preserved. Original SM1312/5epoch histories verified for seeds101/202/303; no validationcurve fabricated.
2. New generic feature/head modules under `research/sm13_backbone_side/`: registered13relative-depth layers, W2V2width768 versus WavLMwidth1024; exact original radius8/crop/target/gap semantics. CPU tests prove W2V2features and both head logits/state bit-exact against unmodified historicalSM13 code. WavLM requires new heads/features; input mismatch fails closed.
3. Cache identity binds checkpoint/config/normalization/clock/manifest/order/layers/precision/source. Backbone-training speaker overlap is rejected for claimed independentTRAINOOF. Gold error labels/realizations/timestamps are removed from inference requests; original SM13 source is not edited.
4. **209CPU regression tests passed** in `CPU_REGRESSION_TESTS_V3.log`, including main-runtime checkpoint/placement/stage tests, side feature/head/cache/leakage tests and controller reattachment tests. Synthetic CPU head backward tests are unit checks, not production training.
5. Created one tiny **private allowlisted side package**, `minkhoaa/langfens-sm13-backbone-side-v1/1`, containing source and8TRAINrequests/predicted centers, no audio/weights/rawarchives/test/credentials. It reuses authorizedassets1 and immutableverified optimizedsmoke482output, not a partial/current-training file.
6. **Kaggle CPU-only native feature smoke RUNNING**:
   https://www.kaggle.com/code/minkhoaa/langfens-sm13-backbone-cpu-feature-smoke
   GPUdisabled verified via pulled servermetadata/scriptSHA (`NATIVE_CPU_LAUNCH_RECEIPT.json`),2CPUthreads,1800driverseconds/3600platformseconds. Full315Mnative/adapted backbones are frozen;8representativeTRAINutterances each, no backward/optimizer/AP/F1. Source482is explicitlyPROVISIONAL, not final/bestCROTTC. This tests the real loader/features and finiteN/N+1topology, not quality improvement.
7. One bounded local CLI retrieval process **PID1310766** waits/downloads/verifies this single sidejob. State: `native_smoke_v1/RETRIEVAL_STATUS.json`; log: `NATIVE_RETRIEVAL.log`. Not a cron/timer, not a training launcher. Outputbase `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_side_cpu_feature_smoke_v1/`. It stops on errors; no silent profile retry.

## Findings and remaining gates

-18TRAINutterances have nonmonotonic predictedcenters from legacyfallback. Preserve/report them, never silently sort/remove difficult events. This is a possible alignment limitation, not proof backbone replacement fixes it.
- Head parameters: originaltarget633361/gap215120,total848481; WavLMtarget829969/gap280656,total1110625. S1/S2are geometry/capacity matched; S0versusS1has capacity/normalization differences to disclose.
- Full13layerWavLMcache estimates14.95GiBFP32/7.47GiBFP16 perarm. Do not materialize bothlocalfullcaches or change precision without parity and storage/headroom registration; main checkpoint retrieval has priority.
- Native smoke result has now been downloaded/hash/source/geometry checked for loader readiness. It cannot supply finalS2results. Full final-parent cache, head fitting, learningcurves and permitteddevelopment metrics are still pending.
- FullTRAIN-adapted CROTTC must not be called independentTRAINOOF. Future fullTRAIN→DEV_SEARCH comparison is exploratory because developmentdata have already been reused. Officialtest/DEV_CAL remainclosed. Primarymetric is strictS/O/IAP, not publishedpaperdetectionF1.
- Historical backbone/adaptation experiments were inspected; this is not a newname for already-proven novelty. FullSM13fine-tuning and epochextension require separate registered amendments, not automatic parallel sweeps.

## Primary training health / controller repair

During this task we discovered the local optimizedcontroller had stopped at13:22on a failed read-onlyKagglestatus call (403), **while the remote training kernel remained RUNNING**. Error logs/status were preserved. Added `--attach-window` to reattach exact existingprivatekernel/script/parent without anypush/restart, plus atmost3logged transport-status read retries (10/20s); actual remoteERROR/training/nonfinite failures still stop, never retry training.

CPU regression tests cover attachidentity/parent/script hashes and boundedread-onlyretry. Reattached **PID1291861**, active `minkhoaa/langfens-crottc-if-optimized-window001/1`, mainlive state `reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json`. Evidence: its `window001/ATTACH_RECEIPT.json`, `STATUS_API_FAILURE_1322.log`, timestampedpreviouscontrollerstatus and `CONTROLLER_ATTACH_1404.log`. Native source/checkpoints/recipe300epochs/microbatch32/2T4unchanged. The sideCPUjob does not run on/steal the mainGPUruntime.

No Git commit/push/merge, originalartifact overwrite, localnative productiontraining, testopening, cron/timer, or fullpaper/SIDE_GOALcompletion claim.
