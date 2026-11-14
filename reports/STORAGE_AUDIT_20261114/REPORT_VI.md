# Local storage audit — 2026-11-14 23:06–23:12 +07

**Read-only audit; no files deleted/moved, no sudo/system cleanup, no training launch.**
User asks why obsolete methods are not cleaned after CROTTC moved tocloud.

## Critical distinction: two filesystems

- `/` and `/var/tmp`: `/dev/nvme0n1p2`, Btrfs, free14,596,730,880bytes at snapshot.
  Mainexportfloor14GiB=15,032,385,536bytes; short435,654,656bytes≈415.47MiB.
- `/home` and repo: `/dev/nvme0n1p3`, free7,658,291,200bytes≈7.13GiB.
- Deleting repo method artifacts frees `/home`, **not** CROTTC `/var/tmp` floor.
  `/tmp` is RAMtmpfs, cleaning it does not free either disk partition.
- Btrfs compression/reflinks and hardlinks mean `du`/logicalsize are not exact
  reclaimedbytes; verify`df`after any later approved cleanup, do not promise sums.

## What occupies space

Repo≈102G, experiments≈82G, methods sourceonly≈3.4M.

| Location | `du -h` | Status |
|---|---:|---|
| `experiments/rq1_encoder_bench_01/features/` |22G|Mixed: contains activeSM13S0originalcache; do not delete directory wholesale.|
| `experiments/sm13_three_directions_20261113/` |16G|Historical experiments/raw archives/shared source/cache; retain until explicit archival plan.|
| `experiments/realized_phone_pipeline_20261113/` |14G|Historical receipts/labels/models/predictions/current referenced source; not blanketgarbage.|
| `experiments/crottc_if_full_local_20261113/` |11G|CROTTC source/native/historicalstate still referenced; protected.|
| `experiments/rq1_train_phase_01/` |9.5G|Historicaltrainingstate; preserve unless explicit retention amendment+verified backup.|
| `research/long_running_method_discovery/` |4.8G|Mostlyhistoricalfeatures, not4.8Gcode; no inference/openingsealeddata performed.|
| `.venv/` |5.6G|**In use by official environment through project_runtime.pth**, cannot call unused.|
| `/var/tmp/langfens-crottc-full-20261113-t_car8m4/` |54G|Includes~16Goptimizedcampaignexports,~15GactiveS1cache and otherprotectedhistory/package.|

Featureinventory(metadataonly) in encoderbench:
-26files `w2v2_960h`:**11.210GiB logical**, actively referenced bySM13audits/profile/alignment;
  preserveall26plusalignment metadata.
-8filesHuBERT:**3.449GiB logical**;
-8filesW2V2base:**3.449GiB logical**;
-8fileslegacyWavLM:**3.449GiB logical**.
These24legacy-featurefiles total≈**10.348GiB** are **candidate archival review**, not
certifieddeletable. CurrentlargeWavLM315Mcache is separate. Need source/manifests/
hashes/backup/recomputeability andstandinghistorical-preservation decision before removal.
Noaudio/label contents were read for this size audit. Legacyfeaturepopulation/split
contents have not been inspected; do notuploadthem toKaggle/remote or stage sealed
features by assumingoldcacheisTRAIN-only. Any externalarchive needsboundaryreview.

## Non-research caches

| Cache | Observed size | Notes |
|---|---:|---|
| `/var/cache/pacman/pkg/` |9.6G|Root-owned; relevant to blockedrootfilesystem. Notall9.6Gare safeprunecandidates.|
| `~/.cache/uv/` |16G|Homefilesystem; onlyprunereference-aware, notrmall. OtherSM13sessionalreadyreported~1.6GiBpruned; do notcountagain.|
| `~/.cache/pip/` |418M|Re-downloadcache,homeonly; no systemrootspacegain.|
| `~/.cache/huggingface/` |20G|Models maybesymlinked/reused/offlinepinned; do notdeletewholesale.|
| `~/.cache/yay/` |4.3G|Homebuildcache,notroot;reviewseparately.|

Actualread-only`paccache`dry-runs:
- keep3versions: **0candidates**;
- keep2versions:11candidates,**338.45MiB**logical;
- keep1version:192candidates,**1.93GiB**logical.

Root-ownedoperation needs explicit admin approval/operator action. Keeping1reduces
rollbackhistory comparedtokeep2/3. Even reclaim1.93GiB mayonlyclear14GiBthreshold;
future5–6GiBcheckpointdownload stillrequiresfreshcapacityplanning. No automaticCROTTC
restart after cleanup and nofloorreduction.

UVcacheblockaudit:~15.673GiBblockcount;~5.431GiBmultilinked,~10.243GiBsinglelink.
Hardlinks explain why blanketUVcache deletion need notfree16GiB; existing environments
mustremainintact. This is not exact Btrfsphysicalreclaimmeasurement.

## Recommended retention plan, pending approval

1. First rootFS: prune approvedobsoletepackageversions via packagecachetool, preserving
   chosenrollbackversions; no deleteprotectedresearchstate to clear415MiBshortfall.
2. HomeFS: reference-awarepackagecachecleanup, then cold-archive selectedlegacyfeature
   groups before any removal; keepcode/config/seeds/metrics/predictions/manifests/hashes.
3. Cloudtrainingstates: preserve latestfullresume(best+optimizer/RNG/cursors/pendinggradients)
   locally and verifiedimmutablecloudexport. Earlierlocalcheckpoints are not authorized
   for deletion by assuming RUNNINGcloudsession is durable backup.
4. Explicitlyselectedobsoletecheckpoint deletions require retentiondecision different
   fromcurrentAGENTSstandingpreservation; do not infer authority frommethodfoldernames.
