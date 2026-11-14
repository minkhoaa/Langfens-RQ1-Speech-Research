# Optimized full-paper training mandate — 2026-11-14

User clarified: “tôi muốn chay full paper train cho cấu hình optimized luôn ấy, sau đó đưa tôi link training đang chạy để tôi monitor”. The optimized configuration is for actual registered full training, not a benchmark-only deliverable. Do not stop after scratch profiling or the resume smoke. Do not call one AM300 stage full-paper reproduction.

## Latest execution update — 13:02 +07

**Optimized long training has now launched:** `https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001`, private kernelversion1, platformRUNNING/initializing. Server metadata/privacy/script hash were downloaded and checked; `OPTIMIZED_CAMPAIGN_02/LAUNCH_RECEIPT.json` binds it to the verified parent and full300epoch/no-update-cap registration.

Smoke export is fully downloaded/verified: **482/internalepoch8/cursor832**, SHA `7db07c849fd1ca0c0a2f14720212c9f7954e0b197d662d8c7265b05a3e753582`,64ordered TRAIN utterances, exact full source-state import, two-GPU RNG, active scaler, and all537Adam moment states with exact per-parameter active/inactive step ledger. Receipt: `OPTIMIZED_CAMPAIGN_02/resume_smoke/VERIFICATION.json`. The prior12:50notes below are historical snapshots, not the current execution state.

## Gates already completed

- Legacy window002/1 COMPLETE and full latest checkpoint **480/epoch8/cursor768** downloaded/SHA/load/full-state/row-order verified. SHA `382bc2e0ee69caaf54ac0ca8d52ed5a1713166615840e9bbd0c17dc0a23f6179`, receipt `AM_CAMPAIGN_02/window002/VERIFICATION.json`. Eight completed AM epochs; preserve all previous training, no rollback125. Legacy controller stopped cleanly between windows due to registered STOP.
- Allocator confirmation COMPLETE/downloaded/verified. `OPTIMIZED_CAMPAIGN_02/PROFILE_VERIFICATION.json` selected **two T4, FP16 autocast, FP32 losses/master/Adam, microbatch32/effective32**, expandable segments. Median5.0276s/update versus same-runtime legacy baseline (~7.742x). Original1.5GiB/device reserve and4GiB RAM floor retained. Scratch timing is not yet full-window end-to-end speed.
- Optimized native resume smoke from480 produced real updates **481 and482**, finite loss and positive SSL feature-projection weight deltas. At12:50 its Kaggle status is COMPLETE; full export currently being downloaded for independent SHA/load/state/ordered-batch/Adam-per-parameter-ledger audit. Do not bypass this remaining gate.

## Full training execution

Local CPU-only controller PID1141950 (`tools/kaggle_crottc_optimized_campaign.py`, code-v8) is configured to submit the first long kernel automatically **only after smoke export verification**:
`https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001`.

That URL is the planned long-training kernel, **not yet a running link at the12:50snapshot**. Verify actual platform admission plus live update events before telling the user it is training. Send this link once running, not the completed profile/smoke links.

Each long window:14400driver seconds/16800platform seconds, target300totalAM epochs, no optimizer-update cap. Checkpoint full state at most50updates or300seconds, whichever first, plus everyepoch/window. Download/verify each full parent before next window. At most64long windows; stop on error/admission refusal/nonfinite/hash/state/row mismatch, insufficient RAM/VRAM/disk, or explicit STOP. Never delete historical artifacts or silently drop partial gradients. CPU/RAM used for cached audio and bounded preprocessing, not dummy saturation. No local production model training, no cron/timer.

Status/log/registration/export receipts: `OPTIMIZED_CAMPAIGN_02/`. Outputbase: `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_optimized_campaign_02/`. Laptop must remain awake/networked for download/verify/submission; remote active windows run on Kaggle.

## Remaining full-paper scope

AM CROTTC300 → registered AM CTC300/IF CTC200/IF CROTTC200 parent gates (IF schedule interpretation remains provisional), then remaining baselines/ablations/main-result/analysis/generalization/challenge coverage. Current optimized controller implements AM CROTTC only; other full-paper stages still require their native optimized adapters/profiles/state/recipe/evaluation gates. They remain mandatory pending work, not a claimed automatic completion. Test remainsSEALED; unresolved exact-paper access staysNOT_REPRODUCED. No guarantee of author's71.77% or full-paper completion date.
