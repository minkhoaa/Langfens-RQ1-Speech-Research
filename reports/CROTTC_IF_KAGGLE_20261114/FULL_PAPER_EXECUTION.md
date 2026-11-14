# Full-paper execution mandate — 2026-11-14

User reaffirmed: “tiếp tục full training toàn bộ full paper đi, 300 epoch”, then “nhiệm vụ của bạn là tái lập full paper nhé, sau khi train xong thì hoàn thành full paper luôn”. This is the active reproduction task, not permission to stop at a training milestone or to open sealed official test assets.

## Scope and completion rules

- Finish full native WavLM Large AM CROTTC to the registered **300 total epochs**, continuing immutable update124; do not restart, shorten the schedule, substitute a compact model, or discard seven pending microbatches.
- Preserve registered parent-gated stages: AM CTC300, IF CTC200 and IF CROTTC200. The IF epoch interpretation remains provisional pending recipe/source parity; the new user message does not silently change each IF stage to300.
- After training, continue the entire existing paper coverage matrix: main result, acoustic baselines, IF ablations, LLM comparisons/oracle separation, posterior/attention/decoding analyses, cross-corpus transfer and challenge experiments. The launch-time inventory snapshot is `PAPER_COVERAGE_CURRENT.csv`:31paper experiments still `NOT_REPRODUCED`, plus one active full-native development AM run. The original inventory is `reports/CROTTC_IF_FULL_LOCAL_20261113/PAPER_COVERAGE.csv`; historical statuses are not results of the current Kaggle campaign.
- Register each exact paper experiment and parent/data/resource gates before execution; no unrelated architecture sweeps or thesis-method branches.
- Evaluate permitted development data using the source evaluator and report deviations. Exact paper benchmarks requiring sealed test or unapproved corpus access stay `NOT_REPRODUCED` until their gates are explicitly resolved. Never claim reproduction of the author's71.77% from development metrics or a public-checkpoint smoke.
- Completion of AM300 is not full-paper completion. Full-paper reproduction is also not completion of the thesis's later method-improvement goal.

## Execution control

Native profile v3 has been downloaded and verified:32 authorized representative TRAIN samples, one scratch optimizer update,49.41driver seconds, FP32,2.51GiB peak allocated VRAM. `NATIVE_PROFILE_V3_VERIFICATION.json` binds all downloaded evidence to the registered kernel. Campaign state was not mounted during profiling.

Resume smoke `minkhoaa/langfens-crottc-if-resume124-smoke` v1 is **COMPLETE, DOWNLOADED_FULL_STATE_VERIFIED**: update125, epoch2/cursor352, no pending gradient. New checkpoint SHA-256 `66fcf24144d74b575cf173def0e5f9f69a19d9117a70f4416fb9d90da30d7194`;537Adam states/scheduler at125, RNG complete. All25remaining microbatches8–32 occurred in exact frozen row order. Evidence: `RESUME124_SMOKE_V1_VERIFICATION.json` and `RESUME124_SMOKE_V1_EVIDENCE/`. It used code-v4, assets-v1, state-v1,1800driver seconds and a one-additional-update cap. The state migration audit must prove weights, CPU master weights, Adam, scheduler, scaler, RNG and pending gradients restored before microbatch8. Export must be downloaded, SHA-checked and CPU-loaded before a sustained launch.

The CPU-only local CLI controller is **running, PID511439**: `tools/kaggle_crottc_campaign.py`. Its98CPU regression tests passed (`CAMPAIGN_FULL_CPU_REGRESSION.log`). It performs no local training. For AM continuation it registers unique, private, immutable-parent Kaggle windows, each with **14400driver seconds /16800platform seconds**, no optimizer-update cap that truncates the300-epoch stage, and checkpoints every5updates/each epoch/final partial batch. Each finished export must be downloaded and independently audited before the next push.

The finite controller campaign allows at most64windows (256driver hours), stops early at300AM epochs, and stops on any quota/admission refusal, nonfinite/data/state/contract mismatch, failed runtime, status deadline, or local free disk below14GiB. No retries that hide failures, no deletion of historical checkpoints, no alternate platform. Quota balance is unknown, not assumed. Outputs are retained on Kaggle and downloaded under `/var/tmp`; filesystem limits may halt continuation before300 and must be reported rather than hidden.

A `STOP` file in the controller report directory prevents the next window, without killing the active runtime or losing its checkpoint. Controller lock prevents duplicate controllers for that campaign. The initial controller handles **AM CROTTC only**; subsequent stage adapters/native gates remain work to complete, not an implied automatic all-paper implementation.

## Current status

At2026-11-14T02:45:24+07:00, sustained kernel **`minkhoaa/langfens-crottc-if-am-window001` v1 RUNNING**, source smoke output version1 pinned explicitly. No additional-update cap; target300total AM epochs, not300additional epochs. Current verified campaign parent is125 (completed2epochs); progress inside the active window is not yet independently exported/verified. Live local control status: `AM_CAMPAIGN_01/CONTROLLER_STATUS.json`; log: `AM_CAMPAIGN_01/CONTROLLER.log`; registration: `AM_CAMPAIGN_01/window001/WINDOW_REGISTRATION.json`. Future outputs: `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_am_campaign_01/windowNNN/`.

**Keep this laptop awake/networked for checkpoint retrieval and subsequent-window submission.** It runs only the CLI controller, never model training. The active Kaggle window continues if the local controller disconnects, but the next window will not launch until the controller can retrieve/verify its parent. Do not launch a duplicate controller. To prevent the next window, create `reports/CROTTC_IF_KAGGLE_20261114/AM_CAMPAIGN_01/STOP`; this does not kill the active runtime.

Original local update124 remains unchanged. Official test remains SEALED. No production local training, Git commit/push/merge, or paper-complete claim. AM stage orchestration is active; all remaining paper experiments are still mandatory work after their respective gates, not automatically completed by this controller.
