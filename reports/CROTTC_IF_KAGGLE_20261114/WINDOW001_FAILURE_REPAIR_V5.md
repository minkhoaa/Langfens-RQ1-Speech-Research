# Window001 failure and registered audit repair — 2026-11-14

At the user's08:03status check, remote `am-window001` v1 was ERROR and controller PID511439 had exited at02:51. It did **not** train throughout the intervening hours. It stopped during restoration, before the first new microbatch. Last independently verified campaign state remains update125/internal epoch2/cursor352 (two completed epochs, partway through the third).

## Evidence and root cause

Downloaded JSON/log evidence is preserved at `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_am_campaign_01_failed_window001_logs/`. Remote `DRIVER.log` traces the failure to `kaggle_runtime.py:272`, comparing saved gradients directly with `CPUAdamOffload.capture_gradients(modules)`.

An update-boundary checkpoint deliberately stores `gradients={}`. Capture returns `{'ssl':{},'heads':{}}` when no gradients exist. The old audit considered those different dictionary keys and aborted. This is an adapter-audit bug, not an optimizer/data/nonfinite/quota failure. The original partial-gradient124→125 smoke did not expose this case because it started with537pending tensors; its previous unit tests also missed the native empty-gradient boundary. No new last.pt or new training microbatch was produced by the failed window. No completed epochs were lost, but the failure stopped intended progress.

## Repair and provenance

- `assert_restored_gradients` expands only absent **empty module dictionaries**. It still compares all named tensors exactly and rejects unexpected modules, missing/nonempty gradients, changed values and dtype differences. It never resets gradients or changes training.
- The adapter's own hash necessarily changes. `validate_adapter_migration` permits exactly one registered transition: immutable update125 SHA `66fcf24144d74b575cf173def0e5f9f69a19d9117a70f4416fb9d90da30d7194`, previous adapter SHA `eb388a80a088a4ef572a2584cfee9b23719a49069fdae1cdf6098b49c4353bf4`, to this explicitly frozen audit repair. Every other execution-contract field (including all five historical driver hashes, packages, paths, recipe) and provenance must remain identical. Source125must have537Adam states/scheduler125, complete RNG, cursor352 and no pending gradients. Subsequent windows resume strictly under the repaired adapter hash.
- `ADAPTER_REPAIR_V5_REGISTRATION.json` records old/new contracts and the exact parent. It was validated against the real locally downloaded checkpoint after rechecking its SHA. Historical receipts and original checkpoints are not rewritten.
- Runtime private code-v5 inventory: `945a8051ee5939aa75d0f562c40df24f5fcc9cd721b5dbe2acf19cc9ec44f7c5`. Authorized assets/state are reused, not re-uploaded. Native profile/training implementation is unchanged; only adapter audit/migration dispatch changes.
- Bootstrap now tees driver logs to the live Kaggle stream as well as durable `DRIVER.log`, so future status checks can see actual microbatch/update events rather than only platformRUNNING. A subprocess timeout sendsSIGTERM with checkpoint grace before any hard kill.
- Controller gains a first-window number to preserve failedwindow001 and submit a **new window002**, and downloads JSON/log diagnostics automatically on future remote failure. It remains stop-on-error, not silent automatic retry. Export verification accepts only the same pinned adapter-only migration on the first repaired export; later exports require exact contract equality.

## Verification and continuation plan

`ADAPTER_REPAIR_V5_TESTS.log`: **115passed**, including a real CPUAdam empty-gradient restore/capture regression, tensor-exact pending-gradient rejection, thirteen negative adapter-migration cases, and driver live-log/timeout tests. These are CPU development tests, not local production training.

New campaign will use the already downloaded/verified immutable125parent, code-v5 and unchanged assets-v1, with new report/output directories `AM_CAMPAIGN_02` / `kaggle_am_campaign_02`, first window2, maximum63windows,4hdriver/16800splatform per window, target300total AM epochs. No update cap, no data or optimizer reset, no unrelated training. Exact test remainsSEALED. Full-paper reproduction remains the task and is not complete.

Status at this registration: repair tested, code-v5 upload READY; replacement campaign not yet launched. This document preserves the failure history; consult controller live status for later execution.
