# Additional local checkpoint capacity — 2026-11-14

Status: **PROVISIONED_AND_VERIFIED; PRODUCTION_RUN_NOT_MIGRATED**.

The full native GPU/RAM AM run remains active and unchanged. Home has about
11.4 GiB available; a private directory on the separate local root filesystem
has about 62.4 GiB available:
`/var/tmp/langfens-crottc-full-20261113-t_car8m4`.
This is local disk, not Drive, Colab, or Kaggle. No dataset is copied here.

Ownership UID1000/mode0700, a real 16 MiB atomic write/fsync/read SHA-256 probe,
distinct filesystem devices, the live retention guard and its directory lock,
and refusal to relocate an active GPU run were verified. See
`LOCAL_STORAGE_VERIFICATION.json`. The guard PID at registration is389676,
exec session34894; check actual process liveness before relying on that PID.
State and log are under the experiment root as `LOCAL_STORAGE_GUARD_STATE.json`
and `LOCAL_STORAGE_GUARD.log`. A guard is not a training job and holds no GPU lock.

The planning allowance is39 GiB:24 GiB for all four stages' last/best states,
5 GiB for an atomic replacement,2 GiB for logs/receipts and8 GiB reserve. These
are storage estimates, not measured native IF checkpoint sizes. Runtime disk
checks still apply. Historical profile/checkpoint copies remain untouched.

## Retention evidence

The installed `systemd-tmpfiles` was exercised only against an explicit config
and prefix inside disposable pytest directories. Sticky regular files survived
age-zero cleanup; ordinary files were removed. A shared lock on the private
child directory protected its subtree during cleanup of the configured parent.
An initial test incorrectly locked the explicitly configured cleanup root;
that test failed and was corrected to match the actual `/var/tmp` parent policy.
No global cleanup command was executed.

The guard holds the private directory lock and marks owned regular files sticky
every30 seconds. Sticky regular files retain protection when the guard exits;
the directory's own sticky bit would not provide this behavior. These semantics
follow [systemd temporary-directory documentation](https://github.com/systemd/systemd/blob/main/docs/TEMPORARY_DIRECTORIES.md)
and [tmpfiles.d](https://www.freedesktop.org/software/systemd/man/latest/tmpfiles.d.html).
This protects against tmpfiles ageing, not arbitrary manual deletion or disk loss.

Eleven tests passed, including installed tmpfiles behavior, actual Linux
`RENAME_EXCHANGE`, hashes and preservation of the source, active-GPU-lock refusal,
and rejection of dataset files/symlinks. The log is
`experiments/crottc_if_full_local_20261113/LOCAL_STORAGE_TESTS.log`.
These tests use small storage fixtures; they do not claim production migration.

## Cutover at the next terminal training window

1. Inspect `RUNNING_WINDOW.json`, `am_crottc/STATUS.json`, the actual process,
   log and checkpoint receipt. Wait for the current training window to finish
   and release the shared GPU lock. Never launch duplicate training.
2. Verify the guard is live and root matches `LOCAL_STORAGE_REGISTRY.json`.
   If the guard exited, restart it with the command below; its singleton lock
   prevents duplicate guards.
3. Run the relocation command. It acquires the shared GPU lock exclusively
   before reading or changing the run. It checks the current checkpoint receipt,
   copies all run artifacts, hashes both copies and fsyncs the destination.
4. Linux [RENAME_EXCHANGE](https://man7.org/linux/man-pages/man2/rename.2.html)
   atomically changes the logical run path to a symlink. The entire original
   directory remains under `am_crottc.before_local_storage.<uuid>`; no source
   checkpoint is deleted. The logical experiment path, model/training code and
   execution contract remain unchanged. Check the relocation receipt, destination
   and original hashes, and registry before resuming.
5. Resume `gpu_train --stage am_crottc --resume --wall-seconds 14400` with the
   existing offline/thread environment from `GPU_RAM_EXECUTION.md`. Register
   its actual PID/session/log in the control manifests. Confirm advancement and
   a new loadable atomic checkpoint. Do not reuse the CPU-to-GPU migration receipt.

```sh
.venv-crottc-official/bin/python -m research.crottc_if_official.local_storage guard --root /var/tmp/langfens-crottc-full-20261113-t_car8m4
.venv-crottc-official/bin/python -m research.crottc_if_official.local_storage relocate --root /var/tmp/langfens-crottc-full-20261113-t_car8m4 --stage am_crottc
```

Use `relocate` for the other registered stages only while GPU work is idle and
after checking their existing state. An absent stage receives storage only,
with no training status/checkpoint. This grants no parent-stage completion:
IF gates still require actual AM300/IF200 evidence. Any interrupted relocation
retains staging and receipts for inspection; never delete/retry over them blindly.
Native production relocation and resume on the new filesystem remain unexecuted.
