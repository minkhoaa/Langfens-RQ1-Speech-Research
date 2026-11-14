# Session recovery — 2026-11-14

Prior session `01a1d62f-4dbc-71d9-a830-faa52bd757e5` was inspected read-only. It stopped after three provider responses containing `{"detail":"Bad Request"}`; no further assistant work ran after the YAML bootstrap edit. The transcript does not establish the provider's underlying cause.

## Separate Kaggle failures

- Native profile kernel v1 failed before model construction: HyperPyYAML 1.2.2 with ruamel.yaml 0.19.1 raised `AttributeError: 'Loader' object has no attribute 'max_depth'`. Evidence preserved in `FAILED_PROFILE_V1/`.
- The prior bootstrap edit pins ruamel.yaml 0.18.16, ruamel.yaml.clib 0.2.14 and PyYAML 6.0.3 (matching working local versions), and asserts a real HyperPyYAML parse before staging/model initialization.
- Kernel v2 successfully passed that parser/environment probe, but failed closed at `code inventory changed`: this recovery attempt accidentally selected the original local code-v1 package while mounting server code-v3. It performed no model optimizer updates. Evidence: `FAILED_PROFILE_V2/`. No hash gate was relaxed.
- Kernel v3 uses the exact frozen local `kaggle_code_v3` inventory, SHA-256 `12ec6b6f52998c4899f8c21811d270bcd1b191d78b85389d837f54cebaf37793`, verified against a freshly downloaded server inventory. It runs scratch native profiling only, with the same 1800-second driver/4200-second bootstrap limits. It does not mount campaign state.

## Current-source packaging

The prior session had modified `kaggle_runtime.py` after uploading code-v3 (adding explicit parent SHA and strict subsequent-window resume support). That unuploaded source is now frozen as private runtime dataset version 4, inventory SHA-256 `cab51cdb1dbcb54ec5ccef2f4a9710bb62dba86059f30b7e06827c23ad2d33d6`. Server inventory and every current source file were verified. Only the adapter differs from code-v3; AST comparisons confirm `profile`, `configure` and `validate_migration` are identical. The five historical driver hashes remain unchanged.

Prepared, NOT launched until profile evidence is downloaded/verified: `tools/kaggle_crottc_resume124_v2/`, mounting code-v4 and immutable state-v1, capped at one additional update and 1800 driver seconds. `resume124_v1` was prepared with the superseded package and was never pushed; do not use it.

All three datasets were READY and private at recovery checks. State dataset file listing contains the full 5,058,796,894-byte `last.pt` and 1,277,328,598-byte `best.weights.pt`; a transient listing totalBytes=0 was not treated as proof of an empty upload. Files remain hash-checked after mount. No assets/state re-upload was needed.

## Verification

- Existing adapter/resume/offload/migration tests plus new bootstrap regression: **31 passed**, `ADAPTER_TESTS_V4.log`.
- Broader CPU-only regression: **82 passed**, `CPU_REGRESSION_TESTS_V4.log`. Initial wider invocation exposed CUDA and six IF fixtures intentionally refused to run; corrected invocation set `CUDA_VISIBLE_DEVICES=''` and passed. No production local training was started.
- Original local update124 is never an output target. Official test remains SEALED. Provider transcript unchanged. No Git commit/push/merge performed.

## Execution snapshot at recovery handoff (historical)

**Subsequent status:** profile v3 has now been downloaded/verified; smoke124→125 also downloaded/full-state verified; sustained AM300 window001 is running under the CPU-only CLI controller. Current evidence/control paths are in `FULL_PAPER_EXECUTION.md` and `AM_CAMPAIGN_01/CONTROLLER_STATUS.json`. The paragraph below preserves the earlier snapshot, not the live state.

Kernel `minkhoaa/langfens-crottc-if-native-profile` v3 was RUNNING at **2026-11-14T02:10:37+07:00**. No downloadable output files were available during execution; an empty snapshot folder is not completion evidence. Original local checkpoint SHA was reverified as `d060833722f1898800d00778c05e39df636c3aecf3584723a3ae0dfe0a71ee5c`, and all five historical implementation hashes match the local decision receipt. This is not yet evidence of completed native profiling or campaign continuation. Next: wait for terminal status, download into `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_native_profile_v3`, verify profile/registration/environment, then launch the registered one-update smoke only if successful.
