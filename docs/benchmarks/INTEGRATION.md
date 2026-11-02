# Canonical integrated benchmark interface

This coordinator integration section supersedes earlier *implementation-interface* descriptions in the three lane histories, not their executed evidence or scientific protocols. Both private kernels now use byte-identical reviewed bootstraps except `PROFILE`. Both require the same source SHA, one fold, input inventory and common runtime; default remains preflight. C's earlier directory-only bootstrap is historical.

## Local

See `BENCHMARK_SETUP_REPORT.md` at repository root. Reuse `.venv`; registered direct dependency/Python versions are in `configs/bench_dependency_contract.json`. The root lockfile is stale and must not overwrite this environment. CPU/CUDA build suffixes are recorded separately. Matching CUDA wheels/Python availability is unverified: do not invent a wheelhouse or silently install a different stack.

`python -m langfens_bench {preflight,evaluate,merge,compare}` is the unified CLI. Preflight/evaluate use the same evaluator, contract and source fingerprint on all profiles. Remote commands require `--fold 1..4`; output metadata declares `scope_fold`. Merge requires four complete, distinct fold bundles and reconstructs all 80 frozen IDs, pooled/per-fold metrics and latency quantiles from original records. Peaks are maximum of separate processes, not concurrent model residency. Comparison validates output hashes, both source/weight identities, same IDs/offline references, O/Z score/logit and attention drift, pooled/per-fold AP/AUC/PER/omission/attention metrics. Null required quality yields INCOMPLETE, never PASS.

## Source and mounted-input isolation (local preparation only)

Build an allowlisted source ZIP outside Git, never `git archive` (inherited history contains restricted records):

```bash
.venv/bin/python audit_scripts/package_bench_source.py --output /private/langfens-bench-source-NEW.zip
# After final source freeze, for each k=1..4, write a new inventory locally:
PYTHONPATH=.:src .venv/bin/python -m langfens_bench.staging \
  --fold 1 --source-archive /private/langfens-bench-source-NEW.zip \
  --cache-snapshot /private/verified-hf-snapshot/22aad52d435eb6dbaf354bdad9b0da84ce7d6156 \
  --output /private/fold1-input-inventory-NEW.json
```

The inventory generator hashes only source/pretrained-cache bytes and reads frozen metadata, not TEST payloads or new datasets. Preserve its separately printed inventory SHA and source SHA in a reviewed deployment record. It is not permission to upload/stage anything remotely.

An authorized **single-fold** runtime must contain exactly these six regular files across ALL `/kaggle/input` (directories may exist; symlinks/unknown files/other-fold inputs fail closed):

1. `/kaggle/input/langfens-rq1-bench-fold<K>-artifacts/<registered checkpoint relative path>`.
2. Same artifact root plus `<registered package path>` **or** `<registered package path>.bin`.
3. `/kaggle/input/langfens-bench-source/langfens-bench-source.bin` — unchanged source ZIP bytes, opaque extension prevents automatic extraction.
4. `/kaggle/input/langfens-bench-cache/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156/config.json`.
5. Same snapshot plus `model.safetensors`.
6. `/kaggle/input/langfens-bench-inventory/fold<K>.json` — only the active fold inventory.

Copy the two cache files as ordinary regular files in an approved private staging preparation; do not mount the entire laptop cache, refs/blobs/symlink tree. The explicit external inventory SHA is the trust anchor; inventory bytes do not self-hash. Inventory limits auxiliary files to the exact approved source/cache paths; contract binds fold/checkpoint/package hashes. Every mounted path is inventoried before extra-file hashing, archive/member checks and any WAV/TextGrid read. No inventory means conservative artifact-only mode, which intentionally blocks extra source/cache mounts. Unknown inputs are blockers, never silently skipped.

Kaggle may auto-extract ZIP uploads. Preserve authorized archive bytes as `.zip.bin` at the same logical relative path plus `.bin`, not an extracted directory. SHA is still the original registered archive SHA; no contract/checkpoint modification. Only the selected fold's authorized train/val archive is attached, never the raw corpus or all-fold union. No TEST payload is staged/read. The four runs' output metadata is merged later; speakers legitimately rotate across folds.

## Remote commands — documentation only, NOT authorization

Do not push or execute these kernels during setup. After separate explicit authorization, resolve dependency/wheel availability, verify allocation/quota with official CLI, configure private attachments and supply measured values in a reviewed private deployment copy. A plain Save & Run All on the checked-in script intentionally refuses missing arguments. No credentials/HF_TOKEN/network are needed by these offline inference kernels.

Same bootstrap arguments for both scripts (only fixed profile differs):

```text
--fold 1
--source-sha256 <MEASURED_SOURCE_SHA256>
--input-inventory /kaggle/input/langfens-bench-inventory/fold1.json
--input-inventory-sha256 <MEASURED_INVENTORY_SHA256>
--outdir /kaggle/working/UNIQUE_PROFILE_FOLD_RUN
```

Default command is preflight. To evaluate later requires **both** `evaluate` and `--authorize-remote-inference`, plus successful real preflight; the flag alone is not deployment authorization. Source extraction root must be new on repeat runs. Use cuda:0 T4 only; FP32 TF32 disabled; FP16 CUDA autocast only around forward, FP32 postprocessing and float64 metrics; no fallback, training or backward. Local/mock tests do not certify actual CUDA behavior.

Only after authorization, stage without running: `kaggle kernels push -p <reviewed-private-package> --no-run`. Verify privacy/T4 in UI, attach only approved private fold/source/cache/inventory inputs, run preflight first, preserve evidence in approved private persistent storage with hashes, release idle runtimes. No such command was executed in setup. Current dependency availability, private remote staging/allocation and cross-platform parity remain unverified.

## Coordinator integration checkpoint

A's first parallel run timed out after two completed historical CPU runs; dirty fold edits were preserved, then one same-worktree bounded recovery committed them. B/C completed normally. Coordinator cherry-picked only owned seams, unified bootstraps, reused B's source packager and A's scorer, added explicit approved-input inventory to resolve the otherwise incompatible strict source/cache mount gate, and added measured dependency identity checks. No original model/evaluator/protocol/checkpoint changed. Real predictions remain outside Git. See the global process checkpoint for final commit/test/replay evidence and publication/privacy blocker.
