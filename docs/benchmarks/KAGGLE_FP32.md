# KAGGLE_FP32 — private, offline inference boundary

## Status and ownership

**Locally tested infrastructure; remote execution BLOCKED / NOT RUN.** This lane
owns `deploy/bench_kaggle_fp32/`, `src/langfens_bench/kaggle.py`, and its tests.
The single evaluator is `langfens_bench.runtime.main(argv)`, supplied by lane A;
there is no Kaggle evaluator fork. FP16 calls the same shared wrapper with only
its fixed profile changed. See the [frozen contract](../../manifests/RQ1-BENCH-01/CONTRACT.md).

No push, remote API/quota call, dataset upload, credential access, GPU operation,
model forward, backward, optimizer update or training was performed by this lane.
DIAG-02B remains **PARTIAL_ARCHIVE_PARITY_FAILED; NO_GO_TRAINING_CAUSAL_GATE**.
No numerical parity, deployment readiness, speed or memory advantage is claimed.

## Shared API and gates

- `langfens_bench.kaggle.main(profile, argv=None) -> int`: profile must be
  `KAGGLE_FP32` or `KAGGLE_FP16`; default command is `preflight`. Forwards to
  `langfens_bench.runtime.main(argv)` with the fixed profile. Cannot override it
  through `--profile`.
- Common wrapper options: `[preflight|evaluate] --contract PATH --artifact-root
  PATH --outdir NEW_PATH [--authorize-remote-inference]`. Evaluate without the
  explicit authorization flag is refused **before** any staging/runtime import.
- `validate_kaggle_hardware(torch_module=None) -> dict`: checks Kaggle mounts,
  CUDA-enabled PyTorch, and NVIDIA T4 compute capability 7.5 at **cuda:0**. A
  second allocated T4 is reported but unused; no device-1 execution or fallback.
- `build_source_package(repo_root, destination) -> sha256`: exclusive-creation,
  deterministic source-only ZIP. Requires integrated common runtime; excludes
  datasets, checkpoints, Hugging Face cache, `.env` and arbitrary repository
  artifacts. Includes Python source, YAML configs and the four required
  contract/split/CMUdict/prompt resources, preserving project and CMUdict license
  notices. Max uncompressed size: 64 MiB.
- `validate_mount_paths(artifact_root, outdir)` requires existing artifacts
  strictly below `/kaggle/input`, new output strictly below `/kaggle/working`,
  and rejects symlink escapes/overwrites. Preflight creates no runtime output.

Before importing torch/runtime, the helper sets `MKL_CBWR=COMPATIBLE`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, OMP/MKL threads to 1, Hugging Face/Transformers
and datasets offline mode, and disables Hugging Face telemetry. Presence checks
cover torch, torchaudio, transformers, NumPy, SciPy, soundfile, psutil and PyYAML.
Installed versions, import compatibility, pinned offline backbone revision,
checkpoint/resource/package hashes and validation IDs remain mandatory common
runtime checks. No presence-only dependency check is a readiness PASS.

The common runtime, **not this wrapper**, implements FP32 stored weights,
`cuda:0` forward with no autocast, TF32 disabled, frozen eval/inference mode,
seed 1337, deterministic algorithms, one CPU thread, batch size 1, both protocols,
FP32 score reductions and float64 metrics. The wrapper never bypasses its gates.

## Immutable mounts and fold isolation

Expected private inputs (not provisioned by these files):

- `/kaggle/input/langfens-bench-source/langfens-bench-source.zip`: measured,
  reviewed source/resources only. Keep ZIP bytes intact; if the platform extracts
  `.zip` uploads, preserve a private immutable byte-file and pass its actual path
  via `--source-archive`. The filename extension is not significant.
- `/kaggle/input/langfens-bench-artifacts/`: preserve each relative checkpoint
  and authorized train/val ZIP path from `contract.json`; do not substitute a
  Kaggle auto-extracted directory for the hash-pinned archive bytes.
- `/kaggle/input/langfens-bench-cache/`: existing offline HF_HOME including
  `hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156/`
  and all local blobs/configs needed by that pinned snapshot. Cache symlinks must
  resolve within the immutable cache mount. No network cache repair.

The source bootstrap requires both Kaggle mounts, the cache directory, a new
source extraction root and a separately supplied source SHA-256. It checks that
SHA before opening the ZIP, validates **all** source members before extraction,
and refuses traversal, symlinks, duplicate names, encrypted entries, data/weight
members and oversized source packages. It never extracts the fold data ZIPs.

Only the common runtime may inventory/stage the four separately authorized fold
archives. It must verify all hashes/IDs and validation whitelists before any
WAV/TextGrid read, reject each active-fold test speaker in archive inventories,
and retain fold-specific paths/readers. Never stage the raw corpus, merge the
fold audio into one cross-fold union, or extract a sealed active test partition.
Speakers legitimately rotate across folds; authorization is per fold, not global.
Generated predictions/labels/data/weights must stay out of Git. `/kaggle/working`
is ephemeral (20 GB ceiling), not durable evidence storage.

## Offline dependencies and source build (local preparation only)

Use the existing reviewed integrated environment; do not trust a stale lockfile.
The kernel performs **no pip installation or download**. If dependencies are
missing/incompatible, STOP. A separately authorized operator may prepare matching
CUDA torch/torchaudio and other wheels in approved private storage and install
with `python -m pip install --no-index --find-links /kaggle/input/APPROVED_WHEELS ...`
under a separately reviewed dependency plan; this is documentation, not an
executed command or permission for network installation. Re-run preflight and
record actual versions afterward.

After integrating A/B/C and reviewing the source allowlist, build the source ZIP
outside Git using the main environment Python and this repository on PYTHONPATH:

```python
from pathlib import Path
from langfens_bench.kaggle import build_source_package
# repo_root and new_zip_path are explicit operator-selected Paths outside Git.
digest = build_source_package(repo_root, new_zip_path)
print(digest)
```

Keep this measured digest in the reviewed deployment record separately from the
ZIP. Do not invent a digest, reuse one after edits, or accept a digest supplied
only inside the ZIP. The current B-only worktree intentionally refuses a real
source package until A's common runtime has been integrated.

## Safe deployment and operator restart (NOT executed)

All remote deployment, attachment and inference require **new explicit operator
authorization**. Source, kernel, attached datasets and outputs must remain
private under CC BY-NC 4.0. Do not attach public data sources. The metadata fixes:

- id `minkhoaa/langfens-rq1-bench-fp32-private`;
- `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`, `is_private: true`;
- internet disabled, all source attachment arrays empty (safe initial state).

The checked-in script requires `--source-sha256`; a plain Save & Run All without
configuration intentionally fails closed. Before an authorized push, an operator
must supply the measured source digest and any non-default mount paths in the
final `bootstrap(...)` call in a **private reviewed deployment copy** of the
script. For preflight, pass only bootstrap options and optionally `preflight`;
never add `evaluate` to the default. Kaggle script runs do not automatically pass
shell arguments to `bootstrap`. The same options can be passed directly when
invoking `python kernel.py` in an authorized Kaggle session.

Only after separate authorization, inspect quota with the official Kaggle CLI;
STOP if missing/insufficient. Review private metadata and approved attachments,
then stage without launch:

```bash
kaggle kernels push -p deploy/bench_kaggle_fp32 --no-run
```

If using a deployment copy, `-p` must name that reviewed copy instead. In the Web
UI verify privacy/T4 allocation and immutable private inputs. This offline
benchmark needs **no credential or HF_TOKEN**: it never downloads or mirrors
checkpoints. Save & Run All initially runs preflight only. Record real preflight
blockers/results and runtime environment; do not infer success from a staged
kernel or the mocked unit tests.

After successful preflight and separate inference authorization, explicitly pass
`evaluate` **and** `--authorize-remote-inference` to `bootstrap(...)`, using new
`--source-root` and `--outdir` paths on a reused session. This flag is a mandatory
safety gate, not authorization for deployment, training, uploads or network calls.
Both precision lanes enforce it through the same helper and common runtime.

Monitoring/download commands, only when separately authorized:

```bash
kaggle kernels status minkhoaa/langfens-rq1-bench-fp32-private
kaggle kernels output minkhoaa/langfens-rq1-bench-fp32-private -p NEW_LOCAL_OUTPUT
```

Release idle GPU resources promptly through the platform. Preserve completed
`metrics.json`, `predictions.json.gz`, `environment.json` in approved private
persistent storage with verified SHA-256; never report ephemeral-only results as
durable. An interrupted source extraction root is not reusable: inspect failure,
retain evidence and select a new root/outdir. No automatic delete/retry/fallback.

## Lane process checkpoint

- **State:** `LOCAL_BOUNDARY_TESTED; INTEGRATION_AND_REMOTE_GATES_PENDING`.
- **Starting input:** clean `work/rq1-bench-kaggle-fp32` worktree at
  `955de17020f9f21ad9c28e1b5242d4e39f3b9b93`; read AGENTS, full frozen contract/JSON,
  global index and DIAG-02B process checkpoint before editing. Frozen contract
  SHA-256 `3a92354d182df933f6ecf74a7a405da033791d6165698b95e690d8dba59d9de5`;
  contract JSON SHA-256 `18700f2bf1f29d65f153ba7c34e2b4d9d69eafe33cce86c1756212cc1ddd4ffd`.
- **Outputs:** source-only package helper, identical FP32/FP16 wrapper gates,
  T4/mount/dependency checks, self-contained bootstrap, private metadata and mocked
  tests. No scientific protocol/model/frozen manifest edits. Coordinator owns the
  global index and integration checkpoint; this document is the lane checkpoint.
- **Verification:** explicit-worktree-PYTHONPATH main `.venv` pytest command below:
  **28 passed in 0.07s** (mocked contracts/hardware; no actual CUDA or model
  execution). Python compilation and `git diff --check` also passed.
- **Training counters:** backward passes 0; successful optimizer updates 0;
  skipped updates not applicable. Last valid model artifacts remain the four
  unchanged original hash-pinned 3,600-update checkpoints; none loaded here.
- **Safe restart:** verify lane commit/status and frozen manifest hashes; integrate
  A's runtime and C's fixed-profile bootstrap through coordinator; run both lane
  tests with explicit integrated PYTHONPATH; build/measure source package outside
  Git. Do not push, call remote services, stage datasets, start GPU inference or
  relax archive parity without new authorization. Historical DIAG-02B 9/80
  failures at unchanged tolerance remain unresolved.

```bash
PYTHONPATH=/tmp/langfens-bench-kaggle-fp32:/tmp/langfens-bench-kaggle-fp32/src \
/home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -o addopts='' -q tests/test_bench_kaggle_fp32.py
```
