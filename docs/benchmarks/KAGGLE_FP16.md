# RQ1-BENCH-01 — private Kaggle T4 FP16 lane

## State and boundary

**LOCAL_WRAPPER_TESTED; CORE_INTEGRATION_PENDING; REMOTE_EXECUTION_BLOCKED.**
This is an inference-only deployment wrapper, not another evaluator. It calls
`langfens_bench.runtime.main(argv)` with **KAGGLE_FP16** fixed. Default mode is
`preflight`; neither attaching a GPU nor supplying the authorization flag alone
selects evaluation. `evaluate` additionally requires
`--authorize-remote-inference`. The common runtime must independently verify
Kaggle/T4 hardware and all scientific input gates; the flag is not evidence of
hardware, quota, privacy, or approval. There is no CPU/FP32 fallback.

The 80 ordered validation utterances are omission-enriched (eight speakers),
not a representative population. Protocol O uses gold canonical identities only;
Protocol Z uses pinned CMUDict G2P from prompts. Neither uses gold timestamps as
inference inputs. DIAG-02B's 9/80 unchanged archive-tolerance failures and
**NO_GO_TRAINING_CAUSAL_GATE** remain unchanged. Cross-platform parity is not
historical archive parity, causal validation, or deployment readiness.

## Offline private staging prerequisites — not performed

Metadata: `deploy/bench_kaggle_fp16/kernel-metadata.json`, kernel
`minkhoaa/langfens-rq1-bench-fp16-private`, `machine_shape=NvidiaTeslaT4`, GPU
on, private, internet **off**. Source attachments are intentionally empty until
approved private staging is available. The package is not runnable as staged
metadata alone. Never attach public datasets, notebooks, kernels, or models.
Keep all staging and outputs private under **CC BY-NC 4.0**, preserving dataset
license/attribution notices. Do not publish generated labels/predictions, data,
checkpoints, cache weights, or credentials in Git or public Kaggle assets.

After separate operator deployment authorization, prepare immutable inputs:

| Location | Required contents |
| --- | --- |
| `/kaggle/input/langfens-rq1-bench-source/repo` | Reviewed integrated source snapshot, including `src/langfens_bench/runtime.py`, existing model/evaluation dependencies, frozen contract/split, CMUdict and prompts at repository-relative paths. Freeze and independently verify the integrated source commit and file hashes before staging; the lane base alone lacks A's runtime. |
| `/kaggle/input/langfens-rq1-bench-artifacts` | Four unchanged checkpoints and four authorized train/validation archives at the exact artifact-root-relative paths in `contract.json`. Keep archives separate by fold; do not extract a cross-fold audio union or stage active-test members for that fold. |
| `/kaggle/input/langfens-rq1-bench-cache/huggingface/hub` | Offline Hugging Face cache for `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, including complete snapshot/blob links contained in the private staged cache. No download fallback. |
| `/kaggle/working/<new-run>` | New output directory only; ephemeral, within the 20 GB ceiling, never the sole durable evidence store. |

Install/verify dependencies from approved offline wheels or the preinstalled
runtime before launch; do not clone source or install from the network at run
time. Record actually installed versions. No credentials are needed by this
offline launcher. Do not search for credentials or mount remote data streams.
Keep original Drive archives unchanged; this lane performs no uploads.

The common preflight must verify all contract/resource/checkpoint/package
hashes, all 80 IDs against each fold's validation whitelist, and active-test
speaker exclusion in archive inventories **before reading any WAV/TextGrid**.
Missing staging, dependencies, pinned offline cache, hardware, or authorization
are blockers, not grounds to fall back or relax checks. No blanket extraction.

## Invocation (future authorized Kaggle runtime only)

Run as a fresh Python **script process**, not in a notebook process that has
already imported torch/transformers. The wrapper establishes offline flags,
`HF_HOME`/`HF_HUB_CACHE`, `MKL_CBWR=COMPATIBLE`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and one-thread environment controls before
importing the common runtime. Inputs must resolve strictly under
`/kaggle/input`; outputs strictly under `/kaggle/working`; symlink escapes and
existing outputs fail closed. The runtime is imported only from staged source.

```bash
# Default: read-only preflight checks, not evaluation or a launch command.
python /kaggle/input/langfens-rq1-bench-source/repo/deploy/bench_kaggle_fp16/bench_kaggle_fp16.py \
  --outdir /kaggle/working/rq1-bench-fp16-preflight-001

# ONLY after separate operator approval and successful common-runtime gates:
python /kaggle/input/langfens-rq1-bench-source/repo/deploy/bench_kaggle_fp16/bench_kaggle_fp16.py evaluate \
  --authorize-remote-inference \
  --outdir /kaggle/working/rq1-bench-fp16-evaluate-001
```

Alternate private attachment mount names can be supplied with `--source-root`,
`--artifact-root`, and `--hf-cache`; the same containment rules apply. There is
no `--profile`, device, precision, or fallback override. Common-runtime failure
and nonzero exit status propagate without retrying another profile.

Future deployment must follow the repository's stage-with-`--no-run`, Web UI
review/approval, and monitoring workflow, with verified sufficient GPU quota.
This lane neither pushed metadata nor checked quota nor launched a kernel.
Preserve `metrics.json`, deterministic `predictions.json.gz`, and
`environment.json` in an approved private persistent store with SHA-256
readback before releasing the ephemeral runtime. Record source/semantic hashes,
contract/checkpoint hashes, ordered IDs, completion state, and artifact hashes.
Do not infer completion from an output directory's mere existence.

## Common-runtime precision and telemetry acceptance requirements (A owns)

These are required integration checks, **not claims that CUDA was executed**:

- Use only verified Kaggle T4 `cuda:0`, even if two T4 devices are allocated;
  reject missing CUDA, wrong hardware, or absent authorization. TF32 disabled.
- Keep stored model parameters FP32: no `.half()` conversion. FP16 is only
  CUDA autocast with `dtype=torch.float16` surrounding original forward calls.
  LOCAL_FP32/KAGGLE_FP32 must not accidentally enable this autocast.
- Cast binary/CTC logits and attention to FP32 **before** sigmoid, softmax,
  entropy, sums, or other reductions. Accumulate metrics in NumPy float64.
  Test dtype/order using synthetic inputs and mocked autocast; mocks do not
  establish genuine FP16 numerical parity.
- Set all parameters frozen, `model.eval()`, `torch.inference_mode()`, seed
  1337, deterministic algorithms, one CPU thread, batch size 1, offline pinned
  pretrained loading. No backward, optimizer, train calls, or model changes.
- One untimed warmup per fold; synchronize CUDA before and after timed original
  model forward(s), excluding staging/load/offline scoring. Save per-utterance
  latency, p50/p95, throughput, peak process RSS bytes and peak CUDA allocated
  and reserved bytes. Report actual observations only; no speed/memory claim
  from mocks, estimates, or metadata.
- Preserve Protocol O/Z semantics, active lengths, original averaged attention,
  `diag_del_ins` alignment, null undefined metrics, complete ordered records,
  and source/artifact integrity. No evaluator fork in this wrapper.

A's runtime is absent at this lane's base commit, so its internal precision,
hardware and telemetry behavior cannot be validated here. Coordinator must
review/test these controls after integration and before any remote approval.

## Numerical drift comparison — instructions, not executed results

After authorized genuine LOCAL_FP32 and KAGGLE_FP16 runs of the same integrated
runtime and frozen contract, use the coordinator-owned comparison command:

```bash
PYTHONPATH=.:src python -m langfens_bench compare \
  --reference /private/complete-local-fp32-run \
  --candidate /private/complete-kaggle-fp16-run \
  --thresholds /private/frozen-versioned-thresholds.json \
  --outdir /private/new-fp16-comparison
```

Freeze/version the thresholds file **before** measurements and report its
SHA-256. FP16 gates from the contract: pointwise probability atol **0.005**,
logit atol **0.05**; absolute AP/AUC/PER/omission recall/FNR deltas **0.01**;
attention mass/normalized-entropy deltas **0.01**; absolute peak-offset delta
**0.04 seconds**. Apply to all required corresponding Protocol O/Z fields,
with both pointwise and aggregate gates required. Missing/misordered IDs,
missing fields, nonfinite values, incompatible hashes/semantics, corrupt
artifacts or incomplete runs fail closed. No implicit tolerance relaxation.
Also compare KAGGLE_FP32 versus LOCAL_FP32 under the separately frozen FP32
gates; do not misattribute general platform drift to FP16 alone. Latency and
memory are descriptive, not scientific parity gates. No parity PASS until
actual environments execute and the independent reviewer accepts the evidence.
The historical DIAG-02B archive tolerance of 0.000200 is not waived by FP16 gates.

## Lane process checkpoint

- **Actual starting state:** isolated `/tmp/langfens-bench-kaggle-fp16`, branch
  `work/rq1-bench-kaggle-fp16`, clean HEAD
  `955de17020f9f21ad9c28e1b5242d4e39f3b9b93`. Read AGENTS, complete frozen
  contract/JSON, global index, and DIAG-02B checkpoint before edits.
- **Frozen inputs:** contract JSON SHA-256
  `18700f2bf1f29d65f153ba7c34e2b4d9d69eafe33cce86c1756212cc1ddd4ffd`;
  CONTRACT.md SHA-256
  `3a92354d182df933f6ecf74a7a405da033791d6165698b95e690d8dba59d9de5`.
  No real datasets/checkpoints/audio/annotations read or staged by this lane.
- **Outputs:** owned wrapper/metadata, `tests/test_bench_kaggle_fp16.py`, and
  this lane documentation/checkpoint only. No A/B/coordinator files changed;
  no global index update (coordinator owns it).
- **Executed local evidence:**
  `PYTHONPATH="$PWD:$PWD/src" /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -o addopts='' -q tests/test_bench_kaggle_fp16.py`
  passed all **26** launcher tests in **0.07 s** (mocked common runtime, no CUDA imports or
  model inference). Covers default/fixed-profile delegation, explicit
  authorization, offline/deterministic env before import, no precision/fallback
  overrides, propagated mocked hardware/cache blockers, path containment,
  missing staging, no overwrite (including dangling symlinks), staged-runtime
  origin, and private metadata. Script `--help` passed; unauthorized script
  `evaluate` refused with expected exit **2** before any runtime import.
  Python AST and metadata JSON syntax checks passed.
- **Not executed:** A core integration, real preflight, any local/remote model
  inference, numerical comparison, latency/memory measurement, full repository
  tests, GPU/quota/credential/network calls, deployment, training/backward.
  Successful optimizer updates: **0**; skipped updates: **not applicable**.
  Last valid model artifacts: unchanged four original 3,600-update Baseline R
  checkpoints pinned in the frozen contract; no new model artifact exists.
- **Safe restart:** verify lane commit/status and frozen hashes; review/cherry-pick
  owned files, integrate A's runtime and coordinator comparison; run bounded
  non-training tests and independent review. Obtain new explicit authorization
  before private staging/deployment; verify all remote prerequisites and run
  default preflight first. Do not treat this lane checkpoint as a GPU execution
  record, PyTorch checkpoint, or scientific acceptance.
