# Phase-01 runner bounded verification

## Status and provenance

**INCONCLUSIVE for complete runner acceptance; synthetic primitives VERIFIED.** No formal training, fold fit, seed run, protected/DEV_SEARCH access, GPU, or legacy producer main was run. Config copied to `phase01_config_frozen.json` from the frozen supervisor config; source/config hash receipt `SOURCE_FREEZE.json` was written before tests. The implementation is lane-local.

## Implemented

`runner.py` defines separate spec-shaped H-002 target, 43-class CTC and MB03 exact-gap modules; fit-only target/gap standardizers; target and gap tensor construction; CTC/E4 simplex and fixed 50/50 fusion primitives; fit-partition gap weights; official grouped fold/inner plan; explicit unknown-zero enforcement; N/N+1 keyed export guards; AdamW training functions with fixed epoch defaults, per-step optimizer/model/RNG/sampler/cursor checkpoints and provenance validation; and a smoke-budget guard rejecting multi-epoch/full-formal execution.

There is no unified trainable trunk. Inference feature generation/alignment and exact producer E4 competitor/likelihood pipeline are not wired into a complete end-to-end dataset CLI. No raw keyed inference stream or real-data checkpoint was generated.

## Validation

Frozen source/config provenance: `SOURCE_FREEZE.json`.

Command (CPU, two threads, no bytecode):
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python - <<'PY' ... pytest.main(['-q','test_runner.py']) ... PY`

Result: **5 passed**. Synthetic tests covered all three branch finite forward/backward/update and parameter independence; target/gap shape/scaler and class-weight behavior; fixed E4 simplex arithmetic; official grouping, unknown-zero and N/N+1 key validation; one-epoch target/gap/CTC fit/checkpoint writes; and dropout/RNG/optimizer checkpoint roundtrip exact next target update. Test process measured 1.429s wall / 614488 KiB max RSS (Python `resource.ru_maxrss`; synthetic pytest process only, not fit-budget feasibility).

The initial `pytest` PATH and `/usr/bin/time` attempts failed (pytest unavailable on PATH; time binary absent); successful venv pytest invocation followed. No files outside this lane were written.

## Unresolved contract gaps / risks

- **No eligible TRAIN smoke.** This lane was not provided a staged feature/manifest bundle; bounded inspection found no NPZ/PT feature artifact in the inspected MB07 Agent-A directory. Do not substitute synthetic or legacy outer predictions for a fit-partition smoke. Exact data source/feature-loading contract must be integrated by supervisor before acceptance.
- **E4 parity is not established against producer.** Synthetic arithmetic verified only `softmax([0,SLLR,OLLR])`; competitor generation, exact CTC log-likelihood, canonical-mean aggregation and producer parity remain untested/not implemented.
- **Full inference/export** from audio plus supplied intended phones, explicit handling/keying of all 59562/61362 real TRAIN events, and label-free dataflow are not established. The current helper accepts explicit IDs but there is no producer adapter.
- CTC synthetic one-epoch training is finite, but complete optimizer/RNG/sampler resumed-vs-uninterrupted equality is only directly tested on target branch. Must extend exact resume equivalence to gap and CTC.
- Full-budget guard is a function, not an enforced formal-run CLI (no CLI is supplied); no formal fold can be launched from this implementation.
- No real smoke checkpoints, keyed predictions, official terminal-epoch selection receipts, or full-run compute feasibility measurement.

This artifact is not a readiness pass for formal controlled training. Supervisor review/integration is required; all unfulfilled claims remain NOT_TESTED/INCONCLUSIVE and may not be treated as passed gate criteria.