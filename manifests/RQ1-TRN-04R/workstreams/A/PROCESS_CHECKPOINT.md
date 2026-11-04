# LANGFENS RQ1-TRN-04R — Workstream A Process Checkpoint

**Status:** `IMPLEMENTED_SYNTHETIC_SMOKE_PASSED; REAL_INFERENCE_PENDING_COORDINATOR_FREEZE`
**Lane:** Workstream A, branch `review/trn04r-a`
**Base:** `d332997e732315292d673a3bd42a03698182f5f6`
**Scope:** evaluation-only runner, raw Z schema, protocol correction, source/runtime manifest.
**Execution boundary:** zero training/backward/optimizer updates, no real-data/model inference, no TEST payload access, no VAL tuning, no source checkpoint mutation, and no privacy/publication changes.

## Inputs read and frozen identities

- `AGENTS.md`; `docs/process-checkpoints/INDEX.md`; `manifests/RQ1-TRN-04/PROCESS_CHECKPOINT.md`; `manifests/RQ1-P-SUP-BENCHMARK-V1/final-v2/PROCESS_CHECKPOINT.md`; and `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`.
- Recovered runner `audit_scripts/run_psup_recovered_benchmark_v1.py`, SHA-256 `f19d2e1f2d62b1646979a06c9bfe7b0ddb47c689ca01981827d2ba541ef35d40`, read in full; inspected its zero-oracle dataset/G2P, alignment, labels, runtime, encoder, and frozen TRN-04 model/features/evaluator dependencies.
- V1 contract SHA-256: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- All eight checkpoint weight digests were transcribed from each arm's checked-in `CHECKPOINT_INTEGRITY.json` and placed in the runtime specification. The runner verifies the on-disk bytes against the arm/endpoint digest before `torch.load`.

## Implemented

- `src/langfens_prep/trn04/runtime.py`: CPU FP32 waveform+prompt forward closure, frozen eval-only model handling, raw full-precision runtime and greedy CTC recognition rows, gold-only offline `sub_first` alignment join, and deterministic JSONL serialization.
- `audit_scripts/run_trn04r_benchmark.py`: coordinator entrypoint with configurable checkpoint root, split, authorized ZIP, private output, arm(s), endpoint(s), local cached backbone, and CPU threads. CAL+VAL raw Z rows are generated and hashed before the first TextGrid read. It runs all eight cells by default, extracts one batch-one shared feature cache, keeps raw files outside Git, and applies within-speaker cyclic mismatch with prompt held fixed and features recomputed from mismatched audio.
- Correction/spec/index artifacts distinguish implementation from execution and contain no copied benchmark values.

## Executed verification

1. `../Dataset/.venv/bin/python -m py_compile src/langfens_prep/trn04/runtime.py src/langfens_prep/trn04/__init__.py audit_scripts/run_trn04r_benchmark.py` — passed.
2. Gold-free synthetic CPU FP32 smoke using an in-memory encoder and randomly initialized T2 head: forward pass, two full-precision raw rows, greedy CTC row, and offline-only `sub_first` join with a prompt-extra row — passed. This smoke did not load checkpoints, audio, G2P corpus data, gold labels, or run backward/update.
3. `../Dataset/.venv/bin/python audit_scripts/run_trn04r_benchmark.py --help` — passed and confirms CLI options; no inference invoked.

The provided interpreter emitted a benign `sys.prefix` warning because it was invoked through the sibling checkout's relative path. No failures occurred.

## Pending and restart

Real inference remains **not executed** and must only be launched by the coordinator after freezing the integrated revision. No raw predictions or actual run hashes exist yet; `TRN04R_RAW_PREDICTION_INDEX.json` explicitly remains `NOT_EXECUTED`. Output root must be a new private directory outside Git, e.g. `/home/khoa/.cache/langfens/trn04r/<unique-run>`; dataset archive and checkpoint paths must be supplied from the authorized allowlist. Use `.venv/bin/python audit_scripts/run_trn04r_benchmark.py --checkpoint-root <authorized-checkpoint-root> --split manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json --zip <authorized-train-val-zip> --private-output /home/khoa/.cache/langfens/trn04r/<unique-run> --threads 4`. Omit `--arm` and `--endpoint` for all four arms and both endpoints. Do not resume into an existing output directory.

Before execution, independently review the integrated runner and ensure checkpoint bytes match allowlisted hashes. After execution, verify private run output hashes and source hashes, then coordinator populates actual raw prediction index. No execution is licensed by this checkpoint itself.

## Process registry

The global checkpoint index is coordinator-owned and was intentionally not edited in this lane. Coordinator should add this lane checkpoint to `docs/process-checkpoints/INDEX.md` during final integration. No global synthesis, results, or completion claim is made here.
