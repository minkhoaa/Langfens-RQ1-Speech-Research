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
- `audit_scripts/run_trn04r_benchmark.py`: coordinator entrypoint with configurable checkpoint root, split, authorized ZIP, private output, arm(s), endpoint(s), local cached backbone, and CPU threads. CAL+VAL raw Z rows (including separate raw recognition records) are generated and hashed before the first TextGrid read. It runs all eight cells by default, extracts one batch-one shared feature cache, keeps raw files outside Git, and applies a fully recorded within-speaker cyclic mismatch using cached encoder/acoustic features from the shifted audio with the prompt held fixed.
- Correction/spec/index artifacts distinguish implementation from execution and contain no copied benchmark values.

## Executed verification

1. `../Dataset/.venv/bin/python -m py_compile src/langfens_prep/trn04/runtime.py src/langfens_prep/trn04/__init__.py audit_scripts/run_trn04r_benchmark.py` — passed.
2. Gold-free synthetic CPU FP32 smoke using an in-memory encoder and randomly initialized T2 head: forward pass, two full-precision raw rows, greedy CTC row, and offline-only `sub_first` join with a prompt-extra row — passed. This smoke did not load checkpoints, audio, G2P corpus data, gold labels, or run backward/update.
3. `../Dataset/.venv/bin/python audit_scripts/run_trn04r_benchmark.py --help` — passed and confirms CLI options; no inference invoked.

The provided interpreter emitted a benign `sys.prefix` warning because it was invoked through the sibling checkout's relative path. No failures occurred.

## Pending and restart

Real inference remains **not executed** and must only be launched by the coordinator after freezing the integrated revision. No raw predictions or actual run hashes exist yet; `TRN04R_RAW_PREDICTION_INDEX.json` explicitly remains `NOT_EXECUTED`. Output root must be a new private directory outside Git, e.g. `/home/khoa/.cache/langfens/trn04r/<unique-run>`; dataset archive and checkpoint paths must be supplied from the authorized allowlist. Use `.venv/bin/python audit_scripts/run_trn04r_benchmark.py --checkpoint-root <authorized-checkpoint-root> --split manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json --zip <authorized-train-val-zip> --private-output /home/khoa/.cache/langfens/trn04r/<unique-run> --threads 4`. Omit `--arm` and `--endpoint` for all four arms and both endpoints. Do not resume into an existing output directory.

Before execution, independently review the integrated runner and ensure checkpoint bytes match allowlisted hashes. After execution, verify private run output hashes and source hashes, then coordinator populates actual raw prediction index. No execution is licensed by this checkpoint itself.

## Follow-up: preserve class-focal three-way score semantics

Coordinator guidance identified a P-SUP-specific ambiguity: its runtime conditionally nulls dedicated subtype scores outside `operation_ce`. The TRN-04R serializer does not use or edit that old runtime. It serializes **all four TRN-04 arms** from the unchanged three-logit operation head as three-way softmax outputs. Runtime/offline rows now explicitly carry objective (`operation_ce` or `class_focal`), `score_semantics`, all three class probabilities, logits, and the derived `s_error`, `s_omit`, `s_sub`. Thus T1/T3 retain dedicated scores; no B-Control semantics or null subtype behavior is applied. The protocol manifest records this rule.

Follow-up synthetic regression command exercised T1 and T3 runtime serialization from finite three-class logits and checked non-null dedicated scores, probabilities summing to one, and explicit focal objective/semantics. It uses no checkpoint/audio/gold and performs no backward/update. D remains responsible for its independent regression coverage.

## Follow-up: integration review corrections (execution remains pending)

Addressed coordinator integration review findings without editing other workstream paths:

1. Mismatch rows are grouped by the complete `speaker_id/utterance_id` key; `AUDIO_MISMATCH_MAPPING.json` stores and hashes the complete ordered within-speaker permutation.
2. Raw `recognition_runtime_Z.jsonl` is frozen before gold reads. Offline `recognition_Z.jsonl` is generated afterward from `labels.ctc`, with `ref_phones`, `hyp_phones`, `edit_counts` ordered `(total, substitutions, deletions, insertions)`, and `reference_phone_count`. The frozen raw recognition file is never rewritten.
3. Runtime rows contain no `operation_target`, `binary_target`, `gold_canonical_index`, or annotation-derived phone fields. Written-prompt phones are named `prompt_phone`; offline joins add gold fields separately.
4. Runtime serialization rejects wrong shapes/dtypes/devices, nonfinite logits/probabilities/scores, inconsistent phone mappings, missing/duplicate/out-of-order prompt positions, and gold-key fields. `s_error` uses Python addition of the serialized FP32 softmax values, consistent with recovered V2 semantics.
5. Mismatch evaluation reuses the immutable batch-one cache instead of re-encoding source audio per endpoint; both model features and 4D features in cache originate from the mismatch audio. The complete permutation is output and hashed.
6. Checkpoint loading verifies exact hash, objective/features/contract/endpoint metadata, successful updates (fixed 3600; best C0/T3 2247, T1 897, T2 3600), and skipped attempts (3) before loading. Run provenance records those counters.
7. The pinned prompt table (`87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e`) is verified before data access; frozen dictionary checksum is verified. Run manifest records command, interpreter, Python/dependency versions, hardware, source hashes including labels/trainer/runtime dependencies, and requires source hashes unchanged at end.

**Executed synthetic validation only:** actual TRN-04 head instances with a synthetic encoder were run in CPU FP32 eval/inference mode for T1 and T3. Tests covered absence of gold-key names in raw rows, complete prompt-map validation and rejection of an incomplete map, full-precision operation serialization, offline `sub_first` join/prompt-extra accounting, and CTC reference/hypothesis/edit-count construction using synthetic CTC reference IDs. No checkpoints, corpus audio, TextGrid, real inference, backward pass, or optimizer update were used. Compilation passed. No final benchmark was launched.

## Follow-up: integration review blockers (benchmark still pending)

Applied the coordinator's latest bounded fixes in A-owned files only:

- Mismatch joins key rows by complete `speaker_id/utterance_id`; mismatched encoder and 4D features now reuse the immutable batch-one cache, and `AUDIO_MISMATCH_MAPPING.json` records the full ordered same-speaker cyclic permutation.
- Frozen raw recognition is `recognition_runtime_Z.jsonl`; `recognition_Z.jsonl` is created only after gold access, using `labels.ctc` reference IDs and the frozen raw hypothesis, with `ref_phones`, `hyp_phones`, `(total,S,D,I)` `edit_counts`, and `reference_phone_count`. Raw recognition remains hash-frozen and unmodified.
- Runtime rows now use prompt-only identity, omit every gold/annotation target/index field, and validate exact keys, CPU FP32 tensor shapes, finiteness, class score/probability consistency, and complete ordered prompt positions. `s_error` sums the Python floats serialized from softmax output, preserving recovered V2 semantics.
- Checkpoints are rejected before load unless exact endpoint hash, arm/objective/features/contract/endpoint metadata, update count and 3 skipped attempts match. Best update counts: C0/T3 2247, T1 897, T2 3600; fixed endpoints 3600.
- Prompt table SHA-256 is checked before data use; dictionary identity remains pinned. Source hash inventory now includes TRN-04 package/model/features/objectives/evaluator/runtime, allowlisted dataset/G2P/alignment/audio/TextGrid/phoneme code, P-SUP labels/trainer, and runner. Command, interpreter/Python, dependencies, hardware, prompt/dictionary hashes and unchanged-source verification are included in the run manifest.

**Executed validation:** runner/runtime compilation; CLI `--help`; JSON specification/index parse; and CPU FP32 synthetic smoke using actual `CompactFactorialOperationHead` instances with a synthetic encoder for T1 and T3. The smoke verified gold-free raw-key schema, non-null focal dedicated scores, serialized FP32 addition semantics, rejection of nonfinite logits and incomplete prompt maps, offline DP join/prompt extra, and offline CTC reference/hypothesis/edit-count output. No checkpoint, corpus audio, TextGrid, real inference, backward pass or optimizer update was used. No benchmark launched.

## Process registry

The global checkpoint index is coordinator-owned and was intentionally not edited in this lane. Coordinator should add this lane checkpoint to `docs/process-checkpoints/INDEX.md` during final integration. No global synthesis, results, or completion claim is made here.
