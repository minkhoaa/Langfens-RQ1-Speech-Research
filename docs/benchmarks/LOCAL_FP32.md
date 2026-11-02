# RQ1-BENCH-01 — LOCAL_FP32 lane / process checkpoint

State: **IMPLEMENTED; LOCAL_EXECUTION_COMPLETE; CROSS_PLATFORM_PARITY_UNMEASURED**.
Execution base: `955de17020f9f21ad9c28e1b5242d4e39f3b9b93`, branch `work/rq1-bench-local`.
Read frozen `manifests/RQ1-BENCH-01/CONTRACT.md`, `contract.json`, `AGENTS.md`, global process index and DIAG-02B checkpoint before implementation. Global index/report integration belongs to the coordinator; no shared scientific files were changed.

## Scope and boundaries

One runtime serves `LOCAL_FP32`, `KAGGLE_FP32`, and `KAGGLE_FP16`; remote wrappers must call this runtime, not duplicate it. CLI:

```bash
PYTHONPATH=.:src python -m langfens_bench preflight \
  --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json \
  --artifact-root /path/to/artifacts --outdir /new/preflight
PYTHONPATH=.:src python -m langfens_bench evaluate \
  --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json \
  --artifact-root /path/to/artifacts --outdir /new/evaluation
```

`runtime.main(argv=None)` returns 0 on readiness/completion, 2 on blocked/failed execution. Existing output paths are refused. Relative contract/code/resource/split paths resolve against the source repository, not cwd. Artifact paths resolve under the explicit root, with escapes refused. Preflight writes only its new output report, never extracts payloads or evaluates. Evaluation requires all hashes, validation IDs, per-fold archive inventories, dependencies, original checkpoint configurations and pinned offline cache to pass first. Full inventories admit only exact authorized train/validation WAV/TextGrid members and reject duplicate, missing, traversal, unlisted or active-test members. Direct archive reads use each fold's immutable local package; no cross-fold audio staging union is created.

Remote evaluation requires `--authorize-remote-inference`, Kaggle runtime identity and verified T4 `cuda:0`. Missing requirements are blockers, never CPU/network fallback. No remote execution was attempted. `__init__` sets MKL compatibility, CUBLAS determinism and offline environment before torch import; model loading additionally uses the verified cached snapshot path, so prior Hugging Face imports cannot enable downloads. Preflight reports installed package versions, torch build and hardware rather than trusting a lockfile.

Reuse: original audio loader, TextGrid parser, pinned CMUDictG2P, strict `build_eval_model`, original forward, tied-score AP, ROC-AUC, greedy CTC decoder, Levenshtein edits and canonical alignment. No original evaluator/model/protocol/checkpoint changed. Existing alignment spelling is **`sub_first`**, implementing **diagonal → deletion → insertion** ties; both names are recorded. No prior fit, mismatched control, masking or forced-attention intervention is run.

All stored model parameters remain FP32/frozen/eval. FP16 uses CUDA autocast only around forward, never `.half()`. Logits and returned averaged attention are converted to FP32 before sigmoid, argmax or diagnostic reductions; metrics use NumPy float64. Gold labels/times are scoring-only; Protocol O forwards gold phone identities, while Z forwards phones derived only from written prompts. Batch size 1, seed 1337, deterministic algorithms and one CPU thread are fixed. There is one untimed **O+Z** warmup utterance per fold. Latency is the synchronized sum of the O and Z forwards per utterance, excluding tensor preparation, loading, staging and offline scoring.

## Stable output schema for comparison

Evaluation produces:

- `predictions.json.gz`: deterministic gzip encoding (`mtime=0`, no embedded filename) of `{metadata, utterances}`. Ordered flat utterances contain `prediction_id, fold, speaker, utterance_id, checkpoint_sha256, labels, error_types, canonical_phones, binary_logits, binary_scores, ctc_ids, ctc_reference_ids, attention_metrics, protocol_z, latency_seconds`.
- `attention_metrics`: `active_frames`, per-token `tokens`, and token-mean `in_window_mass, uniform_in_window_mass, normalized_entropy, absolute_peak_offset_seconds`. Window indices use floor(start/.02), ceil(end/.02), active-frame clipping and at least one frame; peak time is argmax frame × .02 seconds; uniform mass is K/T. Aggregates weight active tokens, not utterances.
- `protocol_z` per prediction: `canonical_phones, binary_logits, binary_scores, aligned_pairs` (`[gold_idx,g2p_idx,op]`, op 0=match/1=substitution), `unaligned_gold_count, unaligned_g2p_count, tie_break, tie_break_semantics`. Alignment is offline, after forwards. Aligned AP/AUC use both diagonal match and substitution pairs, consistent with existing alignment scoring.
- `metrics.json`: `{metadata, protocol_o, protocol_z, attention, telemetry, by_fold}`.
  - `protocol_o`: `n, positives, ap, roc_auc, ctc_per, ctc_edits, ctc_reference_tokens, omission_operating_points`. PER is an unrounded fraction. Operating points `"0.20"` and `"0.50"` each contain `n, recall, fnr`, restricted to gold deletion tokens.
  - `protocol_z`: `aligned_ap, aligned_roc_auc, aligned_tokens, coverage, unaligned_gold, unaligned_g2p`. Coverage denominator is all gold canonical tokens.
  - `attention`: token-pooled means of the four attention fields above.
  - `telemetry`: `timed_utterances, latency_p50_seconds, latency_p95_seconds, throughput_utterances_per_second, peak_process_rss_bytes, peak_cuda_allocated_bytes, peak_cuda_reserved_bytes`. CUDA fields are null locally. RSS is process high-water memory, including loading/preflight, not model-only memory. Per-fold entries repeat aggregate metrics without separate memory attribution.
- `environment.json`: `{metadata, python, platform, installed_versions, deterministic_environment, artifact_sha256, preflight}` and `error` if blocked/failed. `artifact_sha256` covers completed metrics/predictions; no circular environment self-hash.

All three artifacts share metadata: contract hash, profile, source-file hash map, semantic hash (source map + explicit semantic definitions), execution-base commit, all checkpoint hashes, exact ordered IDs, seed, thread/batch size, precision, preprocessing and completion status. Source commit can be a pre-commit execution base; actual executed source hashes are authoritative. Undefined metrics serialize as JSON null. Nonfinite predictions fail the run rather than being silently converted to null. A failed run retains environment failure evidence and is not comparable as COMPLETE.

## Actual validation and evidence

Python/pytest reused read-only from `/home/khoa/Projects/KLTN/Dataset/.venv`; explicit `PYTHONPATH=/tmp/langfens-bench-local:/tmp/langfens-bench-local/src` ensured this worktree's source. Main repository checkpoint/package/cache artifacts were read only. No training, backward, optimizer updates, GPU execution, downloads, remote command, push or fanout.

Executed:

1. `MKL_CBWR=COMPATIBLE CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/tmp/langfens-bench-local:/tmp/langfens-bench-local/src /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_bench_local.py` — **20 passed** (initial implementation validation; final test count recorded below).
2. Same Python/PYTHONPATH, `-m langfens_bench preflight --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json --artifact-root /home/khoa/Projects/KLTN/Dataset --outdir /tmp/rq1-bench-local-preflight-8191d56d` — **PREFLIGHT_READY**. All four package/checkpoint hashes, pinned resources/cache, all IDs, configs and inventories verified before payload access.
3. Same Python/PYTHONPATH, `-m langfens_bench evaluate --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json --artifact-root /home/khoa/Projects/KLTN/Dataset --outdir /tmp/rq1-bench-local-evaluate-8191d56d` — **COMPLETE**, 80 ordered utterances / 2,951 canonical tokens / 107 gold omissions. No runtime errors. Expected pretrained backbone task-head warnings occurred; the production model then loaded the full checkpoint strictly.
4. Independent Python output audit checked equality of all artifact metadata, COMPLETE status, exact ordered IDs, 2,951 token count, every evaluator source hash, both artifact hashes, all vector dimensions, finite logits/scores and Protocol O aggregate recomputation — **passed**.
5. Final `tests/test_bench_local.py` rerun — **23 passed**, including synthetic FP16 cast-before-sigmoid/warmup checks, strict checkpoint budget/fold/FP32-weight rejection and remote identity gate. The synthetic precision test uses a CPU fake model; it is not a real CUDA FP16 run.
6. Same deterministic Python/PYTHONPATH environment, `-m pytest -q tests/test_bench_local.py tests/test_rq1_diag_01_phoneme_omission_fn.py tests/test_rq1_diag_02_temporal_alignment_validation.py tests/test_rq1_diag_02b_counterfactual_validity.py` — **54 passed** (23 new benchmark + 31 existing DIAG regressions). Inspected these selected tests for backward/optimizer/training calls before execution.
7. `git diff --check` — passed. Full repository suite deliberately not run: unrelated tests may invoke prohibited backward/training.

Contract SHA-256: `18700f2bf1f29d65f153ba7c34e2b4d9d69eafe33cce86c1756212cc1ddd4ffd`.
Executed semantic SHA-256: `458683f9eb3885f20abe7238f8e14de99f4877574f9e7152c4689f7045e07fc5`.

Generated artifacts are **outside Git**, not durable storage; preserve privately if needed:

| Artifact | SHA-256 |
|---|---|
| `/tmp/rq1-bench-local-evaluate-8191d56d/metrics.json` | `adf19a619b08b016f1122ed32a05a39d2f3cff3322c0ea974cef61d4bcc55e28` |
| `/tmp/rq1-bench-local-evaluate-8191d56d/predictions.json.gz` | `30d82c4c6ff9296a286be2a91c3a4024524c3165e6e356d6e118fe9bc8ee6fa4` |

Measured subset metrics (not representative-population claims): O AP **0.2921056810**, ROC-AUC **0.7016605764**, CTC PER **0.2355198883** (675/2,866); omission recall **0.3364485981** at .20 and **0.0186915888** at .50. Z aligned AP **0.2862954393**, AUC **0.6947429300**, coverage **2935/2951 = 0.9945781091**. Attention mass **0.0339700628**, K/T **0.0274542440**, normalized entropy **0.2605000607**, absolute peak offset **1.3922784803 s**. Synchronized O+Z latency p50 **2.231465 s**, p95 **3.210272 s**, throughput **0.460491 utterances/s**, process peak RSS **2,102,087,680 bytes**. These are local measurements only, not comparative speed claims.

Installed versions are recorded in environment: torch `2.14.1+cpu`, torchaudio `2.11.0+cpu`, transformers `5.20.2`, NumPy `2.5.4`, SciPy `1.18.1`; no claim of matching historical CUDA environments.

## Restart / open gates

Verify branch/commit and current source hashes against the environment artifact before comparing. Re-run only into **new** output directories with unchanged contract, original artifacts and explicit validation whitelists. Keep generated predictions/labels/weights out of Git. Forward successful optimizer updates **0**, skipped benchmark updates **0/not applicable**, last valid model artifacts remain the unchanged four original 3,600-update endpoints.

Remote prerequisite checks and real CUDA FP32/FP16 execution are unmeasured, requiring separate explicit deployment authorization. Local tests of precision boundaries do not establish real T4 parity. Coordinator must integrate packaging, run independent review and implement the frozen comparison thresholds before declaring any cross-platform PASS. Historical DIAG-02B archive failures remain unchanged and were not waived or relabeled; this lane does not implement the separate historical archive comparison.
