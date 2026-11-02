# RQ1-BENCH-01 — benchmark infrastructure setup

**Status: PARTIAL — integrated source/tests ready; real Kaggle dependency/staging readiness and safe publication remain BLOCKED. No remote execution authorized.**

## Frozen inputs and evaluator

- Original HEAD: `467c8233c3f19381c66f10d5fa64d1f44a1c6066`.
- Pre-delegation contract commit: `955de17020f9f21ad9c28e1b5242d4e39f3b9b93`.
- [Shared contract](manifests/RQ1-BENCH-01/CONTRACT.md), [machine-readable inputs](manifests/RQ1-BENCH-01/contract.json).
- Contract SHA-256: `18700f2bf1f29d65f153ba7c34e2b4d9d69eafe33cce86c1756212cc1ddd4ffd`.
- Four unchanged Baseline R endpoints, each 3,600 successful updates. Fold 1 original epoch-6 best.pt; Folds 2–4 fixed-budget checkpoints, never validation-best substitutions.
- Same 80 IDs as DIAG-02B: ten omission-containing utterances per speaker, eight own-fold validation speakers. This is omission-enriched, not a representative sample.
- One common inference runtime, Protocol O and zero-oracle G2P Protocol Z scored separately. Gold timestamps only offline diagnostics; no training/backward or TEST payload access.
- [Registered thresholds](configs/bench_parity_thresholds.json), SHA-256 `a7ae27142555118549b8126f0b7a8603f0a6a164c82ef297db6716f1a62a4c72`. Explicit pointwise and aggregate gates; historical DIAG-02B tolerance unchanged.

## Parallel isolation and integration

Three agents were launched concurrently, from the same contract commit, in manually created isolated Git worktrees:

| Lane | Worktree | Ownership |
|---|---|---|
| LOCAL_FP32 | `/tmp/langfens-bench-local` | Common runtime, local wrapper/tests/documentation |
| KAGGLE_FP32 | `/tmp/langfens-bench-kaggle-fp32` | FP32 private package and shared Kaggle staging guards |
| KAGGLE_FP16 | `/tmp/langfens-bench-kaggle-fp16` | FP16 private package/tests/documentation |

Coordinator owns comparison, metadata-only fold merge, source packaging, integration and final checkpoint. All lane commits were reviewed and cherry-picked without overlapping ownership. A's timeout was recovered once in its same worktree; all three lane worktrees are clean. FP16 and FP32 bootstraps were unified during coordinator review (identical except PROFILE), replacing C's earlier directory-only interface. Canonical final deployment/inventory instructions: [INTEGRATION.md](docs/benchmarks/INTEGRATION.md). No original checkpoint, model, evaluator or registered scientific protocol is changed. The old 50-update training benchmark is not imported or reused.

## Interface / commands

Use the compatible existing CPU `.venv`; do not install the stale root requirements-lock.txt over it. `configs/bench_dependency_contract.json` records the actually imported Python 3.14.8 / torch 2.14.1+cpu stack and pins package versions (CPU/CUDA build suffixes separately recorded). Matching private CUDA wheels and Kaggle Python/dependency availability are **unverified blockers**, not a promised installable environment; dependency drift fails closed. An alternate pinned stack requires a reviewed revision and fresh local reference. All output directories must be new. Keep real output bundles and source staging ZIPs outside Git/private only.

```bash
# CPU preflight; evaluates no utterances
PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/python -m langfens_bench preflight \
  --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json \
  --artifact-root "$PWD" --outdir /tmp/langfens-local-preflight-NEW

# CPU inference only: all 80 authorized validation utterances, no backward
PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/python -m langfens_bench evaluate \
  --profile LOCAL_FP32 --contract manifests/RQ1-BENCH-01/contract.json \
  --artifact-root "$PWD" --outdir /tmp/langfens-local-eval-NEW

# Allowlisted offline source bundle: never git archive (history includes restricted records)
.venv/bin/python audit_scripts/package_bench_source.py --output /tmp/langfens-bench-source-NEW.zip

# AFTER separately authorized, isolated per-fold remote runs produce downloaded private bundles:
PYTHONPATH=.:src .venv/bin/python -m langfens_bench merge \
  --inputs /private/fp32-fold1 /private/fp32-fold2 /private/fp32-fold3 /private/fp32-fold4 \
  --outdir /private/fp32-merged-NEW
# Repeat merge independently for FP16; no audio/checkpoint is read by merge.

PYTHONPATH=.:src .venv/bin/python -m langfens_bench compare \
  --reference /private/local-fp32 --candidate /private/fp32-merged-NEW \
  --thresholds configs/bench_parity_thresholds.json --outdir /private/parity32-NEW
PYTHONPATH=.:src .venv/bin/python -m langfens_bench compare \
  --reference /private/fp32-merged-NEW --candidate /private/fp16-merged-NEW \
  --thresholds configs/bench_parity_thresholds.json --outdir /private/drift16-NEW
```

Each completed evaluation exports `metrics.json`, `predictions.json.gz`, `environment.json`. Compare checks same ordered IDs, checkpoint/contract/source/seed semantics, output hashes, offline reference labels, G2P alignment, pointwise scores/logits, AP/AUC/PER/omission recall/FNR/attention deltas. Missing/incompatible bundles block; undefined required quality is INCOMPLETE, not PASS. Latency p50/p95, forward throughput and peak RAM/VRAM are reported without a speed acceptance gate.

## Remote safety and prerequisites

Use [canonical integrated instructions](docs/benchmarks/INTEGRATION.md); [FP32](docs/benchmarks/KAGGLE_FP32.md), [FP16](docs/benchmarks/KAGGLE_FP16.md) and [local](docs/benchmarks/LOCAL_FP32.md) retain historical lane checkpoints. An externally SHA-pinned input inventory authorizes exactly one fold's two artifacts, one source archive, two pinned cache files and the inventory itself; unknown/symlink/other-fold mounts fail closed. Opaque `.zip.bin` package transport preserves the registered bytes and prevents Kaggle automatic extraction. Both remote packages are private T4 inference only, cuda:0 only. FP32 disables TF32; FP16 stores original FP32 weights and uses explicit CUDA FP16 autocast, then FP32 postprocessing and float64 metric accumulation. No CPU fallback. Dependencies/offline cache/private staging/hardware must pass preflight on the real Kaggle runtime later; they are not certified by mocked tests.

**One active fold per Kaggle runtime/input mount.** Do not attach a union of all train/val archives: a speaker authorized in one fold may be TEST in another. Four fold runs merge only metadata/predictions locally after download. Do not stage/extract active-fold TEST audio or labels. No public dataset sources or credential-bearing source packages. Private source/artifact staging and any Kaggle push/launch require separate explicit authorization. This session executes none of them.

## Evidence and limitations

**Final executed checks: 198 selected inference-safe tests passed in 28.91s.** Compilation, all command help interfaces and whitespace checks pass. Source freeze `1ca31c268ba1a8aa4cd6d59ee5887f4cb81c234a` completed CPU preflight and full 80-utterance/2,951-token replay at `/tmp/rq1-bench-integrated-local-fp32-1ca31c2`. Independent audit verified all output/source hashes, exact IDs, shared metadata, recomputed metrics and baseline logits (max difference 0 against canonical DIAG-02B CPU). Full hashes and evidence: [FINAL_VERIFICATION.json](manifests/RQ1-BENCH-01/FINAL_VERIFICATION.json).

Local-only measurements: AP **0.292106**, ROC-AUC **0.701661**, PER **23.552%**; O+Z forward latency p50 **2.085 s**, p95 **3.088 s**, throughput **0.4723 utt/s**, peak RSS **2,101,837,824 bytes**; VRAM null/not applicable. These are not cross-platform comparisons or representative-population estimates. The final output bundle is the source-compatible local reference for later separately authorized remote runs.

Deterministic 32-file source ZIP built twice byte-identically: SHA-256 `3bb18f11ace5cb3e54ff9c90e6d358df881cd165226cbfdbf7830c18f2228dc6`, 1,104,824 bytes, no datasets/weights/credentials/Git history. Every source fingerprint matches its packaged byte content. An exported-package **CPU** preflight also returned PREFLIGHT_READY (no .git, source_commit null, source hashes authoritative). Four fold inventory drafts were generated locally, not uploaded/approved remote staging. All real outputs/packages remain private local `/tmp` artifacts, ephemeral/non-durable; preserve in an approved private store only after authorization.

| Environment | Verified setup evidence | Unresolved gate |
|---|---|---|
| LOCAL_FP32 | Compatible measured environment, real preflight and 80-utterance replay, exact baseline logits | Private durable retention of generated results |
| KAGGLE_FP32 | Private metadata, shared package/bootstrap, fold/SHA/dependency/hardware guard tests | Matching Python/CUDA wheels, private mounts/allocation and real remote preflight NOT VERIFIED |
| KAGGLE_FP16 | Same interface/manifests as FP32; controlled autocast/postprocessing/telemetry tested with CPU fakes | Same remote prerequisites; real CUDA FP16 and numerical drift NOT EXECUTED |

Historical coordinator checks: 80 IDs validated (8×10), all own-fold validation, no active-test speakers; 80 selected inference-safe regressions passed. Separately, 18 comparison/dependency tests passed. The original parallel workflow ended FAILED when A timed out after 30 minutes; B/C completed cleanly. A's committed two CPU replays were independently verified against executed commit source objects, ordered IDs and artifact hashes. Its 2,951 saved baseline logits match canonical DIAG-02B CPU logits exactly; scalar/vector sigmoid differences are at most 5.96e-8. A fold-scope partial patch was preserved before one bounded same-worktree recovery; recovery commit `5d1d3a7` passed 66 lane/DIAG tests and was integrated. These historical CPU outputs are not final-source cross-platform reference bundles. Full repository suite deliberately not run because it includes prohibited training/backward tests and raw/test-corpus reads.

DIAG-02B remains PARTIAL / NO-GO for causal training authorization: 9/80 archived-score failures at unchanged 0.000200 tolerance, maximum 0.001304954. No benchmark setup result resolves that scientific blocker. No cross-platform parity, precision quality or performance claim is made until real runs exist.

## Publication blocker

`gh repo view minkhoaa/Langfens-RQ1-Speech-Research --json isPrivate,nameWithOwner` returned `isPrivate:false`. The requested branch descends from unpushed DIAG-02B commit `9428d15`, which adds token-level labels/predictions/acoustic diagnostics. Pushing the benchmark branch would also publish those inherited records, even if this setup commit contains source only. This is not an approved private data destination. No push will be attempted without a private remote or an explicitly reviewed source-only publication plan; no history rewriting or force push. Existing unrelated untracked `manifests/RQ1-DIAG-02B/DELIVERY.md` is preserved untouched.

## Safe restart

Read [process checkpoint](manifests/RQ1-BENCH-01/PROCESS_CHECKPOINT.md), verify final commit and hashes, re-run setup-only tests. Resolve prerequisites and publication destination before delivery. **STOP before any Kaggle deployment/benchmark launch.**
