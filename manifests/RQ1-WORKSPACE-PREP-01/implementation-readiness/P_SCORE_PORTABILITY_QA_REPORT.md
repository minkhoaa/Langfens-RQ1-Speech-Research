# Independent QA — checkout-local shared-validator bridge portability

Date: 2026-11-03. This is an append-only recheck following the accepted synthetic schema integration correction. All earlier reports and failed-QA evidence are preserved; this report verifies the portability change rather than relying on worker claims.

## Result

**P-SCORE synthetic implementation integration: GO.** The bridge test now derives `REPOSITORY_ROOT` from `Path(__file__).resolve().parents[1]`, constructs the shared validator path under that checkout, asserts it lies inside the checkout and checks the imported module's `__file__`. No coordinator, sibling-worktree, environment variable, or alternate path fallback is present. Narrow diff from `bcee145d5527c8d3ecf14bc0d82e9ebe21044b93` to source/test commit `69d05e36c83b7ce10cb679730855ea440cf10e8e` changes only `tests/test_p_score_shared_schema.py` (15 insertions, 4 deletions); shared validator/schema, scorer source and historical files are unchanged. The full P-SCORE suite was independently rerun from the P-SCORE worktree and passed **269 tests**.

**P-SUP:** bounded synthetic prototype review remains **GO only** based on its previous independent 88-test run. It was not rerun in this pass; the arm source/test/config/entrypoint bytes remain unchanged. No P-SUP trainer exists; the preflight is not training.

**Both real-data experiments, training and deployment: HOLD.** No scientific performance PASS, winner, ranking, or AP/PER experiment result is claimed.

## Identities and shared-file verification

- Coordinator at recheck start: branch `prep/rq1-shared-base`, HEAD `b7a6a5e3750b051614dec57e2ec2a88cf3d414b7`.
- P-SCORE: branch `prep/rq1-p-score`, final source/test commit `69d05e36c83b7ce10cb679730855ea440cf10e8e`, current doc receipt tip at recheck start `0f65dd26119ef1c38ce413fe6fa15d773f67fa21`.
- P-SUP: branch `prep/rq1-p-sup`, HEAD `a5604a5499248480b5cb11f5497b12a0febcdea8`.
- All three merge-bases: frozen base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`.
- Immutable V1 contract SHA-256 in all three trees: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Shared strict-validator SHA-256 in all three trees: `1eecf657cad3a1831246a9e21ca13618483fd396bcccd714f543cf038bf24e81`.
- The prior 15 shared non-generated readiness files compared identical across all three trees before the portability addendum. Eight inventory SHA entries passed from the inventory directory. All 57 source/input/archive fingerprints independently matched per tree (171/171). The report and checkpoint created for this pass are synchronized identically to both arms; the final shared tree will be re-compared before coordinator receipt.
- Protected V1/model/parser/normalizer/trainer/evaluator/metrics source paths are unchanged. P-SUP source/test/config/entrypoint is unchanged from the prior QA. No concurrent writer was indicated by clean arm worktrees before QA edits.
- Nested repo baseline is verified byte-for-byte with `cmp`: `repo` HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`; `git diff --binary` matches `/tmp/langfens-rq1-implementation-parent/nested-before.diff`; `git status --short` matches `nested-before.status`. The diff digest remains `68e0959819ed0673918da0d097bb365cd5f1a749055399f9fc477c44c8ec2b50`. Nested state was not changed or staged.

## Test commands and results

Environment: Python 3.14.9, pytest 9.1.1, torch 2.14.1+cpu, `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python`; all tests local CPU/offline using `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

- P-SCORE **269 passed** (245 CTC/oracle+controls, 6 runtime/cache/CLI, 18 shared-validator/portability bridge). Executed from the score worktree with `PYTHONPATH=.:src`:
  `cd /home/khoa/Projects/KLTN/langfens-rq1-p-score && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_score_ctc.py tests/test_p_score_runtime.py tests/test_p_score_shared_schema.py`
  Its new test asserts the actual imported validator path is inside this checkout and its SHA matches the unchanged shared validator. Scenarios include all four scorers, repeated phones, no-frame abstention, recording failure, invalid emissions, strict finite JSON, and gold-field rejection. The suite's cache CLI test is synthetic-only.
- Shared validator JSON-only suite **22 passed**, exact command run in the coordinator's inventory directory:
  `cd /home/khoa/Projects/KLTN/Dataset/manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness && PYTHONPATH=.:/home/khoa/Projects/KLTN/Dataset/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m unittest -q test_shared_runtime`
- Selected safe inherited regressions **9 passed**:
  `cd /home/khoa/Projects/KLTN/Dataset && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_normalizer.py tests/test_parsers.py::test_parse_gold_textgrid_intervals tests/test_protocol_z_alignment_and_scoring.py::test_alignment_correctness_synthetic_cases tests/test_protocol_z_alignment_and_scoring.py::test_ambiguous_alignment_tie_break_divergence`

P-SUP 88 tests are prior evidence only, not a new run. No full inherited test suite was run.

## Process errors, separate from implementation tests

The parent reports the first `sha256sum -c` invocation ran from repository root, so its eight inventory-relative paths were missing. The corrected run from `manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness` passed **8/8**. Parent also reports a workflow syntax error before any child launch; no child was launched and no edits occurred in that attempt. A corrected single-route retry launched this saved work. These are logged as process/tool invocation errors, not implementation or test failures.

## Blockers, resource estimate, and safe next boundary

Remaining research/experiment gates: distinct latest omission proposal still missing; author implementation/code license and restricted topology/numeric parity unresolved; no prospectively approved effect/noninferiority/false-alarm/coverage margins; private cache/source/model provenance and durable storage readback not verified; separate experiment authorization required. DIAG-02B remains PARTIAL with 9/80 parity discrepancies and causal-training NO-GO; E1 rollout remains NO-GO. P-SCORE is an independent `ctc-union-language-v1` S/SD implementation, not author-code or paper Eq.13 reproduction. No metric evaluator or AP/PER performance evidence exists.

P-SUP has no end-to-end trainer, allowlisted/hash-verified input adapter, atomic private checkpoint/recovery, or executable training package. **Do not invent a P-SUP training command.** The offline preflight is preparation-only and returns HOLD.

A P-SCORE frozen-cache command template exists only after all gates and separate authorization, and was not executed:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python audit_scripts/p_score_infer.py --mode frozen-cache --cache <verified_private_fp32_emission_cache.json> --authorization <separately_approved_private_authorization.json> --authorization-sha256 <reviewed_authorization_sha256> --output <new_private_run_directory> --run-id <unique_safe_run_id>
```

Resources remain qualitative/unmeasured unless prior planning docs are cited; no runtime/VRAM/performance benchmark was made here. Training and inference NOT_STARTED; 0 successful updates, 0 skipped, 0 attempts; no model artifact. No real data/audio/labels/timestamps/weights/TEST, inference/model loading, fitting/backward/optimizer/training, GPU/API/remote staging/deployment was performed.
