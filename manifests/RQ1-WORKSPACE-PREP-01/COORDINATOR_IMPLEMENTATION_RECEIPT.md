# Coordinator implementation receipt — independent RQ1 QA handoff

Date: 2026-11-03. Final coordinator receipt after P-SCORE checkout-portability recheck. This coordinator-only artifact intentionally lives outside the shared implementation-readiness directory; it is not copied into arm worktrees.

## Actual final branch identities

Final arm tips were captured **after the final arm-specific QA commits**; no further arm edits are planned.

| Lane | Worktree / branch | Final HEAD | Tested source/test commit(s) |
|---|---|---|---|
| P-SCORE | `/home/khoa/Projects/KLTN/langfens-rq1-p-score` · `prep/rq1-p-score` | `5e4806e1ba03aa549b370f78b8a00938eda27ad1` | occupancy schema correction `2ffcaee162c210fadd0dce3c35b1bfe719b17e84`; checkout-relative bridge source/tests `69d05e36c83b7ce10cb679730855ea440cf10e8e` |
| P-SUP | `/home/khoa/Projects/KLTN/langfens-rq1-p-sup` · `prep/rq1-p-sup` | `cf9c82b606a5bd4517bd1b2b3f80d1660f7f944b` | prototype source `d2b9702627fa03c7d2de97ff8e825bb3ffbe0b86`; prior independently tested documented tree `599c9e65da3bf156f5ebc6a936372bd0c09c46ac` |
| Coordinator | `/home/khoa/Projects/KLTN/Dataset` · `prep/rq1-shared-base` | Resolve with `git log -1 --format=%H` after this receipt commit (self-hash deliberately not embedded) | shared-source commit `21999dcf57f3e6f72c39158f1a959441b80095ed` |

All three merge-bases are frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`. V1 contract SHA-256 is identical in all three: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.

## Current scoped readiness

- **P-SCORE:** GO for synthetic runtime/schema integration and checkout-local bridge-test portability only. Original schema-name defect and subsequent fix are both retained in append-only QA history. The bridge resolves `strict_runtime_validator.py` under the P-SCORE checkout, explicitly asserts imported `__file__` is within that checkout, and has no coordinator/sibling fallback. Shared validator/schema were not changed.
- **P-SUP:** GO for bounded synthetic prototype review only, based on the earlier independent **88-test** run (27 label, 11 model, 50 runtime/preflight). That suite was not rerun in this portability pass; source/test/config/entrypoint bytes were verified unchanged. No P-SUP trainer exists. Preflight is a preparation audit returning HOLD, not training.
- **Experiments, training and deployment for both arms:** HOLD. No scientific performance PASS, AP/PER result, winner or ranking.

## Reproduced validation

Environment: Python 3.14.9, pytest 9.1.1, torch 2.14.1+cpu, `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python`; offline CPU (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).

- P-SCORE at tested source SHA `69d05e36c83b7ce10cb679730855ea440cf10e8e`: **269 passed** (245 CTC/oracle+controls, 6 runtime/cache/CLI, 18 shared-validator bridge/portability). Exact command:
  `cd /home/khoa/Projects/KLTN/langfens-rq1-p-score && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_score_ctc.py tests/test_p_score_runtime.py tests/test_p_score_shared_schema.py`
- Shared strict JSON validator: **22 passed**. Exact command run from the synchronized inventory directory:
  `cd /home/khoa/Projects/KLTN/Dataset/manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness && PYTHONPATH=.:/home/khoa/Projects/KLTN/Dataset/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m unittest -q test_shared_runtime`
- Selected safe inherited synthetic regression nodes: **9 passed**:
  `cd /home/khoa/Projects/KLTN/Dataset && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_normalizer.py tests/test_parsers.py::test_parse_gold_textgrid_intervals tests/test_protocol_z_alignment_and_scoring.py::test_alignment_correctness_synthetic_cases tests/test_protocol_z_alignment_and_scoring.py::test_ambiguous_alignment_tie_break_divergence`
- Bridge scenarios cover all four scorers and repeated phones, no-frame abstention, recording failure, invalid emissions, strict JSON, complete prompt accounting, and nested gold rejection. The added portability assertion verifies the same-worktree validator path. P-SCORE suite includes synthetic-cache CLI smoke; no real cache or model was opened.
- P-SUP’s prior 88 tests are recorded only as earlier independent evidence, not a new run.

## Shared/protected-tree verification

- All 57 `SHARED_PROVENANCE.json` fingerprints matched in each of three repositories: **171/171**.
- The original **15** shared readiness files were byte-identical before the portability addendum; after synchronization, all **17** shared non-generated files compare identical in coordinator and both arms. New portability report/checkpoint SHA-256 values: report `38506233db6a3cc4ab404a114bf0b9790bfd4a2cb432ca32343a43752794670c`; checkpoint `1035ebfc8da7616af38c6b562a445a2323d1fd35fdfb06e78400ff962f9e1f8a`.
- Current 17-file shared-tree manifest: SHA-256 `fea638f1775b261a086c47563393699c3a9c67403f817c81cdf29240d47ccd1f` (canonical sorted `sha256sum` listing of regular files at shared-directory top level; excludes generated `__pycache__`). Core identities: schema `3a0c13695bcf44342107691addb53cc31c30124ddeaaafe3b3feed503ea928a2`; strict validator `1eecf657cad3a1831246a9e21ca13618483fd396bcccd714f543cf038bf24e81`.
- Coordinator SHA inventory was first invoked by parent from repository root, producing eight expected relative-path-not-found errors. Corrected command was run from `manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness`; result **8/8 OK**. This is solely a working-directory invocation issue, not a test failure.
- Protected historical source/contract paths have no diff. P-SCORE code/test tree after commit `69d05e3` and P-SUP code/test tree since `d2b9702` are unchanged by QA documentation commits.
- Nested repo was checked against initial snapshots with bytewise `cmp`: HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`; current binary diff equals `nested-before.diff`, current short status equals `nested-before.status`. Diff SHA-256 remains `68e0959819ed0673918da0d097bb365cd5f1a749055399f9fc477c44c8ec2b50`. No nested action or staging.
- Parent’s initial workflow text had a syntax error (`Unexpected token`, line 14, column 1830) before launch. No child launched and no edits occurred from that attempt; clean arms were confirmed, and one corrected same-route retry launched the saved task. This orchestration failure is distinct from implementation/tests.
- No staged arm paths; coordinator retains only the pre-existing nested `m repo` status. Final arm commits are complete; do not make further arm changes.

## Remaining gates and resource claims

The latest distinct omission proposal is still missing. Author-code license/permission and exact author restricted-topology/numeric parity are unresolved. Prospective owner effect/noninferiority/false-alarm/coverage margins are absent. Private cache/model/source provenance, allowlist/readback and durable storage are unverified. Separate operator experiment authorization remains required. DIAG-02B remains PARTIAL with 9/80 parity discrepancies and causal-training NO-GO; E1 rollout remains NO-GO. P-SCORE `ctc-union-language-v1` is independently authored S/SD, not author code or Eq.13 reproduction. No AP/PER experimental evaluator/result is claimed.

P-SUP additionally lacks an end-to-end trainer, data allowlist/hash adapter, atomic durable checkpoint/recovery and runnable training package. **There is no P-SUP training command.** Its preflight is not training.

Resource estimates are qualitative and unmeasured here; no runtime, GPU, VRAM, or performance benchmark was performed. Training/inference NOT_STARTED; 0 successful updates, 0 skipped updates, 0 attempts, no new artifact. No real audio/data/labels/timestamps/weights/TEST, inference/model loading, fitting/backward/optimizer/training/GPU/API/staging/deployment was accessed.

## Future command templates (not run or authorized)

P-SCORE has an inference-only frozen FP32 emission-cache command, usable only after separate scientific/provenance authorization and verified private files:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python audit_scripts/p_score_infer.py --mode frozen-cache --cache <verified_private_fp32_emission_cache.json> --authorization <separately_approved_private_authorization.json> --authorization-sha256 <reviewed_authorization_sha256> --output <new_private_run_directory> --run-id <unique_safe_run_id>
```

This command was **not** run and no such cache/authorization exists in this QA. No P-SUP execution command is provided or implied because no trainer exists. Parent retains final acceptance/publication.
