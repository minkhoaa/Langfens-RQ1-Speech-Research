# Independent recheck — P-SCORE shared-runtime schema correction

Date: 2026-11-03. This is an append-only correction to the earlier independent QA report, which remains unchanged and records the original `posterior_target_occupancy` failure. The correction is verified here independently; the worker summary alone was not accepted as evidence.

## Finding and scoped readiness

**The specific synthetic shared-schema integration blocker is resolved.** The narrow source diff changes only the serialized key `posterior_target_occupancy` to `posterior_phone_state_occupancy` and its definition string to `exact_posterior_nonblank_prompt_phone_state_visits`. Internal `PhoneScore.posterior_target_occupancy`, numeric calculation, CTC graph, score orientation, semantics, schema and strict guard remain unchanged. A new bridge suite imports the existing shared validator by exact path, not a copied/relaxed validator.

Independent result: P-SCORE **synthetic runtime/schema integration GO**, scoped to the emitted synthetic rows and validator rules. This is not real-data/cache/model readiness, author topology/code/license parity, numeric parity with a publication, or scientific performance PASS. Both real experiments remain **HOLD**. No winner/ranking claim. Prior P-SUP disposition remains **bounded synthetic prototype review GO only**; no P-SUP trainer exists. No new P-SUP source/test run was performed in this recheck; its source tree is unchanged from the prior independently reviewed commit.

## Identity and integrity

- Frozen common base / all merge-bases: `0d268ff8c444e16fd76aca2f4ad6ece966908edc`.
- P-SCORE source-and-bridge-test commit (tested source SHA): `2ffcaee162c210fadd0dce3c35b1bfe719b17e84`; lane receipt HEAD at start of recheck: `1a5fc5d5d1d87332e06388cea94742116b9e8203`.
- P-SUP unchanged source tree / prior QA documented tree: `d2b9702627fa03c7d2de97ff8e825bb3ffbe0b86` / `599c9e65da3bf156f5ebc6a936372bd0c09c46ac`; current doc-only branch tip at start of recheck `126378ef01fdf00801175f9aecbd6b8876fa5102`.
- Coordinator branch at start: `prep/rq1-shared-base`, HEAD `29f107a558dfa750f6c2b9c3fa53f6810d150e3b`.
- V1 SHA-256 in each tree: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Shared validator SHA-256 in coordinator and both arms: `1eecf657cad3a1831246a9e21ca13618483fd396bcccd714f543cf038bf24e81`; it was not changed or weakened.
- All 57 `SHARED_PROVENANCE.json` fingerprints independently matched in all three trees (171/171). Protected historical source/contract paths have no changes. No concurrent writers were indicated by the clean isolated arm statuses at start. Nested coordinator `repo` remains at `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`, binary diff digest `68e0959819ed0673918da0d097bb365cd5f1a749055399f9fc477c44c8ec2b50`; no nested operations performed.

## Independent tests

Environment: Python 3.14.9, pytest 9.1.1, torch 2.14.1+cpu, `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python`; CPU-only/offline environment with `PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (shared tests use their directory and coordinator `src`).

1. P-SCORE suite: **268 passed** (245 CTC/oracle and control tests, 6 runtime/cache/CLI, 17 shared-validator bridge tests), exact command:
   `cd /home/khoa/Projects/KLTN/langfens-rq1-p-score && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_score_ctc.py tests/test_p_score_runtime.py tests/test_p_score_shared_schema.py`
2. Shared JSON suite: **22 passed**, exact command:
   `cd /home/khoa/Projects/KLTN/Dataset/manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness && PYTHONPATH=.:/home/khoa/Projects/KLTN/Dataset/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m unittest -q test_shared_runtime`
3. Selected safe existing synthetic regression nodes: **9 passed**, exact command:
   `cd /home/khoa/Projects/KLTN/Dataset && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_normalizer.py tests/test_parsers.py::test_parse_gold_textgrid_intervals tests/test_protocol_z_alignment_and_scoring.py::test_alignment_correctness_synthetic_cases tests/test_protocol_z_alignment_and_scoring.py::test_ambiguous_alignment_tie_break_divergence`
4. P-SCORE CLI `audit_scripts/p_score_infer.py --help`: passed. The P-SCORE suite's CLI smoke uses a synthetic namespace only, checks all four scorer coverage keys, `training=NOT_STARTED`, `weights_loaded=false`, and refuses overwriting.
5. Independent bridge coverage: every scorer (`sf_s`, `sf_sd`, `greedy_dp`, `gop_sa`) accepted for normal repeated-phone prompt `[1,1,2]`, no-frame abstention, explicit upstream recording failure, and invalid/nonfinite emissions. Each position was present with unique positional keys; failure/abstention rows had null decision scores; shared strict parsing and finite JSON serialization passed. Nested gold fields remained rejected. Empty prompts were additionally run independently for all four scorers: zero runtime positions, utterance status `abstain`, reason `empty_prompt`, and zero/zero bundle ledger accepted. Duplicate repeated-position injection was rejected.

One initial ad-hoc fixture attempt failed because its synthetic probability row summed to more than one (the test correctly rejected invalid log probabilities). The fixture was corrected to normalized rows and the full scorer/failure/empty check passed. This was a fixture-authoring issue, not an implementation failure.

## Source/bridge review and residual risks

The bridge test uses synthetic arrays and verifies the already synchronized strict validator by absolute coordinator path. It does not change the schema. That absolute path is environment-specific; test portability to a fresh clone remains a minor maintenance risk, but the bridge did run against the canonical validator required by this QA. The bridge's expected position map is generated from its known synthetic prompt fixture, and length/index checks plus strict bundle validation reject missing/duplicate rows.

The scorer remains an independently authored `ctc-union-language-v1` S/SD formulation: no author restricted-topology reproduction, code-license determination, exact paper Eq.13 identity, or real-data parity. No metric evaluator is implemented (no AP/tie aggregation or PER/experiment analysis). Remaining real-experiment gates: distinct latest omission proposal not located; author implementation/license/topology parity; prospective owner effect/noninferiority/false-alarm/coverage margins; approved and verified private cache/model/source provenance and readback; separate experiment authorization. DIAG-02B remains PARTIAL with 9/80 parity issues and causal-training NO-GO; E1 rollout remains NO-GO.

Resource estimates remain qualitative/unmeasured; no runtime/VRAM/performance benchmark. Training/inference NOT_STARTED; 0 successful updates, 0 skipped, 0 attempts, no new artifact. No real dataset/audio/labels/timestamps/weights/TEST, model loading/inference, backward/optimizer/fitting/training, GPU, API, remote staging or deployment occurred.

## Future separately authorized commands

No command below was run or is currently authorized. P-SCORE has an inference-only external-cache CLI (not a model or audio loader); after *all* scientific, proposal/license/parity/provenance gates are resolved and separate operator authorization exists, a future private-cache scoring invocation could be:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python audit_scripts/p_score_infer.py --mode frozen-cache --cache <verified_private_fp32_emission_cache.json> --authorization <separately_approved_private_authorization.json> --authorization-sha256 <reviewed_authorization_sha256> --output <new_private_run_directory> --run-id <unique_safe_run_id>
```

This requires actual verified, allowlisted, authorized cache and authorization files; no such files were read or created. Do not substitute synthetic success for those gates. **P-SUP has no trainer or runnable training command.** Its offline `audit_scripts/p_sup_preflight.py` is preparation auditing only and returns experiment HOLD; do not describe/run it as training or push its metadata package.

This recheck did not alter scientific approvals or authorize either experiment. Parent/orchestrator should record any eventual arm HEADs after the final shared documentation sync commits; implementation/test SHAs above are exact and non-self-referential.
