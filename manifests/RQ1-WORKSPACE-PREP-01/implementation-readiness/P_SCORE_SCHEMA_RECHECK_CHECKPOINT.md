# P-SCORE schema remediation — independent recheck checkpoint

## Initial state (before recheck edits)

This is an append-only continuation of the earlier independent QA; the old report/finding remains historical and must not be rewritten. The score worker's follow-up claims are not taken as proof. No source or tests have yet been run by this recheck at the time this checkpoint was created.

- Coordinator `/home/khoa/Projects/KLTN/Dataset`, branch `prep/rq1-shared-base`, HEAD `29f107a558dfa750f6c2b9c3fa53f6810d150e3b`; only existing status is nested `m repo`, no staged files.
- P-SCORE `/home/khoa/Projects/KLTN/langfens-rq1-p-score`, branch `prep/rq1-p-score`, HEAD `1a5fc5d5d1d87332e06388cea94742116b9e8203`; clean/no staged files. Fix source commit claimed by worker: `2ffcaee162c210fadd0dce3c35b1bfe719b17e84`.
- P-SUP `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`, branch `prep/rq1-p-sup`, HEAD `126378ef01fdf00801175f9aecbd6b8876fa5102`; clean/no staged files.
- Three worktrees are distinct; expected frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`; immutable contract expected SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Shared strict validator SHA-256 in coordinator and both arms: `1eecf657cad3a1831246a9e21ca13618483fd396bcccd714f543cf038bf24e81`.

Read AGENTS, global process index, original independent QA report/checkpoint, P-SCORE's append-only correction/checkpoint and new bridge tests. Scope is narrowly independent revalidation and documentation/shared-copy/index only: no arm source patch, no validator/schema change, and no real data, model weights, audio, gold labels/timestamps, TEST, inference, fitting/backward/optimizer/training, GPU/API/remote staging/deployment. Keep all work offline CPU and synthetic. P-SUP source is unchanged; do not rerun its suite or call historical results new evidence. Preserve nested-repository status/diff bytes.

## Planned checks

Verify branch/base/V1 and 57 provenance hashes in all three trees; inspect only the P-SCORE fix/new test diff versus previous QA tip; prove shared validator bytes unchanged and identical; inspect output field keys and accounting/finite/gold guards. Re-run P-SCORE 268 scoped suite, shared JSON suite 22 and selected inherited safe 9. Independently exercise each scorer in normal/repeated, no-frame, recording-failure and invalid-emission cases via shared validator; ensure no forbidden emitted names, nonfinite JSON or missing/duplicate positions. Review CLI smoke is synthetic-only. Then append dated correction/recheck report and checkpoint, sync both byte-identically, update all indexes append-only and commit only these documentation/shared paths on each branch. Final report should distinguish synthetic implementation revalidation from author-topology/scientific performance, and mention P-SUP trainer absence.

## Completion — independent reproduction (2026-11-03)

Rechecked P-SCORE's fix commit `2ffcaee162c210fadd0dce3c35b1bfe719b17e84`; latest arm receipt before this QA's doc commits was `1a5fc5d5d1d87332e06388cea94742116b9e8203`. The original failure finding remains unchanged in `INDEPENDENT_QA_REPORT.md` and the earlier checkpoint section; this recheck is append-only at `P_SCORE_SCHEMA_RECHECK_REPORT.md`. P-SCORE suite **268 passed** (245 CTC, 6 runtime/cache/CLI, 17 shared bridge); shared JSON suite **22 passed**; selected safe existing synthetic regressions **9 passed**. CLI help passed. Independent runs through the unchanged shared validator accepted every scorer for repeated positions, no-frame abstention, explicit recording failure and invalid/nonfinite emission failure; empty prompt produced zero rows with utterance-level `abstain/empty_prompt`; duplicate positions and nested gold metadata were rejected. The initial ad-hoc fixture had invalid non-normalized probability rows and failed as expected; corrected normalized synthetic fixture passed all cases.

All 57 hashes matched in all trees; V1 contract SHA is unchanged; validator hash identical in all trees; all 14 shared non-generated files now byte-identical in coordinator and both arms. Protected historical source/contract unchanged. P-SUP source remains unchanged; its prototype review is GO only based on the prior independent 88-test run; this recheck did not rerun that suite. It has no trainer. Experiments remain HOLD; no author topology/license or scientific performance claim. No real data/model/TEST/inference/training/GPU/API use; training NOT_STARTED, zero updates/attempts, no artifact. Resource estimates unmeasured.

Restart for any newly discovered issue from the current exact source/validator/contract hashes and run only the documented synthetic offline suites. Future frozen-cache P-SCORE CLI template is documented in the report and is not authorized or run. P-SUP has no training command; preflight is not training.
