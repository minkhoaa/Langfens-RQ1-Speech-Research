# P-SCORE bridge portability — independent QA checkpoint

## Initial state before this recheck

The previous finding and reports remain immutable history. This task independently verifies the new checkout-relative test fix; worker assertions are not accepted without rerun.

- Coordinator `/home/khoa/Projects/KLTN/Dataset`, `prep/rq1-shared-base`, HEAD `b7a6a5e3750b051614dec57e2ec2a88cf3d414b7`; only pre-existing status is nested `m repo`; no staged paths.
- P-SCORE `/home/khoa/Projects/KLTN/langfens-rq1-p-score`, `prep/rq1-p-score`, HEAD `0f65dd26119ef1c38ce413fe6fa15d773f67fa21`; clean/no staged paths. Portability test source commit claimed `69d05e36c83b7ce10cb679730855ea440cf10e8e`.
- P-SUP `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`, `prep/rq1-p-sup`, HEAD `a5604a5499248480b5cb11f5497b12a0febcdea8`; clean/no staged paths.
- Three worktrees are distinct; common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`; V1 digest expected `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Shared validator is expected unchanged; verify hash and prove test imports it from the P-SCORE checkout, not coordinator or P-SUP.

No source edits by QA. Safe scope: inspect the narrow test path diff, run offline CPU P-SCORE suite + shared JSON 22 + selected safe existing 9, check branch/base/V1/57 provenance/shared bytes/protected sources/nested dirt, then append report/checkpoint/index and per-arm QA/index evidence. Do not rerun P-SUP 88 in this pass; its source must remain unchanged and previous evidence must be labeled historical. No real data/audio/labels/timestamps/model/weights/TEST/inference/backward/optimizer/training/GPU/API/deployment. Preserve all older reports and do not make repeated tip-refresh commits. After final arm commits, record exact arm HEADs; create a coordinator-only receipt outside shared directory and update coordinator index once.

Parent audit notes to record separately from implementation tests: first shared checksum invocation from repository root produced 8 wrong-working-directory errors; rerun from the shared inventory directory passed 8/8. Parent workflow initially had a syntax error before launch (no child launched/no edits); corrected native retry was the only launch. These are process/tool invocation issues, not product test failures.

## Completion — independent verification

Narrow diff from initial arm HEAD `bcee145d5527c8d3ecf14bc0d82e9ebe21044b93` to tested source/test commit `69d05e36c83b7ce10cb679730855ea440cf10e8e` is confined to `tests/test_p_score_shared_schema.py`, adding root derivation and an inside-this-checkout assertion. The strict validator resolves inside `/home/khoa/Projects/KLTN/langfens-rq1-p-score/.../implementation-readiness/strict_runtime_validator.py`; no coordinator/peer fallback. Shared validator hash unchanged.

Independent tests: P-SCORE **269 passed** (245+6+18), shared JSON **22 passed**, selected safe regressions **9 passed**, all offline CPU. P-SUP 88 remains prior evidence and was deliberately not rerun. All 57 provenance hashes matched in each tree, frozen V1 SHA matched in all three, previous 15 shared files identical, inventory check passed 8/8 from the right directory, and nested diff+status compare byte-identically with baseline. No protected source changes. P-SCORE synthetic runtime integration/validator portability **GO**; P-SUP bounded prototype review **GO only**; both experiment/training readiness **HOLD**.

Remaining: proposal/source license/topology parity/scientific margins/private provenance and separate authorization; P-SUP trainer, input adapter and durable recovery do not exist. Training/inference NOT_STARTED, zero successful/skipped/attempts, no artifact. See `P_SCORE_PORTABILITY_QA_REPORT.md` for exact commands and process-error distinctions.
