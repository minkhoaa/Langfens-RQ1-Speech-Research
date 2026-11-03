# Independent Luna QA checkpoint — initial handoff

## Initial state and scope

QA began after the three implementation lanes reported completion. Handoff/model switch was intentional; preceding partial Sol-authored work and subsequent Luna additions must retain their recorded authorship. No active peer writers were identified. This independent QA owns only shared-file synchronization, review/evidence, checkpoints, and index updates; defects are to be reported, not silently patched in arm source.

- Coordinator `/home/khoa/Projects/KLTN/Dataset`, branch `prep/rq1-shared-base`, HEAD `21999dcf57f3e6f72c39158f1a959441b80095ed`; initial status only ` m repo` (nested subrepository dirty), no staged paths.
- P-SCORE `/home/khoa/Projects/KLTN/langfens-rq1-p-score`, branch `prep/rq1-p-score`, HEAD `e9a97c620b7cf4e01284583ef5fee2cfdd35dc4e`; initially clean.
- P-SUP `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`, branch `prep/rq1-p-sup`, HEAD `599c9e65da3bf156f5ebc6a936372bd0c09c46ac`; initially clean.
- All three merge-bases equal frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`; each V1 contract SHA-256 is `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Nested `repo` HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`; current binary diff digest `68e0959819ed0673918da0d097bb365cd5f1a749055399f9fc477c44c8ec2b50`, status includes modified audit scripts and two deleted paper PDFs. This matches the provided historical nested baseline digest. It is not to be touched/staged.

Read AGENTS.md, global process registry, current shared checkpoint and both arm checkpoints. Explicit safety boundaries: no real audio/dataset/labels/timestamps/active TEST, model weights/checkpoints, inference/loading/download, training/backward/optimizer/fitting, GPU/remote/API staging/deployment/push, or repository reset/clean/stash/force/rebase/merge. Inspect relevant tests before CPU execution. Preserve V1 and historical E0/E1/model/parser/trainer/evaluator/metrics source byte-identically. No scientific PASS, winner, or ranking.

## QA plan / restart

Verify branch/base identities, commit ancestry, three V1 digests, 57 shared provenance fingerprints, and original nested baseline. Review shared runtime files, scorer math and brute force suite, P-SUP labels/masks/control/routing/no-grad behavior, strict schema compatibility with actual output rows, metric populations/AP ties/PER/coverage, CLI/preflight gating, and all scoped diffs. Synchronize only the eight new shared implementation-readiness artifacts (excluding generated `__pycache__`, checkpoint/index, and inventory) byte-identically into both arms and add the same registry entry. Inspect test files before running only targeted safe synthetic CPU tests and the already enumerated existing synthetic tests. Record exact commands/results and defects without patching arm code. Add independent QA reports/checkpoints/index updates in all worktrees and coordinator, commit only reviewed scoped paths, verify nested diff/status bytes, and leave experiments HOLD.

No tests or edits had been made by this QA at checkpoint creation. No inference/training performed: NOT_STARTED, 0 successful updates, 0 skipped, 0 attempts; no new artifact. Resource usage is not measured; future resource estimates must be labeled estimates.

## Completion — independent review

Shared new artifacts and identical registry entries were synchronized into both arms. The independent report is `INDEPENDENT_QA_REPORT.md` and identical in coordinator and both arms. Arm-specific reports/checkpoints are under `manifests/RQ1-P-SCORE/implementation/` and `manifests/RQ1-P-SUP/implementation/`.

Verified branch/base/contract identities, 57/57 shared provenance hashes in each of 3 trees, 8/8 shared-file byte identity in each arm, protected historical file equality, and nested baseline. P-SCORE suite **251 passed**, P-SUP **88 passed**, shared JSON suite **22 passed**, selected inherited synthetic regression tests **9 passed**. P-SUP actual emitted treatment/control/failure rows pass the shared strict validator. P-SCORE actual rows fail it because output metadata field `posterior_target_occupancy` contains prohibited name substring `target`; removal of that field alone permits the tested row. No arm code changed. This is an independent integration blocker; do not claim P-SCORE implementation integration GO until corrected and revalidated.

Final scoped dispositions: P-SCORE implementation integration **HOLD** pending strict schema-compatible field and regression; P-SUP synthetic prototype review **GO** only; both real-data experiments/training/deployment **HOLD**. No metric evaluator is implemented (no AP/PER experiment result). Distinct proposal, author license/topology parity, scientific margins, private input/output provenance and separate authorization remain open. DIAG-02B PARTIAL/causal-training NO-GO and E1 rollout NO-GO unchanged. Training/inference NOT_STARTED, 0 successful/0 skipped/0 attempts, no new artifact. No real data/audio/labels/timestamps/weights/TEST/inference/backward/training/GPU/remote/API access.

Safe restart is correction of only the P-SCORE output name with a bridge regression test and rerun; then preserve a new exact source/test SHA. P-SUP has no trainer or deployment-ready package. No resource measurement; qualitative estimates only.
