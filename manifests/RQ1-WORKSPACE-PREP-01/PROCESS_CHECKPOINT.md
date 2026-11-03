# RQ1 two-workspace preparation — shared-base checkpoint

## Preparation record

State: **COMMON_DOCUMENTATION_PREPARED; WORKTREE_CREATION_NEXT; experiments NOT_AUTHORIZED.** This is a documentary continuation of RQ-RESEARCH-01, not an experiment result. Subsequent verification is appended separately in `WORKSPACE_VERIFICATION.md` when available. No current model artifact created; training NOT_STARTED, 0 successful optimizer updates, 0 skipped updates/no attempts.

### Inputs and preservation

- Initial local branch `research-phase`, HEAD `b25a7595360294798d8d06b37592dd65dd4c9c33`; remote tracking happened to match, but local working bytes were newer.
- Pre-existing uncommitted files: global checkpoint index and parent RQ-RESEARCH-01 checkpoint modifications, entire closure/revalidation directories, dirty nested `repo` at `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. Index initially empty. No reset/clean/stash/rebase/force/push/merge or nested repair.
- Created dedicated branch `prep/rq1-shared-base` without discarding changes. Existing closure/revalidation documentary bytes will be committed explicitly with shared preparation, not silently omitted. Nested gitlink and its mode/symlink dirt excluded from staging and preserved. Other worktrees untouched, including stale `/tmp/langfens-e1-launch` registration (not pruned).
- Read AGENTS, registry, latest coordinated closure/checkpoint, source/data/thesis/pilot reports, original UIT proposal and current RQ1 source seams. Latest distinct omission proposal not found in scoped documentary search; conformance remains conditional, as recorded in the addendum.

### Outputs frozen before branching

- `PROPOSAL_CONFORMANCE_ADDENDUM.md`: scope reconciliation and explicit missing-source disposition.
- `EVALUATION_CONTRACT_V1.md`: new immutable contract, not a historical contract replacement.
- `CONTRACT_SHA256`: exact shared contract digest.
- `SHARED_PROVENANCE.json`: current source/input/archive fingerprints, checkpoint identities as archival metadata only, future artifact-verification gates.
- This checkpoint and append-only registry update. No arm-specific documents exist on the frozen base.

### Verification actually executed before common commit

- Initial branch/HEAD/status/branches/worktree list inspected; requested sibling paths and branch names not occupied. No remote commit assumed newer than local state.
- Closure checksum inventory: 17 entries verified; revalidation inventory: 10 verified. Original thesis digest `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3`; split digest `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` verified.
- Frozen TRAIN/VAL lists and speaker-disjoint reduced-TRAIN FIT/SELECT/CAL selectors computed solely from split metadata, not labels/audio. TRAIN 2,399; VAL 300; FIT 1,799; SELECT 300; CAL 300. TEST metadata used only for exclusion.
- Initial tracked-file hashes and outer/nested status/diffs saved at `/tmp/langfens-two-workspaces-audit/` (ephemeral audit aids, not durable artifacts). 640 tracked regular files fingerprinted, no model/data loaders run. Source/evaluator/config/archive contents unchanged except permitted registry additions; hashes are rechecked before commit and in final verification.
- Git inventory contains inherited prediction audit tables, dictionaries and paper PDFs (some >1 MiB); these already exist in history. No new dataset/audio/TextGrid/model/secret/large binary is staged or copied into Git by this process. Worktrees will check out existing tracked source/audit assets normally, not clone ignored datasets/checkpoints/credentials or nested checkout dirt. No publication/push authorized; inherited privacy issues remain.
- No weight/durable-store readback: E0/E1 identities are recorded from archived metadata, not freshly certified. Untracked ignored local weights are not loaded, copied or edited; Git-only verification cannot certify remote artifact availability.
- Executed source/input/archive verification: 57 hashes match; 37 staged files pass the explicit documentary allowlist, regular-file/no-symlink and <100,000-byte checks. New shared Markdown contains no Markdown-link targets; registry relative links are checked in final verification. Scoped `git diff --cached --check -- manifests/RQ1-WORKSPACE-PREP-01 docs/process-checkpoints/INDEX.md` is clean.
- Whole staged `git diff --cached --check` reports 12 pre-existing Markdown hard-line-break trailing-space findings in three original closure reports (B/C/E). Their checksummed historical bytes are intentionally preserved, not repaired or called clean. This is a documented whitespace exception, not a source/test failure. No unexecuted unit/model tests are PASS.

### Exact safe continuation / stop

1. Review/stage only explicit research-closure/revalidation and shared documentary paths plus index/parent checkpoint. Verify historical bytes, nested diff and protected source hashes.
2. Commit shared preparation on `prep/rq1-shared-base`; resolve `git rev-parse HEAD` as the **frozen common base SHA** and contract SHA. Record that exact SHA in both arm manifests/checkpoints after creation; this avoids a self-referential commit hash. The later coordinator verification commit may advance this preparation branch, but MUST NOT change the frozen base used by the arms.
3. Verify branches/paths still absent, then `git worktree add -b prep/rq1-p-score ../langfens-rq1-p-score <frozen-base>` and equivalent `prep/rq1-p-sup`. No `--force`, branch reset, copying dirty nested checkout or initializing submodules.
4. Prepare arm-only docs/checkpoint/registry in each respective worktree; both use exact same contract bytes. Commit docs only. Verify merge-base, both contract hashes, disjoint arm documents and unchanged source/E0/E1 artifacts. Append cross-workspace verification to coordinator branch and registry with exact base/tips.
5. **STOP.** Setup authority does not authorize scorer/model/trainer implementation, tests that instantiate models, any fitting/inference/backward/training/GPU/remote staging or sealed TEST access. Safe next step is document review plus separate bounded implementation authorization, missing-proposal disposition, prospective effect margins and P-SUP objective-policy amendment.

Retained references: E0 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`; E1 fixed `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24` (archives only). DIAG-02B PARTIAL and E1 rollout NO-GO unchanged.

## Append-only completion record

**SETUP_COMPLETE_WITH_ACCOUNTED_SOURCE_LIMITATIONS; EXPERIMENTS_NOT_AUTHORIZED.** Frozen common base is `0d268ff8c444e16fd76aca2f4ad6ece966908edc`; shared contract SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Both isolated arm documentation commits are direct children: P-SCORE `de1db5cc2f0b4ed8b750ccc6120a314679dc3b6f`, P-SUP `d6ef18443752b0add3f3dfcfff42f7fa7be967c1`. Exact paths, evidence, unresolved source/objective/implementation gates, one corrected audit-predicate failure, preserved inherited whitespace and safe commands are in [WORKSPACE_VERIFICATION.md](WORKSPACE_VERIFICATION.md) and [JSON receipt](WORKSPACE_VERIFICATION.json). Original nested dirt remains unchanged. No implementation/tests/inference/fitting/backward/training/GPU/deployment/TEST access. Current successful/skipped updates remain 0/0; no model artifact. STOP after documentation. This coordinator completion commit records receipts only and does not change either arm's frozen base or authorize execution.
