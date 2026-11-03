# P-SCORE preparation checkpoint

State: **WORKSPACE_CREATED; DOCUMENTATION_READY; IMPLEMENTATION_PENDING; EXECUTION_NO_GO.** Continuation of RQ-RESEARCH-01. Operator authorized isolated setup/documentation only. No scoring implementation, synthetic tests, model/corpus inference, fitting, backward, training, GPU or remote staging. Training NOT_STARTED; 0 successful optimizer updates, 0 skipped updates/no attempts; no new model artifact.

## Identity and outputs

- Workspace `/home/khoa/Projects/KLTN/langfens-rq1-p-score`; branch `prep/rq1-p-score`.
- Exact frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`, created from verified local `b25a7595360294798d8d06b37592dd65dd4c9c33` plus preserved uncommitted research closure and new shared documentation. `git worktree add -b` used without force; no existing path/branch overwritten.
- Contract `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- `EXPERIMENT_SPEC.md`: frozen-emission S/SD primary contrast, self-aligned/segmental and greedy-DP/binary controls, stability/occupancy/graph/repeated-phone requirements and gates.
- `OUTPUT_SCHEMA_AND_TASK_PLAN.md`: runtime/offline schema, private bundle, pending implementation sequence and synthetic CPU no-model/no-training test matrix (NOT RUN).
- `PROVENANCE_MANIFEST.json`: base/source/input/checkpoint references, speaker selectors, artifact status and stop boundary. Checkpoint weights not copied/loaded/rehash-certified; identities from archives only.

## Verification actually executed

- Before arm edits: branch and HEAD exactly matched base; clean arm status; contract SHA256SUMS verified in both new worktrees.
- After design docs: same base/branch/contract identity, all **57** shared source/input/archive fingerprints match. Other arm's directory absent. `.env`, `datasets`, `checkpoints` absent in this worktree; nested repo not initialized/copied. No private corpus/model/credential inputs cloned from dirty main checkout.
- Only arm documentation/metadata and an additive registry entry are staged. Lightweight JSON/key/hash/relative-link/filesize/Git checks and final exact commit/merge-base evidence are recorded by the coordinator in `manifests/RQ1-WORKSPACE-PREP-01/WORKSPACE_VERIFICATION.md` on `prep/rq1-shared-base`; not inherited into this frozen branch unless explicitly reviewed later.
- One document-audit assertion initially treated staged new files as pre-existing tracked-file differences; it failed before either commit or any source/data change. Corrected predicate excludes new allowlisted arm documents and is rerun before commit. This failure is audit logic, not a scorer/model test result.
- Historical O/Z evaluator/model/config/archive bytes unchanged; no current unit/regression/model test or performance PASS claimed. Inherited historical closure Markdown whitespace is preserved; arm additions must pass scoped diff check.

## Blockers and exact safe restart

Missing latest distinct omission proposal (supply exact version/path or approve source limitation); unresolved reused-code license and graph/next-phone/occupancy registration; practical-effect/full-substitution noninferiority and false-alarm/coverage margins; synthetic implementation/tests; actual checkpoint/package/cache readback and parity qualification. E0's historical CAL acoustic-training overlap remains; two reused VAL speakers are exploratory, not confirmation. DIAG-02B PARTIAL and E1 rollout NO-GO unchanged.

Read-only resume commands:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
git status --short
git log -1 --format='%H %s'
git merge-base HEAD 0d268ff8c444e16fd76aca2f4ad6ece966908edc
(cd manifests/RQ1-WORKSPACE-PREP-01 && sha256sum -c CONTRACT_SHA256)
```

Read AGENTS, global index, shared addendum/contract/provenance and this checkpoint/spec/task plan. Check clean branch/expected base/source hashes, resolve pending registration, then ask operator for separate bounded scorer implementation plus synthetic CPU test authorization. **STOP before implementing, scoring or accessing real data.** No dependency on P-SUP. Last valid model references remain archived E0/E1; this process creates none.
