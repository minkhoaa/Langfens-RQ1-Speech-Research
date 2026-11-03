# P-SUP preparation checkpoint

State: **WORKSPACE_CREATED; DOCUMENTATION_READY; DISTINCT_OBJECTIVE_POLICY_AMENDMENT_AND_IMPLEMENTATION_PENDING; EXECUTION_NO_GO.** Continuation of RQ-RESEARCH-01. Operator authorized setup/documentation only, not model/trainer/label code changes. No label extraction, synthetic model tests, inference/fitting/backward/training/GPU/deployment. Training NOT_STARTED; 0 successful updates, 0 skipped updates/no attempts; no new model artifact.

## Identity and outputs

- Workspace `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`; branch `prep/rq1-p-sup`.
- Exact frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`, preserving newer local research closure/addenda rather than choosing an older remote. Fresh `git worktree add -b` without force; no existing branch/path replaced.
- Contract `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- `EXPERIMENT_SPEC.md`: compact natural-operation/existence alternatives, contemporaneous matched binary comparison, objective/initialization/schedule and fixed/best accounting. ONE treatment/capacity control still requires registration; no complex architecture selected or implicit AGENTS amendment.
- `LABEL_COLLATOR_AND_OUTPUT_DESIGN.md`: natural strict operation labels/masks, optional realized labels, blinded-loader separation, output schema and synthetic CPU no-model/no-training test plan (NOT RUN).
- `KAGGLE_T4_DEPLOYMENT_DESIGN.md`: private T4/FP16, matched states/update budget, permanent CLI/UI workflow, atomic durable per-epoch checkpoints and safe recovery (DESIGN ONLY).
- `PROVENANCE_MANIFEST.json`: exact base/shared hash, input selectors, archived weight references, future output identities and authorization limits.

## Verification actually executed

- Fresh worktree initially clean on exact common base; distinct expected branch and shared contract SHA verified in both worktrees.
- After docs: all **57** shared source/input/archive fingerprints unchanged; other arm directory absent. No `.env`, `datasets` or `checkpoints` in this worktree; nested repo not initialized/copied. No original ignored local datasets/weights/credentials or nested dirt copied.
- Only this arm's text documentation/metadata and an additive registry entry are staged. Final lightweight JSON/schema-reference/hash/link/size/Git checks, exact arm commit SHAs and merge-base are saved in coordinator `manifests/RQ1-WORKSPACE-PREP-01/WORKSPACE_VERIFICATION.md` on `prep/rq1-shared-base`, not automatically merged here.
- One coordinator document-audit assertion failed before arm commits because it incorrectly counted new staged arm files as pre-existing tracked-file changes. Predicate corrected to distinguish additions; no source/data mutation or test execution occurred. Rerun required before commit, not marked PASS from failed attempt.
- Model/trainer/collator/label/evaluator/config/E0/E1 archive bytes unchanged. Current test status NOT_RUN, not scientific PASS. No remote/API/quota/auth/secret verification attempted; no kernel created/pushed.

## Blockers and exact safe restart

Missing latest distinct omission proposal or explicit acceptance of limitation; explicit new-objective policy authorization; ONE compact parameterization and parameter-matched binary control; loss/mask/module initialization and schedule registration; prospective effect/noninferiority/false-alarm margins; subtype support and synthetic CPU tests; actual private package/weight/readback/storage/quota/T4/AMP and recovery readiness. Existing blinded-collator defect is documented but not fixed. Baseline parity limitations remain; no P-SCORE dependency.

Read-only resume commands:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-sup
git status --short
git log -1 --format='%H %s'
git merge-base HEAD 0d268ff8c444e16fd76aca2f4ad6ece966908edc
(cd manifests/RQ1-WORKSPACE-PREP-01 && sha256sum -c CONTRACT_SHA256)
```

Read AGENTS, global index, shared addendum/contract/provenance and this checkpoint/spec/label/deployment design. Confirm exact base, clean state and hashes. Request objective-policy amendment plus separate bounded label/collator implementation authorization first; no backward/model fitting locally even for a smoke test. Remote training/deployment needs its own later authorization and verified allocation. **STOP after documentation.** Archived E0/E1 remain last valid references; no model produced here. DIAG-02B PARTIAL and E1 rollout NO-GO retained.

---

## Append-only continuation — Luna implementation (2026-11-03)

The above documents the historical preparation-only state. At the explicit user's later implementation authorization, Luna resumed the intentionally paused partial source on branch `prep/rq1-p-sup` at `d6ef18443752b0add3f3dfcfff42f7fa7be967c1`; no prior work was discarded. Initial Git state, handoff and safety boundaries: [checkpoint 01](checkpoints/01-luna-handoff-and-initial-state.md). Additive prototype and validation: [checkpoint 02](checkpoints/02-implementation-and-validation.md).

Bounded isolated implementation is complete; the synthetic CPU forward-only suite passed **88 tests**; preflight explicitly reports EXPERIMENT_HOLD and no trainer. This is prototype code review GO only, not experiment/training/deployment GO. The preflight package contains no data/model/remote imports and was not pushed or run remotely. Full actual update counts remain NOT_STARTED, 0 successful/0 skipped/0 attempts, no artifact. Missing remote data allowlist adapter, per-epoch durable checkpoint/resume, authorized T4 trainer, scientific objective and execution approvals, missing-proposal/margin/license/support verification remain explicit blockers. No real data/labels/TEST contents, weights, backward, optimizer, GPU, API or remote service were accessed. The original historical state above is retained; use latest checkpoint for current state.
