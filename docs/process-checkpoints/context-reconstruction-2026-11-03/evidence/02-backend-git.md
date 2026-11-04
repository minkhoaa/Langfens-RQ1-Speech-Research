# Backend Git and implementation archaeology

**Observed:** 2026-11-03 23:42–23:43 +07:00 (Asia/Ho_Chi_Minh).  
**Conclusion:** The backend repository contains two materially different strands: the older .NET/Python IELTS-learning services and a newer native Node.js Reading-generation harness. The current `generation-harness` HEAD includes the harness but its working tree has substantial, uncommitted changes. Those changes are evidence of local work, not committed or verified releases.

Confidence labels: **VERIFIED** = directly observed in repository or GitHub API; **CORROBORATED** = consistent primary sources; **INFERRED** = interpretation; **UNKNOWN** = not established.

## 1. Scope and source inventory

- **VERIFIED — Backend checkout:** `/home/khoa/Projects/langfens/Project_Langfens_Microservice`, branch `generation-harness`, HEAD `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48` (`update harness`; authored/committed 2026-10-31 18:14:07 +07:00).
- **VERIFIED — GitHub origin:** `https://github.com/minkhoaa/Project_Langfens_Microservice.git`. GitHub API reports public, not archived, default branch `master`; API metadata `pushed_at=2026-10-31T11:14:11Z`.
- **VERIFIED — Local/remote branch inventory:** Local branches include `master`, `autonomous-generation`, `generation-harness`, `baseline-pre-rg-pipeline`, `refactor/langfens-harness`, `refactor/sync-dotest-admin-renderer`, `backup/refactor-sync-dotest-admin-renderer-before-autogen`, `reading/extension-sdd-20261029`, and five `refactor/recovery-*` branches. Remote-tracking refs include `origin/master`, `origin/autonomous-generation`, `origin/generation-harness`, `origin/refactor/phase15-v2rate-recovery`, `origin/refactor/recovery-integrate-g15`, and `origin/refactor/sync-dotest-admin-renderer`. There are **0 tags**; `git rev-list --all --count` observed **1,058** commits reachable from local refs.
- **VERIFIED — Worktrees:** Main checkout plus worktrees at `reading/extension-sdd-20261029` (HEAD `93d585a`), `recovery-wt-A` (`f4cfbb2`), `recovery-wt-A2` (`6eeb014`), `recovery-wt-B` (`04fdbd7`), `recovery-wt-B2` (`88f23fe`), `recovery-wt-C` (`0622a05`), and `recovery-wt-M` (`e66f74a`). Also detached worktrees at `/home/khoa/.cache/langfens-autonomous-resolution/postgres-recovered` (`1a27aa4`) and `/tmp/langfens-backend-blockers` (`1a98ea0`); the latter is marked prunable because its gitdir target no longer exists. This is inventory only: no worktree was changed.
- **VERIFIED — Dirty tree:** `git status --porcelain` showed **34 changed/untracked path entries** (15 tracked paths with edits/deletion, plus 19 untracked paths counted from the full status listing); `git diff --stat` reported 207 insertions and 450 deletions in tracked files. Untracked files are not represented by that diff stat. There are additional ignored/runtime files; their contents were not inspected.
- **VERIFIED — Scope guides:** Read repository `AGENTS.md`, `services/generation-harness/AGENTS.md`, and the speech repository’s global checkpoint index `docs/process-checkpoints/INDEX.md`. Backend guide maps .NET services, Python AI, Aspire, and gateway; generation-harness guide describes the native Node architecture and offline-test constraints. The speech index records broader ecosystem phases and explicitly distinguishes its research scope; it is not backend execution evidence.
- **VERIFIED — Relevant source inspected:** backend `README.md`; `services/generation-harness/src/server/server.mjs`, `src/server/adapter.mjs`, generation-harness README and source tree; selected diffs, `AppHost/Program.cs`, YARP appsettings, Git log/reflog, branch/worktree refs, and `.github/workflows/ci-cd.yaml` presence.
- **VERIFIED — Existing service footprint:** `services/` includes `ai-service`, `attempt-service`, `auth-service`, `course-service`, `dictionary-service`, `email-service`, `exam-service`, `gamification-service`, `generation-harness`, legacy `generation-service`, `speaking-service`, `vocabulary-service`, and `writing-service`.
- **UNKNOWN — Broader evidence not accessed:** Local secrets/config credentials were not read. No services were started; no tests, builds, provider calls, database access, or jobs were run. CI workflow contents/status were not validated. GitHub Actions API read timed out after receiving a large invalid/truncated response; no CI conclusion is drawn. No remote ref updates/fetches were performed. GitHub PR API returned 14 closed PRs (#1–#14), releases API returned zero. Issues endpoint returned the same 14 PR-associated entries, not evidence of a separate issue backlog. API pagination beyond 100 was not explored.

## 2. Dated event and architecture evolution

| Event time (+07:00 where known) | Goal / decision and rationale | Action / result and limitation | Current validity |
|---|---|---|---|
| 2026-10-01 13:44:56 | **VERIFIED:** Begin refactor toward autonomous exam generation and phase 07 evaluators/repairs (`7a4d0e7`). | Generation-loop architecture appears in backend history. | Historical; later source underwent recovery/refactor. |
| 2026-10-04 11:51:29 | **VERIFIED:** Implement generation engine v2, G07–G09 (`1ea298a`). | Earlier generation-service strand, not the current native Reading harness. | Historical; paths subsequently changed/deleted in later refactors. |
| 2026-10-12 14:48–23:57 | **VERIFIED:** Phase-15 remediation and recovery branches address valid-Draft rate, benchmark integrity, acceptance-path commits, test prerequisites and product regressions (`04fdbd7`, `0622a05`, `88f23fe`, `6eeb014`, `e66f74a`). | Multiple recovery worktrees/branches survive locally; branch tips are divergent snapshots, not a single linear release. | Historical; inspect each recovery tree before resuming old work. |
| 2026-10-14 | **VERIFIED:** Remove obsolete Reading V2 sprint artifacts and consolidate documentation. | Several commits remove planning/report artifacts; source history remains. | The absence of old docs does not mean corresponding behavior absent. |
| 2026-10-23–26 | **VERIFIED:** Intensify Reading content correctness, passage/blueprint alignment, claim/evidence provenance, bounded repair, semantic contradiction detection, callback recovery, durable operation lineage and no-paid-redispatch behavior. | Commits include `fcac168`, `8e26132`, `5455d69`, `63886d8`, `445fee1`, `66c1af8`; commit titles/docs mark release gates and unresolved Section 1 issues. | Evidence of iterative development; not proof of production deployment or quality acceptance. |
| 2026-10-26 22:23 | **VERIFIED:** Scope quality-first 13-question execution to strict 1M-token budget (`3e1dd70`). | Exists on `autonomous-generation` history; `generation-harness` later diverged as a standalone harness. | Historical limit; confirm current uncommitted configuration before any execution. |
| 2026-10-28 22:53:32 | **VERIFIED:** Establish native generation harness (`85d87ea`). | Harness commits introduce independent Node runtime and package layout. | Foundation for later Generation service integration. |
| 2026-10-29 00:04:35 | **VERIFIED:** Add native IELTS Reading vertical slice (`0feaac0`). | Reading-specific tools, templates, contract and extension are in the harness tree. | Committed at current backend ancestor. |
| 2026-10-29 08:17:57 | **VERIFIED:** Add deterministic full-paper verification (`0d7e60d`). | Passive/test artifacts and deterministic verification accompany Reading implementation. | Does not prove generated papers are valid in live paid-provider conditions. |
| 2026-10-30 00:49–03:12 | **VERIFIED:** Update harness source of truth and create SDD reading worktree (`1a98ea0`, `93d585a`). | SDD worktree remains at `93d585a`; later main branch commits advanced. | Worktree is an earlier snapshot; do not mistake it for current HEAD. |
| 2026-10-30 11:46–16:55 | **VERIFIED:** Main branch receives harness updates (`571321c`, `1f14aee`, `4329c0b`). | `generation-harness` reflog records commits and matching pushes. | Superseded by current tip. |
| 2026-10-31 18:14:07 | **VERIFIED:** Commit `2ba12e5` “update harness”; push reflog records same tip at 18:14:11. | Current branch equals `origin/generation-harness`; main branch is reported locally 47 commits behind its upstream tracking ref. | Latest directly observed committed backend state. |
| 2026-11-03 23:42–23:43 | **VERIFIED:** Forensic read-only inspection. | Current checkout is dirty; GitHub API confirms public repo/default branch; no execution or mutation performed. | Current observation; refs can move after inspection. |

## 3. Architecture and implementation seams

### Native generation harness and HTTP integration

- **VERIFIED:** Backend `README.md` describes a .NET 10/Aspire/YARP application with per-service data ownership and HTTP/gRPC/MassTransit boundaries. In parallel, `services/generation-harness/README.md` describes an independent Node.js harness: native Core, Subagents, provider APIs, TUI/RPC, and IELTS specifications under `.generation/`.
- **VERIFIED:** Current committed `generation-harness` tree includes `.generation/reading/` domain instructions, tools, contract, rules, grading/projection and tests; `packages/` provides the runtime/SDK. Harness rules declare Core/SDK/Subagents immutable for IELTS work and domain artifacts confined to `.generation/`.
- **VERIFIED:** Current *dirty* main checkout adds `scripts/server.mjs`, `src/server/server.mjs`, `src/server/adapter.mjs`, and related tests; these files are untracked, so they are **not** part of HEAD `2ba12e5`. `server.mjs` exposes health/config, job creation/listing, SSE, cancel/result/artifact routes; defaults are Reading domain and 13 questions. `adapter.mjs` launches a subprocess and projects a student-safe exam separately from admin inspection. The committed/uncommitted distinction is material.
- **VERIFIED:** Dirty `AppHost/Program.cs` registers a Node app running `scripts/server.mjs` on port 5050 and wires its endpoint to YARP; dirty `gateway/api-gateway/appsettings.json` adds `/api-generation/{**catch-all}` and a 45-minute cluster timeout. These wiring changes are **uncommitted**.
- **VERIFIED:** Dirty `package.json` adds server start/test entries, and dirty `scripts/runtime.mjs` changes offline environment inheritance and Reading prompt injection. These changes likewise are not committed.
- **VERIFIED:** Server authentication helper `verifyAdminAuth` in untracked `src/server/server.mjs:13–74` allows bypass when `GENERATION_HARNESS_AUTH_OPTIONAL=1` or `NODE_ENV=test`; otherwise checks JWT shape, expiration, Admin role, and verifies HMAC only if a signing key is configured. **INFERRED security concern:** a production environment with no signing-key variable would accept a syntactically valid, unexpired token carrying an Admin role without cryptographic signature validation. Deployment configuration and threat exposure are unknown; do not conclude exploitable production deployment from source alone.
- **VERIFIED:** `server.mjs:83–87` reflects request Origin (or `*`) while allowing credentials; actual browser behavior and gateway policy were not exercised.
- **VERIFIED:** Adapter has local run directory `.runtime/reading/runs`, process lifecycle and completion evaluation. Source config/test files and runtime artifacts exist; no run was initiated in this investigation.
- **UNKNOWN:** Whether the untracked server and service wiring are integrated, tested, or deployed elsewhere. Main branch source is mutable and local dirty changes may reflect active work.

### Older generation-service strand and Git archaeology

- **VERIFIED:** Legacy `services/generation-service/app` still exists in the checkout, while current harness is under `services/generation-harness`. Git history contains generation-service commits spanning at least 2026-10-01 through October recovery, with major refactor cleanup and renamed/new harness paths. `git log --all` reports early `7a4d0e7` and October recovery work; current main lineage later adds the harness.
- **VERIFIED:** Local branches/recovery worktrees preserve recovery artifacts and product/test fixes. These are valuable forensic states, not evidence that the changes were merged into current `generation-harness`.
- **VERIFIED:** Current dirty patch deletes tracked `FIXES_PROPOSAL.md`; the tracked deletion is local and absent from committed HEAD unless independently committed elsewhere.
- **UNKNOWN:** Exhaustive rename/deletion mapping for every historical generation-service path was not completed. `git log --follow` and per-commit tree comparisons remain safe follow-up work.

### Production IELTS APIs, grading, speaking, retrieval and storage

- **VERIFIED:** Existing backend source has independent IELTS learning features beyond generation: `services/ai-service/app/routers/speaking.py` contains roleplay, speech/transcript and memory-retrieval routes; `app/prompts/speaking_grade.py` defines speaking grading; writing prompts/services include IELTS grading and exemplar retrieval. Search also found Qdrant-backed speaking-memory helpers and writing/grammar RAG references.
- **VERIFIED:** AI service is Python and described in root map as grading/RAG/ASR; speaking and writing services are separate .NET services. `services/ai-service` also contains test/data/evaluation source.
- **VERIFIED:** These constitute implementation in repository source, not proof of operational production endpoints or successful external model/provider runs.
- **UNKNOWN:** Live endpoint deployment, provider configuration, model availability, Qdrant contents, audio storage retention and service health were not checked. No credentials/config values were read.

## 4. GitHub and CI evidence

- **VERIFIED:** GitHub repository is **public**, default branch `master`, not archived. URL: `https://github.com/minkhoaa/Project_Langfens_Microservice`.
- **VERIFIED:** GitHub branches API (`branches?per_page=100`) returned six branch tips: `master` `2a41286e623d3c23c07c9620a3335361b7ae5c68`; `autonomous-generation` `029978f3dacc10eb4ce87953cc941f770ace02ed`; `generation-harness` `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`; `refactor/phase15-v2rate-recovery` `f255b8b14afbbf7e5856899165c318139e48a797`; `refactor/recovery-integrate-g15` `641721df523b76fbe659cf26e078f4494519a8f3`; `refactor/sync-dotest-admin-renderer` `04fdbd714eb4d9c71b6cf2d0a7985dea83286e4d`.
- **VERIFIED:** GitHub PR API returned 14 closed PRs, #1–#14, including #13 (“feat(reading-generation): autonomous exam generation loop engine v2 & admin dotest synchronization”) and #14 (“Autonomous generation”). Closed status alone does not establish merge contents or runtime verification.
- **VERIFIED:** Releases API returned 0 releases at observation.
- **UNKNOWN:** PR merge commits, review conclusions, and checks were not individually inspected; Actions run query timed out/truncated and was unparsable. CI success/failure is not asserted.
- **VERIFIED:** Repository root `AGENTS.md` explicitly cautions that CI pushes/deploys `master` without tests; source instructions are policy claims, not actual deployed-state evidence.
- **INFERRED:** Public visibility is a material privacy concern if the stated policy of private IELTS assets/artifacts is intended to include committed generation artifacts; visibility verified, exposure impact was not audited.

## 5. Latest state, historical commitments, and contradictions

- **VERIFIED — latest local committed state:** `generation-harness` at `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`, identical to `origin/generation-harness` at the observed moment.
- **VERIFIED — latest working-tree state:** 34 reported dirty paths, including substantive server/API wiring, Reading generation instructions/tests/evidence files, AppHost, gateway and runtime changes. Keep all dirt intact; it was not evaluated or executed.
- **VERIFIED — remote master mismatch:** Local `master` at `af83797` reports `[origin/master: behind 47]`; GitHub branches API independently observed remote master `2a41286…`, a different tip. Do not infer ancestry or synchronize without a separate authorized review.
- **VERIFIED — historical CI policy conflict:** Repository instructions say CI deploys/pushes master without tests; user request forbids remote mutation, and no deployment action occurred. This is a documented policy hazard, not an observed deployment event.
- **VERIFIED — architectural tension:** Generation-harness rules say no custom runtime/coordinator and only public ExtensionAPI for domain behavior; dirty `src/server/adapter.mjs` directly orchestrates a spawned harness process and HTTP job lifecycle. This may be an API adapter rather than a replacement agent loop, but requires an explicit design boundary review.
- **VERIFIED — auth ambiguity:** Current untracked auth code treats signing verification as optional when no signing secret is set; deployment injection is not established. This weakens any assumption that “Admin role check” necessarily means verified signed identity.
- **VERIFIED — product readiness language:** Historical commit titles include “live MiniMax smoke” and phase completion, while later commits mark `NOT_RELEASE_READY`, retain blocked Section 1 gates, and document offline gates. These are different stages and selected worktree histories; don’t summarize them as a clean production pass.
- **VERIFIED — no live execution in this investigation:** No tests/builds were run; no LLM/API/provider/job/service was called. Historical test/benchmark claims were not re-executed and are not promoted to current PASS.
- **UNKNOWN — deployment state:** No evidence in this read-only inspection establishes whether current `generation-harness`, legacy generation-service, speaking/writing AI features, or proposed server wiring are deployed and serving traffic.

## 6. Claim/evidence registry

| Claim | Evidence | Confidence |
|---|---|---|
| Backend currently has a native Reading-generation harness | Committed tree at `2ba12e5`, harness README/source/tests | **VERIFIED** |
| Current local checkout also has an HTTP harness service integration | Dirty untracked server/adapter plus dirty AppHost/YARP diffs | **VERIFIED** as local source only |
| That HTTP service is part of current committed HEAD | `git status` shows the relevant server/source files untracked and wiring modified | **False; VERIFIED absent from HEAD** |
| Backend contains IELTS writing/speaking grading and retrieval source | AI service prompts/router and RAG/memory code | **VERIFIED** as source |
| Those APIs are live in production | No deployment/health/provider evidence collected | **UNKNOWN** |
| Remote repository is public and master is its default branch | GitHub repository API response | **VERIFIED** |
| CI currently passed | Actions API timed out/truncated; workflow not run | **UNKNOWN** |
| Current changes are tested | No tests/builds run; untracked test files not executed | **UNKNOWN** |
| No tests were run for this investigation | Commands were read-only Git/search/API metadata only | **VERIFIED** |
| Existing dirt was preserved | No write-capable action or Git mutation performed | **VERIFIED** |

## 7. Residual risks and exact safe restart

1. **Do not reset, clean, checkout, stash, or rebase** the backend checkout; its dirty contents are potentially valuable. Before any future edit, record a fresh `git status --short --branch`, current HEAD, branch tips, and worktree list.
2. Treat untracked `services/generation-harness/src/server/*`, `scripts/server.mjs`, the 1,425 untracked files counted under generation-harness, and dirty AppHost/YARP/config changes as active local source—not as committed code.
3. Preserve and separately inspect each recovery worktree, particularly the detached/prunable `/tmp/langfens-backend-blockers` worktree. Avoid removing/pruning it until its recovery data is understood.
4. Safe continuation is read-only: inspect `git diff` and `git diff --no-index /dev/null` for selected untracked files; use `git log --follow -- <path>` / `git log --all --name-status` to complete rename/deletion chronology; inspect PR #13/#14 and CI workflow metadata via bounded GitHub GET calls.
5. Before testing, deployment or paid model usage, require explicit authorization; isolate tests and provider configuration per `services/generation-harness/AGENTS.md`; do not infer production quality from offline fixtures or historical claims.
6. Before considering HTTP service deployment, independently review token signature requirements, CORS credential/origin policy, Gateway authentication propagation, durable job persistence/restart behavior, and whether the adapter conforms to harness extension/runtime boundaries.
7. No local files or refs were changed by this research. The requested output path is handled by the run’s output binding rather than a filesystem write.