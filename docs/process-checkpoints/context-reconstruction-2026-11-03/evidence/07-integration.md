# SEAM 7 — Cross-repository dependencies, shared objectives and integration history

## Scope and observation boundary

**Observed:** 2026-11-03 23:42 +07:00. Repository refs and worktree status below were read at that time; refs may have moved since. This is a read-only source reconstruction, not a claim that the system was started, deployed, or exercised. No tests, services, model loading/inference, training, remote API calls, or credential inspection were performed. Confidence labels: **VERIFIED** = directly observed in primary files/repository metadata; **CORROBORATED** = consistent across independent sources; **INFERRED** = interpretation; **UNKNOWN** = not established.

## Source inventory and observed refs

| Source | Observed identity and state | Evidence and limits |
|---|---|---|
| Speech research repo | `/home/khoa/Projects/KLTN/Dataset`, branch `prep/rq1-shared-base`, HEAD `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`; dirty: modified checkpoint index, modified nested `repo`, untracked context-reconstruction directory. | **VERIFIED**, `git rev-parse HEAD`, `git status --short --branch`. Read root `AGENTS.md` and `docs/process-checkpoints/INDEX.md`; index records substantial RQ1 history and current experiment holds. |
| Nested speech clone | `/home/khoa/Projects/KLTN/Dataset/repo`, HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`, remote `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git`. | **VERIFIED**. This is a second checkout/clone of the speech research project, not evidence of a separate product. Its HEAD differs from parent HEAD. |
| Backend repo | `/home/khoa/Projects/langfens/Project_Langfens_Microservice`, branch `generation-harness`, HEAD `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`; extensive modified and untracked worktree state observed. | **VERIFIED**. Root `AGENTS.md` describes an IELTS microservice platform; `AppHost/AGENTS.md` covers service graph and gateway endpoint configuration. Do not attribute local uncommitted changes to HEAD. |
| Frontend repo | `/home/khoa/Projects/langfens/langfens-fe-app`, branch `autonomous-generation`, HEAD `2d9fe94a54e65680eca56aafbe40e691018113da`; dirty modified and untracked files observed. | **VERIFIED**. No frontend `AGENTS.md` found in repository discovery; `README.md` is still mostly create-next-app boilerplate. `ARCHITECTURE.md` is the useful application map. |
| Recovery and worktree footprint | `/home/khoa/Projects/langfens` contains `recovery-wt-A`, `-A2`, `-B`, `-B2`, `-C`, `-M`, `worktrees/Project_Langfens_Microservice`, and `worktrees/langfens-reading-sdd-20261029`, plus primary repos. `/home/khoa/Projects/KLTN` also has `diag02b-recovery-worktree`, `diag02b-recovery-backups`, and separate P-SCORE/P-SUP speech worktrees. | **VERIFIED** directory discovery. Recovery names alone do not establish their purpose, validity, or active role. |
| GitHub metadata | Frontend origin is `https://github.com/julynguyen247/langfens-fe-app.git`; backend remote metadata was not needed for this seam. | **VERIFIED** from frontend `git remote -v`. No GitHub API/`gh` metadata was queried; PRs, issues, releases, and remote refs remain **UNKNOWN**. |

## Intent and architecture reconstruction

1. **Overall product intent — IELTS preparation.** Backend `README.md` describes an IELTS-preparation platform combining exam practice, automated writing/speaking grading, vocabulary retention, and microservices. Backend `AGENTS.md` independently maps exam/attempt, writing/speaking/AI, vocabulary, course, and gamification services. **CORROBORATED** as product intent, not proof of production operation.
2. **Frontend’s role — learner/admin application.** Frontend `ARCHITECTURE.md` places routes in `src/app`, backend operations in `src/services`, shared Axios clients/auth in `src/utils/api.customize.ts`, and practice screens under `src/app/do-test/...`. Its API client maps distinct gateway prefixes, including `/api-speaking`, `/api-ai`, and `/api-generation` (`src/utils/api.customize.ts:20-38`). **VERIFIED** as source architecture.
3. **Backend’s role — service platform and generation harness.** `AppHost/Program.cs:220-226` injects the AI service URL into writing and speaking and adds `generation-service` from `../services/generation-harness` on port 5050. Gateway configuration routes `/api-generation/{**catch-all}` and removes that prefix (`gateway/api-gateway/appsettings.json:472-476`). Backend root `AGENTS.md` says frontend is a separate checkout. **VERIFIED** static wiring; runtime reachability not tested.
4. **Autonomous IELTS generation — direct frontend/backend integration exists in source.** Frontend has an admin generation screen at `src/app/admin/generate/page.tsx` and client in `src/app/admin/generate/generationApi.ts`. The client calls `/config`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/result`, `/jobs/{id}/cancel`, `/jobs/{id}/artifact`, and an authenticated SSE `/events` stream (`generationApi.ts:18-101`). It supports only Reading in its offline fallback and presents configured provider defaults there (`:25-37`). **VERIFIED** that a concrete API contract is coded on both sides; **UNKNOWN** whether the current uncommitted frontend/backend changes work together in a live deployment.
5. **Harness scope is IELTS Reading, not RQ2 test generation.** `services/generation-harness/AGENTS.md` calls it a generation workflow harness and constrains IELTS domain assets to `.generation/`; its implementation and benchmark material name IELTS Reading, examiner and semantic review. `AppHost/Program.cs` wires it as `generation-service`; backend API routes expose generation jobs. **CORROBORATED** that this is exam-content generation. No inspected evidence establishes that it is the thesis’s RQ2 multi-agent *automated software-test generation* project.
6. **Speaking / grading / AI path.** Backend README describes Qwen2.5 LoRA grading, faster-whisper transcription, and a Wav2Vec2 pronunciation scorer; `services/ai-service/app/config.py:82` names `/app/models/pronunciation-scorer-v2/best_checkpoint.pt`; `services/ai-service/app/services/pronunciation_scorer.py:18-30,64-82` defines and loads that scorer/checkpoint. `AppHost/Program.cs:220-222` connects speaking and writing services to the AI service. **VERIFIED** that a pronunciation model path and loader exist in backend source. This model is not established as the speech research checkpoint.
7. **Speech RQ1 checkpoint is not shown wired into the backend.** Speech repository history/index describes a frozen `facebook/wav2vec2-base-960h` two-head CTC+binary MDD research model, its checkpointing and validation work. Backend source instead names a separate pronunciation scorer checkpoint path, and its scorer’s default backbone is `facebook/wav2vec2-base` (`pronunciation_scorer.py:18,77`). No inspected manifest, mount, config, artifact URI, code import, or API contract connects the RQ1 research checkpoint to backend inference. **VERIFIED** for the inspected paths; **INFERRED** that these represent distinct model lines; repository-wide absence cannot be proven from this bounded search.
8. **Historical Qwen fine-tuning is backend product work, not proof of speech integration.** Backend README and `models/qwen25-lora/README.md` describe a Qwen2.5-1.5B-Instruct LoRA artifact; local model tree contains `checkpoint-600` and `checkpoint-766`. This is text-generation/grading lineage and does not establish that the speech RQ1 checkpoint is consumed by speaking APIs.
9. **RQ2 distinction.** Speech root `AGENTS.md` explicitly reserves RQ2 exclusively for Multi-Agent Automated Test Generation and assigns SpeechOcean762/pedagogical pronunciation feedback to RQ1. Backend generation-harness is explicitly IELTS Reading content generation. **VERIFIED** as a documented scope boundary. The repositories’ shared “multi-agent” terminology is insufficient to infer shared implementation or research objective.

## Dated event table

Event timestamps are Asia/Ho_Chi_Minh (+07:00) where the Git metadata supplied a timezone-aware timestamp. A commit/document timestamp is not necessarily the time of execution; claimed execution remains source-attributed unless independently verified here.

| Date (+07) | Goal / decision and rationale | Action, result and limitation | Current validity |
|---|---|---|---|
| 2026-10-20 | Frontend landing-page iteration. | Frontend history includes IELTS/academic listening-stream changes and UI fixes. Git history verifies commits, not user adoption or production release. | Historical UI lineage; unrelated to model integration. |
| 2026-10-29 18:26:26 | Frontend source-of-truth update. | Current frontend HEAD commit subject is `update source of truth for FE`; branch `autonomous-generation`. Its local dirty state includes admin generation and exam contract changes, which are not part of that commit unless committed later. | HEAD remains observed ref; local worktree is distinct and unreviewed here. |
| 2026-10-31 18:14:07 | Backend generation-harness branch state. | Backend HEAD `2ba12e5` is on `generation-harness`; extensive subsequent worktree dirt includes harness runtime/server, reading workflow, benchmarks, gateway/AppHost changes, and test fixtures. | Static HEAD and local changes observed; their intent/results cannot be merged into a single committed state. |
| 2026-11-01 (runtime artifact date appears in paths) | IELTS Reading harness benchmark records. | Backend benchmark scripts reference runtime/session artifacts and trial metrics. Files/scripts demonstrate a recorded benchmark workflow; this review did not replay sessions or validate metric calculations. | Historical artifacts, not verified current service behavior. |
| 2026-11-02 (date in checkpoint index) | Speech RQ1 diagnosis and model-experiment status correction. | Speech checkpoint index says DIAG-02B is PARTIAL, archive-parity failures remain, and causal-training GO is superseded; it separately records P-SCORE benchmark results and P-SUP staging awaiting operator action. Those are **index-recorded claims**, not rerun/validated in this seam. | Scope boundaries/hold are relevant; do not interpret historical staging or metrics as current live state. |
| 2026-11-03 23:42:46 | Cross-repository forensic snapshot. | Read-only refs and dirty status captured; no tests/services/API calls, no edits. | Exact snapshot; refs can move. |

## Integration dependency map

```text
Learner/admin frontend
  ├── /api-auth, /api-exams, /api-attempts, /api-speaking, /api-ai, ...
  │       └── API gateway → corresponding backend microservices
  │                              └── speaking/writing → AI service
  │                                    ├── Qwen/LLM grading (documented product path)
  │                                    ├── faster-whisper (documented product path)
  │                                    └── pronunciation scorer checkpoint path
  └── /api-generation → gateway YARP route → generation-service / generation-harness
                                               └── IELTS Reading generation jobs/SSE/results

Speech research RQ1 checkout and its research checkpoints
  └── No verified artifact/config/API edge to the backend pronunciation scorer.

RQ2 multi-agent automated software-test generation
  └── Scope asserted by speech-research AGENTS; no verified implementation edge to
      IELTS Reading generation-harness.
```

## Contradictions, ambiguity, and confidence ledger

| Claim | Evidence and assessment |
|---|---|
| “Both repositories” means two clones of one repository. | **False for the named primary repos:** backend and frontend have distinct remotes, separate git roots, and different functions. The nested `Dataset/repo` is the second clone of the speech research project. **VERIFIED.** |
| “LANGFENS” refers to only the backend. | **Ambiguous.** Backend README frames the product; frontend is a separate associated checkout and supplies the user-facing application. The phrase can refer to the product ecosystem, not a single repository. **INFERRED** from repository roles. |
| “Autonomous generation” is RQ2. | **Contradicted by available scope evidence.** Frontend/backend generation is IELTS exam Reading content; speech AGENTS reserves RQ2 for multi-agent automated test generation. Similar agent terminology does not establish identity. **CORROBORATED.** |
| Backend speaking scorer is the speech RQ1 model. | **Not established and likely a false equivalence.** Backend path, class and default backbone are separately named; no observed artifact/config wiring identifies the speech two-head checkpoint. **VERIFIED** as to separate source declarations; actual deployment assets **UNKNOWN**. |
| Backend README’s Qwen and scorer statements describe current operation. | README describes intended/reported architecture and historical results. Local files and model paths corroborate artifacts/source, but this review did not start services, inspect runtime mounts, or validate a grading request. Treat operation as **UNKNOWN**. |
| Frontend generation page and backend harness are integrated. | Concrete path and endpoint correspondence, plus gateway/AppHost wiring, make source integration **VERIFIED**. Live compatibility, auth behavior, SSE, deployment config and success are **UNKNOWN**. |
| Branch HEAD captures the latest implementation. | Not safe: both application repos have substantial dirty files. Branch refs identify committed state only. **VERIFIED** dirt means uncommitted modifications must be preserved and reviewed independently. |

## Safe continuation and residual unknowns

- **Preserve all dirty state.** Before any later investigation, record fresh `git status --short --branch` for all three roots and nested speech clone. Do not clean, reset, checkout, stage, or infer that dirty contents are part of the observed commit.
- **Exact safe restart:** read the relevant repository guidance first; capture HEAD/branch/status and observation time; then inspect only static source/docs/metadata needed to resolve a named question. For integration confirmation, compare frontend request/response types with backend route handlers and gateway mapping at the same explicitly pinned commit/worktree state. Do not invoke the services or generate a job absent separate authorization.
- **Resolve model identity statically:** compare backend checkpoint manifest/config and mount/deployment declarations to the speech checkpoint provenance, architecture and artifact digest. Do not load weights. Until a literal artifact/config/code edge is found, report no speech-RQ1 deployment integration.
- **Remote metadata gap:** GitHub PR/issues/releases and frontend remote branch data were not queried; `gh` authentication state was not checked. These remain **UNKNOWN**, not evidence of no PRs or issues.
- **User-intent ambiguity:** available sources show an IELTS product ecosystem and a separate speech-research thesis repository. They do not establish whether “both repositories” means frontend+backend, backend+speech, or the broader product plus research work. Keep those meanings separate.
- **No execution claim:** no test result, service readiness, successful generation, live speaking grade, production state, or model deployment was established in this review.