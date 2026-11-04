# LANGFENS — Project Context & History Reconstruction

**Disposition: RECONSTRUCTED WITH ACCOUNTED COVERAGE GAPS — not a declaration of thesis completion, deployment readiness, or experimental PASS.**

Investigation began **2026-11-03 23:39:51 +07:00** and synthesis continued on **2026-11-04**, Asia/Ho_Chi_Minh. Current-state assertions apply to their individual observation timestamps, not indefinitely. Ten research subagents were used: eight concurrent discovery lanes, then two cross-source reviewers, using `openai-codex/gpt-6-luna` and `antigravity/gemini-3.8-flash`.

Only research documentation and evidence receipts were created. No project code, branches, datasets, remote settings, jobs, or experiment configurations were changed. No training, backward pass, project model inference, notebook execution, generation benchmark, service startup, database operation, or new functional test was run. The requested research-agent LLM calls are distinct from prohibited new project experiments.

## Reading conventions

- **VERIFIED:** directly inspected primary source, file identity, Git metadata, or remote response. For a historical result, this can verify *what the record contains*, not independently reproduce the experiment.
- **CORROBORATED:** consistent evidence across records; copied summaries are not automatically independent evidence.
- **INFERRED:** interpretation supported by evidence but not established directly.
- **UNKNOWN:** absent, inaccessible, unexamined, or insufficient evidence. This is not proof of nonexistence.

Paths below use these explicit roots:

- **S:** `/home/khoa/Projects/KLTN/Dataset` — speech research.
- **B:** `/home/khoa/Projects/langfens/Project_Langfens_Microservice` — backend and generation harness.
- **F:** `/home/khoa/Projects/langfens/langfens-fe-app` — associated frontend.
- **P:** `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md` — original graduation-thesis proposal.
- **E:** `S/docs/process-checkpoints/context-reconstruction-2026-11-03/` — this investigation's durable evidence bundle.

Source IDs **S01–S15** are resolved to exact references in §7. Raw child reports are investigative leads, not authority; identified errors are preserved and explicitly adjudicated in `E/COORDINATOR_VERIFICATION_NOTES.md`.

## 1. Executive project overview

**LANGFENS is an English/IELTS learning ecosystem, not just the current speech branch or generation harness.** Its product code predates the formal 2026 thesis proposal. The recovered ecosystem includes a .NET/Python backend, a Next.js frontend, a separate speech-research repository, a related IELTS data-export repository, remote research storage, and multiple generations of agent orchestration. **CORROBORATED** [S01–S04, S09–S10].

The proposal establishes two research axes:

1. **RQ1:** phoneme-level acoustic detection of targeted pronunciation errors, plus accurate, clear, error-appropriate pedagogical feedback.
2. **RQ2:** multi-agent generation, review, validation, and repair of **educational learning content/tests**, compared with a single LLM and sequential self-review, including quality/time/cost tradeoffs.

**RQ2 does not mean software-test generation.** The original Vietnamese research question and reading/grammar exercise examples settle the ambiguous English title. The backend IELTS Reading harness is aligned with RQ2; this does not establish completion of the proposal's comparative research. **VERIFIED intent** [S01; §8].

### Evidence-backed state in brief

| Workstream | What is established | What is not established |
|---|---|---|
| IELTS product/backend/frontend | Substantial source history since September 2025; exam/attempt/auth, writing/speaking AI, RAG, vocabulary and other services; concrete local generation API/UI wiring | Current production health, deployment identity, complete end-to-end acceptance, or completeness of dirty local edits |
| Autonomous generation / RQ2-aligned engineering | Native Node harness and Reading domain extension; recorded MiniMax trials, immutable exam revisions, structural checks and a local .NET fixture receipt | Publication-quality completion; OS-enforced blind-review isolation; formal A/B/C research campaign across representative reading and grammar requests |
| Speech / RQ1 | Frozen-backbone CTC+binary baseline, four-fold validation records, negative/interrupted candidate studies, explicit methodological corrections | Sealed-test/cross-corpus confirmation, clinically/pedagogically reliable diagnosis, feedback-quality/learner-benefit evidence, or verified backend integration of the RQ1 model |
| Latest speech candidates | P-SCORE reported omission gains with substitution noninferiority failure; P-SUP remote records now show both arms at 3,600 successful updates, 0 skips | A verified final P-SUP VAL comparison or superiority conclusion |
| Infrastructure/governance | Existing Drive access; Colab currently 0.00 compute units/no active sessions; private fold datasets and private C0/C1/cross-fold HF stores | Privacy consistency: baseline HF is currently public, and two P-SUP notebook API metadata responses report `is_private=false` despite private-staging documentation |

No single workstream is substituted for another. Historical implementation, recorded experiment completion, scientific acceptance, and current operational readiness are separate claims throughout this report.

## 2. Original vision and evolution of project goals

### 2.1 Product origins versus thesis origins

- **VERIFIED implementation chronology:** backend root commit `62072d506164eb2e7b7b4bac5c5022acd6fb38b5`, `init`, **2025-09-15 02:54:06 +07**; frontend root `b7403a454ab2cbc47015e6097a8e42225031f245`, **2025-09-16 13:23:29 +07**. Early backend history adds authentication/session rotation, Redis, RabbitMQ/email, gateway and exam APIs. These dates establish code existence, not the exact date the overall idea was conceived [S02].
- **CORROBORATED product motivation:** provide an integrated English/IELTS practice environment rather than disconnected grading scripts: exams and scoring, conversational speaking, writing/grammar support, learning history, vocabulary and progress [B/README.md; B/AGENTS.md; F/ARCHITECTURE.md].
- **VERIFIED proposal contents:** P carries the date **2026-08-18**, a planned execution window **2026-09-01–2027-01-01**, and the title *Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation*. A dated document/signature area is not independent verification of formal approval or signatures [S01].

The project was therefore not first created by the October speech sessions. Conversely, the 2025 product commits do not prove that the final thesis research questions already existed then.

### 2.2 Original research commitments

P, line 14, §§2.2–2.3, 3(b), 4(c/e/g), proposes:

- Separate **STT transcription** from **acoustic pronunciation-error detection**; a correct transcript is not proof of correct pronunciation.
- Prefer reference-text pronunciation tasks; investigate forced alignment/GOP or pretrained acoustic classifiers on labeled non-native speech. Do not promise exhaustive free-conversation diagnosis.
- Evaluate pronunciation feedback separately for correctness, clarity and appropriateness.
- Generate passage-grounded reading questions and grammar exercises with reference answers, using Coordinator/Generator/Reviewer/Validator roles, bounded repairs and answer-blinded independent solving.
- Compare **A: single LLM**, **B: sequential self-review**, **C: multi-agent**, initially approximately **25–40 requests per representative exercise group**, retaining failed generations in denominators and recording calls/tokens/time/cost.
- Treat Writing, Grammar, RAG and learning-history functions as supporting integration objectives, not replacements for the two principal research questions.

These are **VERIFIED historical intentions**, not reported completed experiments [S01].

### 2.3 How objectives narrowed and expanded

| Historical direction | Subsequent evolution | Current interpretation |
|---|---|---|
| Broad integrated learning platform | Speaking/writing APIs, local/cloud graders, RAG, exam delivery and frontend grew before the thesis research sprint | Existing product engineering remains a distinct stream; operational results need their own evidence |
| Classical GOP/forced alignment as a plausible speech baseline | Manifest-driven Wav2Vec2 MDD, initially three heads, then CTC+binary only; later return to segmentation-free GOP controls | Method selection evolved; avoiding forced gold timing was intentional, not proof that alignment problems were solved |
| A single Fold-1 model | Matched acoustic ablation, independent cross-fold baselines, omission diagnostics, controlled candidate studies | Validation evidence became more rigorous but remains limited by few/reused speakers and protected TEST |
| Prompt-/service-based question generation | Autonomous loop engines, recovery branches, then a native standalone Node harness with project-local Reading tools/roles | Earlier paths remain relevant history and some remain in source; the newest harness is not the whole product |
| Broad Pi/Subagents parity ambitions | Explicit user requests to restore a minimal native core, preserve contracts, then add bounded domain-level reliability and review | “Minimal core plus domain workflow” is a plausible reconciliation, not proof that every boundary is currently satisfied |
| Candidate research shortlist | Separate frozen-emission P-SCORE and supervised P-SUP pilots | No universal winning architecture was selected by literature/rubric scores; empirical tradeoffs must decide |

**Proposed future goals**—more exercise types, prosody/fluency research, adaptive learning, mobile experience—appear in P's future-work section. They must not be reported as current completed requirements or newly authorized work [S01].

## 3. Repository and workstream relationship map

### 3.1 Verified identities

| Repository/worktree | Branch / exact observed commit | Role and state |
|---|---|---|
| S — [Langfens-RQ1-Speech-Research](https://github.com/minkhoaa/Langfens-RQ1-Speech-Research) | `prep/rq1-shared-base` / `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6` | Research coordinator; GitHub default `main`; public. Pre-existing nested-checkout dirt; this investigation adds documentation only |
| B — [Project_Langfens_Microservice](https://github.com/minkhoaa/Project_Langfens_Microservice) | `generation-harness` / `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48` | IELTS backend, AI service and generation harness; default `master`; public; substantial tracked/untracked local work |
| F — [langfens-fe-app](https://github.com/julynguyen247/langfens-fe-app) | `autonomous-generation` / `2d9fe94a54e65680eca56aafbe40e691018113da` | Next.js learner/admin frontend; default `master`; public; dirty exam contracts/UI and untracked generation UI/tests |
| [langfens-ielts-dataset](https://github.com/minkhoaa/langfens-ielts-dataset) | remote `main` / `fcc18c97eee4b93a25ff464fe912399ce1c98e99` | Public related data-export repo, discovered remotely; description names essays, grammar pairs, speaking conversations, RAG/fine-tuning exports. Root metadata only reviewed; licensing/training linkage UNKNOWN |
| `S/repo` | detached / `baa4989243d5ff46b8b3bf5a3c280cdf2857382b` | Separate clone/checkout of the speech repo, **not another research project**; 14 modified script entries and two deleted reference PDFs preserved |
| `/home/khoa/Projects/KLTN/langfens-rq1-p-score` | `prep/rq1-p-score` / `a9f4947985651253cee980086a346056885b51c6` | Isolated frozen-emission scoring arm |
| `/home/khoa/Projects/KLTN/langfens-rq1-p-sup` | `prep/rq1-p-sup` / `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` | Isolated supervised-objective arm; current local and fetched remote train notebooks pin this SHA |
| `/home/khoa/Projects/KLTN/diag02b-recovery-worktree` | `review/rq1-diag-02b-safe-delivery-parity-recovery` / `00c684408f377300b84aacbec1483b8a2e812967` | Clean recovery/delivery worktree; cleanliness does not resolve scientific parity failures |
| `/home/khoa/Projects/langfens/worktrees/langfens-reading-sdd-20261029` | `reading/extension-sdd-20261029` / `93d585a5ab4e8bce75215d0b2099a36242b578d8` | Earlier isolated Reading SDD state, not current backend HEAD |

At discovery, the three main local repositories had **180 / 1,058 / 365 reachable commits**, **85 / 41 / 12 refs**, and **726 / 2,436 / 605 tracked paths**, respectively. No tags were found in the main local inventories. These are repository-object counts, not independent-event counts; stash roots and duplicate/cherry-picked histories are not extra project inceptions [S02].

GitHub GET inventories returned speech **44 branches / 3 PRs**, backend **6 / 14**, frontend **6 / 44**; no non-PR issues or releases returned. Speech PR #1 remains open; #2/#3 have merge timestamps. The main observed working-branch tips matched remote responses. This does not equate them to default branches or deployed versions [S03].

Additional backend recovery worktrees `recovery-wt-A/A2/B/B2/C/M` preserve divergent October recovery states. Exact refs, detached caches and prunable temporary worktrees are retained in `E/SOURCE_DISCOVERY_SNAPSHOT.json` and `E/evidence/02-backend-git.md`; none was pruned or synchronized.

### 3.2 Actual and unproven integration edges

```text
Next.js frontend F
  ├─ gateway /api-auth, /api-exams, /api-attempts, /api-speaking, /api-ai, ...
  │    └─ .NET services → Python AI service
  │         ├─ text grading / conversational LLM / RAG
  │         ├─ ASR (historical Whisper.NET; later faster-whisper)
  │         └─ separate four-output pronunciation regressor
  └─ /api-generation → local YARP/AppHost wiring → Node generation-harness
       └─ Reading authoring → projection → review/repair → evidence gates

Speech RQ1 S → authorized remote GPU training → Drive/HF artifacts
             → validation analyses / frozen-emission scorers
             - - no verified artifact/config/API edge into backend scorer - -

Remote IELTS data-export repo → potential RAG/fine-tuning source
                              - - exact local training provenance unverified - -
```

**VERIFIED static integration, UNKNOWN runtime:** `F/src/app/admin/generate/generationApi.ts` and `types.ts` correspond to local backend config/jobs/result/cancel/artifact/SSE routes; modified `B/AppHost/Program.cs` and gateway settings add the Node service on port 5050 and `/api-generation`. The server/adapter and parts of this wiring are **uncommitted**, not contained in backend HEAD merely because the files exist [S04, S13].

**Distinct model lines:** backend `PronunciationScorerModel` uses `facebook/wav2vec2-base`, mean pooling and a four-value regressor; S uses pinned `wav2vec2-base-960h` with phoneme CTC and binary MDD. No inspected mount/config/checksum establishes that S's E0 checkpoint is the backend model [S04, S15].

## 4. Evidence-backed chronological timeline

**Timestamp discipline:** `G` = Git author/commit time, `D` = document-stated date, `U` = session/request time, `A` = artifact-store commit time, `O` = observation. A commit or report timestamp is not silently converted into execution time. Date ranges indicate unresolved execution precision. Abbreviated SHAs resolve through the full Git snapshot [S02].

| Time (+07 unless stated) / kind | Workstream: original goal → decision, rationale, alternative | Action/result, limitation and downstream effect | Evidence / present validity |
|---|---|---|---|
| 2025-09-15 02:54:06 G | Backend inception; build application rather than a stand-alone model script | Root `62072d5`; exact original conception date UNKNOWN | S02; implementation origin VERIFIED |
| 2025-09-15–23 G | Establish identity, asynchronous boundaries and exams | Google login, rotating sessions/Redis, RabbitMQ/email, gateway, exam APIs; e.g. `e8eadb8`, `087cc9b`, `63b2fe5` | S02; historical foundations, not live-health evidence |
| 2025-09-16 13:23:29 G | Separate user-facing application | Frontend root `b7403a4`; later delivery/admin flows develop in distinct repo | S02; separation remains |
| 2025-11–12 G | Add spoken/written practice and grading | Whisper.NET (`5b54b74`), speaking audio, writing/speaking grading and placement/full-test flows | S02; later providers/ASR evolved |
| 2026-03-16 13:29:31 O/API creation metadata | Organize reusable IELTS AI data exports | Related public data repo created; essays/grammar/speaking/export directories now visible by metadata | S10; exact consumption/licenses UNKNOWN |
| 2026-03-18 G | Ground writing feedback rather than rely only on LLM prior | Qdrant RAG pipeline (`0043ee8`) and RAG+Gemini comparison (`8447205`) committed | S02; source history, not fresh quality evaluation |
| 2026-03-23–04-28 G/API | Extend roleplay and unify writing flows | Roleplay PR #4; writing E2E/grammar/compare changes and merged backend PR #9 / frontend #38 | S02–S03; historical integrations |
| 2026-05 G | Consolidate service orchestration/health and grounded grading | Aspire migration and shared defaults; per-service integration; grounded prompt commit `4afd7cb` | S02; .NET8 proposal later differs from .NET10 current source |
| 2026-05-31 13:44:50 G | Preserve local grading adapters | Qwen model paths first appear in reachable history at `1cd4ec7711279bc19b676e62198e4ebdb24d2a98` (also duplicated lineage `23266fb...`); current trainer state records766steps/one epoch | S15/S02; Git inclusion time is not verified training time |
| 2026-06-01–10 G | Change LLM/provider and feedback envelopes | Local-LLM change; Groq routing (`1596c42`), legacy-provider cleanup, per-criterion RAG/evidence IDs and explainers | S02; provider choice is historical, not timeless |
| 2026-08-18 D | Formalize dual-axis graduation research | P defines speech/feedback RQ1 and educational-generation RQ2, with A/B/C baselines and cost/quality evaluation | S01; authoritative original intent, approval authenticity not verified |
| 2026-08-20 name only | Preserve project copies on Drive | Backup root `langfens-projects-2026-08-20` later found with backend and two frontend-named roots | S10; name is NOT verified backup/event date |
| 2026-09-07–22 G | Align exam types, grading and autogeneration | Paper normalization, blank/flow-chart conventions, Python schema/prompt mirrors, 19-type router and canonical generation contracts | S02; important precursor, not erased by newer harness |
| 2026-10-01–04 G | Move beyond single prompt toward autonomous generation/recovery | `7a4d0e7` loop/evaluators/repair; `1ea298a` engine v2 | S02; older service strand |
| 2026-10-12–14 G/API | Repair valid-Draft acceptance and product regressions | Recovery branches/worktrees; backend PR #13 merged 2026-10-14 17:14:37 +07, `fc3c9fe...` | S02–S03; divergent recovery evidence retained |
| 2026-10-20 API | Integrate frontend autonomous-generation work | Frontend PR #44 merged 11:00:02 +07, `42f77bf...` | S03; merge is not research acceptance |
| 2026-10-21–23 U/D/G | Establish speech datasets and reproducible protocol before training | Archive/annotation audits, schema/phoneme rules, deterministic speaker splits and ingestion; speech root `085bfcd...` at Oct23 15:01:33 | S02, S05; historical audit claims versus directly checked manifest hashes distinguished |
| 2026-10-23–25 D/G/U | Choose Wav2Vec2 MDD and debug learning; alternative classical GOP | Three-head prototype; small sanity run exposes blank collapse and misleading mixed diagnosis denominator; stabilization pilots select frozen-backbone/warmup recipe | S05, S11; third-head diagnosis not retained as trained baseline capability |
| 2026-10-23–26 G | Improve old generation content/evidence/recovery lineage | Backend commits add provenance, bounded repair and no-paid-redispatch behavior; master PR #14 merged Oct24 16:10:28 +07 | S02–S03; distinct from later native-harness branch |
| 2026-10-25–27 D/G/U | Train and verify a two-head Fold-1 reference, then zero-oracle evaluation | Colab run recorded 6,600 updates; selected epoch6/3,600 E0; Protocol Z and numerical/provenance corrections follow | S05–S06; precise run event time not reconstructed from conflicting summary dates |
| 2026-10-27 U/D | Create an independent native generation substrate rather than rely on installed global Pi | Design/plan and sessions request Pi-derived Node harness, local Core/Subagents and isolated runtime | S04, S11; substrate and domain evolution separated |
| 2026-10-27 D/G | Preserve SDD progress and stop on scope/build incident | Progress ledger records tasks1–5 and later 6A incident; no clean whole-plan completion inferred | S04; historical engineering blocker |
| 2026-10-27–28 D/U | Make baseline artifacts durable; launch controlled acoustic ablation | Baseline export/Drive/HF records; Colab quota/Drive-TTY blockers, CPU-FP32 feasibility discussion, then Kaggle/HF migration | S06; historical fallback discussion is NOT current fallback authorization |
| 2026-10-28–29 G | Add native Reading vertical slice and deterministic validation | `85d87ea`, `0feaac0`, `0d7e60d`; schema/builders/evidence/projection/roles | S02, S04; mechanical verification does not establish content quality |
| 2026-10-29 D/A | Test prompt dropout against matched continuation control | C0/C1 each recorded3,600 additional updates from E0; negative C1−C0 point estimate; retain R | S06; hypothesis NOT_SUPPORTED in tested setting |
| 2026-10-29 18:26:26 G | Update frontend source-of-truth contract | F HEAD `2d9fe94...`; subsequent local changes remain dirty | S02–S04; committed versus worktree state distinct |
| 2026-10-30 02:03:59 U and later U | Restore minimal native harness while preserving canonical contracts; then address reliability | Session `01a1890c...` requests removal of Reading-specific orchestration; later requests reintroduce minimal role/profile/review features | S11; evolution, not a single static requirement |
| 2026-10-30–11-01 D/G | Generalize speech baseline without Fold1 warm starts | Folds2–4 independently initialized; dual fixed/best endpoints; four-fold synthesis commit `618e86d` Nov1 02:13:50 G | S07; validation only, heterogeneous cohorts |
| 2026-10-31 18:14:07 G | Update native harness | B current HEAD `2ba12e5...`; later server/benchmarks/fixtures remain local and uncommitted | S02; not a clean release of current tree |
| 2026-10-31–11-01 D/artifact | Measure MiniMax thinking profiles and repair tool failures | Eight runs; minimal/high authoring2/2, medium0/2; all final quality outcomes UNVERIFIED; post-fix revisions and fixture receipts follow | S12–S13; no publication-quality success inferred |
| 2026-11-01 G/D | Screen loss-only speech interventions before cross-fold spend | Focal arm3,600; weighted arm3,598 exploratory; Gate C stops Stage2, corrects bootstrap/p-value and extrapolation errors | S07; four-fold treatment criteria remain UNMEASURED |
| 2026-11-01–03 artifact/U | Investigate long generation runs, blind review, grading parity and revisions | r4 frozen artifact/evidence; Trial3 and Cool Islands aggregates; latest narrow correction requests preserve originals | S12–S13, S11; separate artifacts/review scopes must not be mixed |
| 2026-11-02 D/G | Diagnose speech omissions and test a causal explanation | DIAG01/02; DIAG02B invalidates forced-MHA causal proof, retains9/80 archive-parity failures and NO-GO | S08; alignment hypothesis unresolved, not disproved universally |
| 2026-11-02–03 D/G/artifact | Test alignment-aware E1 before cross-fold rollout | Recorded3,600 updates; omission AP worsens and displayed mean skip mass .000000; rollout NO-GO | S09; does not falsify all alignment/skip methods |
| 2026-11-03 D/G | Reopen literature/intent fairly across method families | Six-family/19-candidate research closure, corrections, frozen contract and separate P-SCORE/P-SUP worktrees | S09, S14; rubric is readiness judgment, not performance ranking |
| 2026-11-03 16:57–17:42 U | Expose generation progress and resolve product startup/routing symptoms | User requests terminal logs/agent progress, then reports missing terminal, AppHost waiting on `exam-internal-api-key`, and generation-job POST returning502timeout | S11/session01a1a0b7; VERIFIED reports of symptoms, not reproduced or proved fixed |
| 2026-11-03 D/G/artifact | Score frozen E0 emissions without retraining | P-SCORE arm `a9f4947...`; omission gain but substitution noninferiority failure | S09; benchmark complete is not overall scientific acceptance |
| 2026-11-03 18:13–22:31 G | Make P-SUP source immutable and fix deployment problems | Earlier `df55f12` pin superseded; S HEAD22:31:54 pins `dd2dd89`; current fetched notebooks agree | S02, S09; historical notebook-version/run binding still incomplete |
| 2026-11-03 ≤23:08:03 A | Persist latest P-SUP control artifacts | HF commit `4aa9f2a...` at23:08:03, summary commit23:07:53; summaries for both arms record3,600 updates/0skips | S09; artifact time is not exact training start/end; final VAL comparison UNKNOWN |
| 2026-11-03 23:39–11-04 O | Recover ecosystem context, not implement or run experiments | Ten reports, direct authenticated metadata, hashes, contradictions and source registry retained | E; this report, with explicitly accounted gaps |

Full reachable commit histories, reflogs, branch tips and selected session IDs are retained rather than replaced with this milestone table [S02, S11].

## 5. Architectural and research decision history

### 5.1 Product platform and generation engineering

**Service boundaries — CORROBORATED:** backend now documents .NET10/Aspire/YARP, Python FastAPI, PostgreSQL per service, RabbitMQ/MassTransit, Redis and Qdrant. HTTP/gRPC/event contracts separate owners; cross-database application queries are prohibited. The proposal's .NET8 was a planned technology, not evidence the current code remains .NET8 [S01, S04].

**Generation evolution — VERIFIED source/history:** older Python prompt/schema and generation-service loop mechanisms precede the native Node harness. They must not be flattened into “single prompt until October27,” nor assumed removed: `B/services/ai-service/app/prompts/autogen.py` and legacy generation-service source remain. Historical current-default routing is UNKNOWN [S02, S13].

**Native substrate versus domain policy:** October27 design/plan names a Pi-derived runtime and Subagents; later source places IELTS assets under `.generation/`, with native read/bash/edit/write plus domain tools and specialist roles. Session requests oscillate from broad parity to minimal-core restoration and then domain reliability. This is evidence of evolving priorities, not authority to restore an old design now [S04, S11].

**Intentional pilot constraint/workaround:** the 13-mark pilot aligns independently scored Question entities with the existing grader. Authentic multi-slot exercises and backend all-or-nothing semantics require explicit contract treatment; schema validity alone cannot certify pedagogical authenticity. Whether to change scoring or retain single-slot authoring is unresolved, not decided by this investigation [S04, S13].

**Review provenance is artifact-bound:** public-first solving, fixed artifact/projection hashes and distinct semantic/examiner roles are intentional safeguards. Shared filesystem permissions mean CWD separation is not an OS sandbox. A reviewer agreeing with keys cannot upgrade another revision's provenance or publication state [S12–S13].

**Current transport adapter is local work:** HTTP jobs/SSE/cancel/result/artifact code and AppHost/YARP/frontend wiring exist, but are largely dirty/untracked. This may be a legitimate transport layer rather than a replacement native agent loop; boundary conformance and runtime behavior remain UNKNOWN. A source-level auth concern also remains: local server signature verification is conditional on signing-key configuration. No exposed production vulnerability was tested or established [S13].

### 5.2 Speech methodology

**Data/protocol first:** historical audit records report3,599 usable L2-ARCTIC WAV/TextGrid pairs,245 orphan WAV exclusions,119,151 canonical tokens and17,518 errors; these are recorded census results, not a new scan here. L2-ARCTIC is governed by CC BY-NC4.0; SpeechOcean762's5,000utterance/250speaker scoring corpus is a distinct CC BY4.0 resource, not interchangeable MDD labels. Fold1 contract is2,399TRAIN /300VAL (NJS/RRBI) /900TEST sealed. Original archives remain persistent; frozen speaker-disjoint manifests define allowed folds. The pipeline normalizes to 39 phones plus CTC blank, separates canonical/realized phones, and rejects active-fold TEST staging. Cross-fold speaker rotation is legitimate; sealing is per fold, not a prohibition on a speaker ever appearing in another fold's training [S05].

**Three heads → two trained objectives:** a diagnosis head was prototyped, but small-run diagnosis accounting mixed correct and substitution tokens. The strict substitution-only result was reported as4/124≈3.23%. The selected baseline trains CTC+binary; the diagnosis head is frozen/untrained. A gradient-conflict explanation is a historical rationale/hypothesis, not an independently established universal cause [S05].

**CTC stabilization:** blank collapse motivated learning-rate/schedule/backbone pilots. The registered recipe uses frozen `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, seed1337, batch2/accum2, warmup100 and a cosine horizon over25epochs. The fixed3,600 endpoint is not the same thing as a3,600-total-update run when early stopping continues afterward [S05, S07].

**Protocol O versus Z:** O supplies annotation-derived canonical tokens, not gold timestamps as model inputs. Z phonemizes written prompts and aligns predictions/labels offline. Z demonstrates bounded zero-oracle evaluation feasibility, not deployed diagnostic readiness. Full versus aligned cohorts and evaluator precision must remain explicit [S05, S08–S09].

**Supervision is not absent:** realized-phone CTC targets exclude deleted canonical phones, so they do not supply an explicit canonical-deletion target. Binary MDD already labels deletions as errors. It is incorrect to conclude that omissions had no supervision, or that every omission-related parameter receives zero gradient [S14; coordinator C-13].

**Diagnostics versus causality:** attention concentration, weak omission ranking and acoustic mismatch gaps motivate hypotheses. DIAG02's forced-attention code bypassed learned projections; its drift cannot support the original causal claim. DIAG02B preserves exploratory sensitivity evidence, not proof that attention routing causes natural omission failures [S08].

**Separate scoring and supervision hypotheses:** P-SCORE holds E0 emissions fixed; P-SUP changes the objective with parameter-matched control. P-SUP does not depend on P-SCORE succeeding first. A study's scoped objective amendment is not blanket permission to alter the historical baseline or launch another arm [S09, S14].

**Graph/license precision:** the selected independent `ctc-union-language-v1` S/SD graph is not exact reproduction of the author's neighbor-restricted graph/occupancy approximation. A deletion-capable denominator is not a calibrated human-omission posterior. Author-code copying remains license-dependent; paper access/CC BY4.0 does not license all linked code, weights or datasets [S14].

## 6. Experiment and model-development history

### 6.1 Backend models and product evaluations

| Record | Established evidence | Limitation / disposition |
|---|---|---|
| Qwen2.5-1.5B LoRA/SFT | `B/models/qwen25-lora/adapter_config.json`, model card, checkpoint600/766 trainer metadata; state766 reports epoch1, max/global_step766, batch1; best_metric/checkpoint null | Training records/artifacts exist; original dataset/platform/run authorization and independent grading-quality validation not reconstructed. No model loaded. README's68s CPU inference is a historical claim, not current measurement |
| Backend pronunciation-scorer-v2 | Five-epoch CSV; smallest listed val_loss .004361358 at epoch2; source four-output pronunciation/accent/fluency/prosody regressor | Different task/model from phoneme MDD; metrics lack recovered dataset/split lineage in this inspection. Not RQ1 E0 deployment evidence |
| RAG/grading/latency README claims | Reported n=5 essay pilot MAE1.20→.90, dictionary/ASR/embedding/LLM timings and operational post-mortems | Useful historical claims, not freshly reproduced benchmarks, adequate population evidence or current live readiness. Exact raw evaluation chain remains incomplete |

Backend Git currently tracks LoRA payload/state paths, unlike the speech repository's current “weights out of Git” policy. This is a historical governance difference requiring provenance/license review, not proof of a newly caused leak or permission to delete files [S15].

### 6.2 Autonomous-generation experiments

| Study/artifact | Recorded outcome | Defensible conclusion |
|---|---|---|
| MiniMax-M2.7 thinking benchmark;8 runs,2/profile | Authoring completion off1/2, minimal2/2, medium0/2, high2/2; reported minimal median420s/$0.253 | Small authoring-throughput comparison. **All eight final quality states are UNVERIFIED**. “Publication-grade” prose overstates the report's own gates |
| Native reliability investigation | Records tool-shape/array-wrapping errors, invalid workflow calls, retries, provider latency and schema/tool improvements | Trace-supported engineering leads; not proof all reliability defects fixed or model-independent quality gained |
| Postfix minimal run2 / r4 | Historical receipt records284/284 .NET tests,11 r4-specific; EF Core InMemory only. Parent recomputed r4 and projection SHA-256, matching manifest | Structural/fixture evidence only. `actual-completion.json`: overall UNVERIFIED, semantic/IELTS/reviewProvenance/runtime UNVERIFIED; publicationReady=false, publicationIntegration=NOT_IMPLEMENTED |
| Trial3 computed aggregate |294turns;18,865,628 total tokens, including17,141,220 cache-read;47 tool failures; modeled cost$1.8609801 | Recorded aggregate, not provider invoice or independently recomputed quality outcome; original and corrected revisions must remain separate |
| Cool Islands computed aggregate |459turns;33,773,831 total tokens, including31,229,239 cache-read;74 failures; modeled cost$3.19117014 | Separate session/revisions; specialist agreement cannot certify unrelated r4 or establish formal RQ2 superiority |

Frozen r4 identity: `4958e156c7e4b0183f55633dedccbab2807166d0fd4f322cdd424303079c6f54`; delivery projection: `c2cc490d88ef72ee8fac9f57b63f098ad7e922552eeeb31c4b05021334960932`. Source: `B/services/generation-harness/benchmarks/runs/postfix-minimal-run-2/runtime/reading/workspace/` and `benchmarks/final-evidence-wiring/actual-completion.json` [S12–S13].

No examined evidence establishes completion of the proposal's fair single-LLM/self-review/multi-agent campaign across reading **and grammar**, failed requests included, with independent quality assessment. Its status is **UNKNOWN / not demonstrated**, not universally proved unexecuted [S01, S13].

### 6.3 Speech studies

These are **CORROBORATED recorded outcomes**, not new training/evaluation runs by this mission.

| Phase / model | Training/exposure and key result | Result/limitation retained |
|---|---|---|
| TRN01B0/B1 sanity and stabilization | Small labeled subset; blank collapse and diagnosis denominator defect; stabilization pilots | Historical debugging; not full-corpus or deployment evidence |
| E0 / Baseline R Fold1 | Colab T4; selected epoch6/3,600; full run reached epoch11/6,600; checkpoint `a2445c8d...` | Keep selected-best and last separate; diagnosis not trained |
| C0/C1 prompt dropout | Both initialize from E0 `best.pt`, then each3,600 additional updates; C0 p0 versus C1 p.25; ProtocolO AP .310304 vs .302386, delta−.007918; CI[−.018328,+.000385] | Hypothesis NOT_SUPPORTED in this Fold1/seed/continuation setting; R retained. **Not** independent original-pretrained cross-fold baselines |
| Baseline R Folds2–4 | Independent original-pretrained initialization; fixed3,600 and selected-best endpoints saved; four-fold fixed macro AP .279207, micro .272580 |39,576tokens/8validation speakers; heterogeneous performance, no TEST/cross-corpus claim |
| TRN03B focal/weighted pilots | Focal3,600 versus weighted3,598 exploratory endpoint; fixed contrasts≈−.0089; paired permutation p .4790/.6190 | Stage2 stopped for lack of positive pilot signal; four-fold treatment criteria UNMEASURED, not confirmatory falsification. Bootstrap sign fractions are not p-values |
| DIAG01/02/02B | Four-fold omission-vs-correct AP .030506 versus substitution .264270; DIAG02 causal intervention invalidated; DIAG02B9/80 pointwise parity failures | Current PARTIAL / NO_GO_TRAINING_CAUSAL_GATE; no full1,199-utterance forward replay established |
| E1 alignment/skip candidate | Recorded3,600 successful/0skipped; omission AP .032252 vs E0 .035020; binary .309293 vs .317021; realized PER improves .236818→.232365 | Omission/full-MDD tradeoff failed; mean skip mass displayed .000000, not proof mathematical zero at every token; no Folds2–4 rollout |
| P-SCORE | Frozen E0 FP32 emissions,300VAL utterances; all-position scoring report; SF-SD versus SF-S omission delta+.021615 and recall delta+.182482 | Substitution delta−.024256 fails −.005 noninferiority; binary delta−.009045. Omission benefit is not an all-task winner |
| P-SUP | Both private HF summaries now record3,600successful/0skipped/3,600attempts, fixed_budget_saved=true;8epochs×450updates | Operational training records recovered; final VAL comparative/scientific acceptance remains UNKNOWN |

**Cross-fold endpoints from the recorded evaluation CSV** [S07]:

| Fold / validation speakers | Fixed3,600 AP | Best AP / updates |
|---|---:|---:|
|1 NJS/RRBI |.317012 |.317012 /3,600 |
|2 TNI/YBAA |.201451 |.217045 /1,200 |
|3 TXHC/ZHAA |.307324 |.307324 /3,600 |
|4 HKK/NCC |.291042 |.316023 /1,800 |
|Macro / pooled micro |.279207 /.272580 |.289351 /.287469 |

Selection gives unequal exposure; selected-best aggregates cannot replace fixed-budget comparisons. Fold3's training-log AP .307479 and evaluator AP .307324 are different recorded evaluations, not silently interchangeable values.

### 6.4 Metric population and precision ledger

| Quantity | Population/protocol | Recorded value / distinction |
|---|---|---|
| Fold1 online baseline AP | ProtocolO, Colab FP16 AMP |.316062 |
| Fold1 CPU replay AP | ProtocolO, FP32, full canonical cohort |.317021 |
| Fold1 cross-fold evaluator AP | Its recorded evaluator/runtime |.317012 |
| ProtocolZ AP | Aligned-only G2P cohort |.313530 historical replay; cross-fold CSV separately lists .313513 |
| Full Fold1 canonical cohort |8,799correct+1,012substitutions+145deletions |9,956tokens |
| Aligned cohort |8,774correct+1,008substitutions+137deletions |9,919;37unaligned=25correct+4sub+8del |
| Legacy omission-vs-correct denominator | Substitutions excluded |Full8,944; aligned8,911 |
| Legacy substitution-vs-correct denominator | Deletions excluded |Full9,811; aligned9,782 |
| Realized CTC PER reference | Not canonical phone count |9,881reference phones; P-SCORE report2,340edits/.236818 |

No numerical trend should be manufactured by mixing these protocols, score types, evaluators, populations or precision regimes. Historical .316062→.317021 differences are documented as precision/evaluator boundaries; this mission did not independently rerun their causal explanation [S05, S07–S09].

### 6.5 Latest P-SUP durable records: training is not final evaluation

**VERIFIED remote record contents**, at HF revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d` in private `minkhoaa/langfens-rq1-cross-fold-checkpoints` [S09]:

| Arm | Objective | Recorded updates/skips | Best SELECT AP / update | Fixed-budget checkpoint LFS SHA-256 |
|---|---|---|---|---|
|B-CONTROL |`binary_collapsed` |3,600 /0 |.46760702566135065 /2,250 |`e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` |
|T-OP |`operation_ce` |3,600 /0 |.46497542138006975 /2,700 |`a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` |

Best checkpoint LFS hashes: control `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f`; treatment `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9`. These are server-reported content identifiers, **not fresh tensor-payload readbacks**.

Registered split is FIT1,799 / SELECT300 / CAL300 / VAL300, with TEST900 sealed; trainable-head capacity is matched at266,667. SELECT values near .467 **must not** be compared with E0's .317 VAL AP. No P-SUP omission/substitution/full-VAL noninferiority conclusion follows. Epoch8 here reaches3,600; baseline's6epochs/3,600 must not be imposed on the smaller FIT set.

Prospective P-SUP acceptance thresholds recorded before candidate evaluation include omission AP delta≥+.030, recall@FPR15 delta≥+.040, protected-metric noninferiority margin−.005, and PER delta≤+.005. P-SCORE's corresponding omission/recall improvement thresholds were+.020/+.030. These are registered decision rules, not achieved P-SUP VAL results or new execution permission; the frozen V1 contract controls exact score/cohort/control definitions [S09].

Both current Kaggle sources pin `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`, include `--remote-execution`, and report COMPLETE. Exact historical notebook version→run→summary linkage, payload cryptographic readback, and launch-authorization provenance were not fully recovered.

## 7. Source-of-truth hierarchy and exact source registry

### 7.1 Hierarchy by claim type

| Claim type | Preferred authority | Do not substitute |
|---|---|---|
| Original intent / research questions | Original proposal; explicit dated user amendments; contemporaneous decision/design records | An English-title guess, newest branch, agent majority or current implementation |
| Current implementation | Exact Git tree **plus separately identified dirty/untracked source**, contracts and diffs | README, commit title alone, test-file existence, or default-branch name |
| Experimental outcome | Run logs, immutable endpoint/history metadata, full-precision predictions, evaluator identity and population | Kernel COMPLETE, summaries without provenance, synthetic tests or requested authorization |
| Data provenance | Frozen manifest/checksum, license/source ledger, allowlists and staging validation | Dataset title, nominal fold number or mere file presence |
| Scientific interpretation | Valid measurement/design and explicit correction ledger, with uncertainty and scope | Uncontrolled counterfactual, causal language in an old report, rubric scores or unreproduced metric claims |
| Current remote state/privacy | Timestamped authenticated metadata for that resource | Local deployment JSON, old “private” description or another resource's visibility |
| Historical reasoning/operations | Timestamped session messages, handoffs, checkpoints, correlated with primary actions/results | Treating chat requests as execution or historical instructions as current authorization |
| Execution permission | Current user authorization and applicable repository safety policy | Old GO, a passed fixture, an uploaded notebook or completed historical experiment |

### 7.2 Registry

| ID | Exact reference(s) | Coverage / confidence |
|---|---|---|
|S01 |P: lines8–16; main line14 §§2.2/2.3/3/4/7; dated area line16 |Original intent directly read, VERIFIED contents; no approval/signature authenticity claim |
|S02 |`E/SOURCE_DISCOVERY_SNAPSHOT.json`, `SOURCE_FILE_INVENTORY.tsv`, `PRESERVATION_CHECK.json`; Git objects at S/B/F roots |All local refs/reachable commit lists/reflogs/worktrees,5,196path inventory; content review selective; no fetch or unreachable-object census |
|S03 |`E/GITHUB_METADATA_OBSERVATION.json` |Direct GET metadata/branches/PRs/issues/releases for three primary repos; one page100 each, returned counts below100; CI/comments/checks not comprehensively reviewed |
|S04 |B/AGENTS.md; B/README.md; B/docs/superpowers/specs/2026-10-27-langfens-harness-design.md; corresponding plans file; B/.superpowers/sdd/2026-10-27-langfens-harness/progress.md; B/services/generation-harness/AGENTS.md and README.md; F/ARCHITECTURE.md |Goals/design/source directly or independently cross-read; current runtime UNKNOWN |
|S05 |S/manifests/RQ1-AUD-01/PROCESS_CHECKPOINT.md; RQ1-STD-01/PROCESS_CHECKPOINT.md; RQ1-SPLIT-01/l2arctic_4fold_splits.json; RQ1-PREP-01/PROCESS_CHECKPOINT.md; RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md; RQ1-TRN-01A/PROCESS_CHECKPOINT.md; RQ1-TRN-01B0/reconciliation/SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md; RQ1-TRN-01B1/PROCESS_CHECKPOINT.md; RQ1-TRN-02A/PROCESS_CHECKPOINT.md |Historical data/model/protocol records; split hash independently recomputed |
|S06 |S/manifests/RQ1-ARTIFACT-01/; RQ1-TRN-02B/PAIRED_SCIENTIFIC_EVALUATION.md; RQ1-TRN-02B/checkpoints/08-live-gpu-runtime-and-staging-verification.md,09-colab-cpu-fp32-feasibility-static-gate.md,10-kaggle-t4x2-migration-feasibility-static-review.md; S/configs/rq1_trn_02b_c0_kaggle.yaml and c1 equivalent |Recorded persistence/platform/ablation evidence; initialization confirmed in config; no fresh experiment |
|S07 |S/manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json; four_fold_metrics_summary.csv; checkpoints/12-official-fold4-baseline-r-training-closure.md,13-four-fold-validation-synthesis-and-closure.md,19-gate-c-scientific-closure.md; evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json |Cross-fold/negative-pilot records. **TRN03B records live under TRN03A**, not an assumed separate03B directory |
|S08 |S/manifests/RQ1-DIAG-01/; RQ1-DIAG-02/; RQ1-DIAG-02B/SCIENTIFIC_REPORT.md,PROCESS_CHECKPOINT.md,evaluation-final/REPORT.json,reconciliation-final/CORRECTIONS.json |Corrected primary report/correction ledger;9/80 parity failure preserved |
|S09 |S/manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json; S/manifests/RQ1-WORKSPACE-PREP-01/{EVALUATION_CONTRACT_V1.md,SCIENTIFIC_QA_AND_COMPARISON_REPORT.md,P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md}; P-SCORE arm/manifests/RQ1-P-SCORE/evaluation/BENCHMARK_REPORT.json; P-SUP arm/manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md; `E/HF_METADATA_OBSERVATION.json`, `HF_SOURCE_REGISTRY.json`, `P_SUP_REMOTE_METADATA.json`, `KAGGLE_METADATA_OBSERVATION.json`, `KAGGLE_DATASET_METADATA.json` |Candidate records and fresh remote metadata. E0/E1 outcomes mostly historical; P-SUP summaries directly retrieved at immutable HF revision |
|S10 |`E/RELATED_SOURCE_DISCOVERY.json`, `DRIVE_COLAB_OBSERVATION.json`, `DRIVE_BACKUP_INVENTORY.json`; `E/evidence/04-drive-colab.md` |Actual remote locations discovered; root/path/metadata inventory, not data/weight payload inspection or complete Drive archaeology |
|S11 |`E/PI_SESSION_INVENTORY.tsv`; `E/evidence/01-sessions.md`; Pi roots `~/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/` and `--home-khoa-Projects-langfens-Project_Langfens_Microservice--/` |260project-related headers; lane indexed73speech/183backend sessions and selected user requests; not full-body review. Raw table contains adjudicated date errors |
|S12 |B/services/generation-harness/BENCHMARK_MINIMAX_THINKING.md; .generation/reading/NATIVE_RELIABILITY.md and BLIND_REVIEW_WORKFLOW.md; benchmarks/metrics/{trial-3-computed.json,cool-islands-computed.json}; benchmarks/runs/ |Recorded generation outcomes/trace aggregates; cost and quality not independently rerun |
|S13 |B/services/generation-harness/benchmarks/final-evidence-wiring/{actual-completion.json,evidence-sha256.json,identity-audit.json}; src/server/{server.mjs,adapter.mjs}; B/AppHost/Program.cs; B/gateway/api-gateway/appsettings.json; F/src/app/admin/generate/{generationApi.ts,types.ts}; `E/evidence/09-verify-platform.md` |Direct source/receipt inspection and18-claim independent review; exact r4/projection hashes recomputed by parent; no runtime tests |
|S14 |S/manifests/RQ-RESEARCH-01/closure-2026-11-03/{INTEGRATED_DECISION_REPORT.md,SOURCE_EVIDENCE_MATRIX.md,SOURCE_DATA_COMPATIBILITY_AUDIT.md,SCIENTIFIC_PROTOCOL_CORRECTION_REPORT.md,UNIFIED_CANDIDATE_COMPARISON.md,PARALLEL_PILOT_SPECIFICATIONS.md}; S/manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness/SHARED_GOP_PRIMARY_SOURCE_AUDIT.md |Method/compatibility/primary-literature correction records; subjective readiness scores, not candidate performance |
|S15 |B/models/qwen25-lora/{README.md,adapter_config.json,checkpoint-766/trainer_state.json}; B/models/pronunciation-scorer-v2/training_history.csv; B/services/ai-service/app/services/pronunciation_scorer.py:18–39,64–82; B/README.md:230–305 |Legacy model metadata/architecture and product claims; no tensor load or fresh live-functionality claim |

Key lightweight identities independently checked:

- Split manifest SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- Evaluation Contract V1: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Historical E0 `best.pt`: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` — provenance corroborated, **not newly payload-hashed in this mission**.
- Authorized Fold1 package: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` — historical ledger, **not newly downloaded/extracted**.

### 7.3 Remote asset locators (metadata versus payload verification)

- **Drive:** existing rclone remote `kltn_gdrive:`; `KLTN-Dataset/` is the documented persistent project root, folder ID `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` ([folder locator](https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK)). Listings also found `KLTN/` original archive copies and `KLTN-Collab/KLTN-Dataset.ipynb`, plus `Colab Notebooks/Untitled0.ipynb`. Notebook bodies/execution history were not reviewed. Historical baseline `best.pt` file ID is `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`; fold package ID `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY`. IDs/hashes are historical ledgers unless a receipt explicitly says newly verified.
- **Drive archived product copies:** `kltn_gdrive:langfens-projects-2026-08-20/{Project_Langfens_Microservice,langfens-fe,langfens-fe-app}/`; directory metadata only, exact archived refs UNKNOWN.
- **HF baseline:** [minkhoaa/langfens-rq1-fold1-baseline-r](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r), revision `6c31f7196c2e03a6766c5525833de465a1a9345d`, currently public.
- **HF C0/C1:** [minkhoaa/langfens-rq1-trn-02b-checkpoints](https://huggingface.co/minkhoaa/langfens-rq1-trn-02b-checkpoints), revision `56c9c419ef2bb5698e25dc71390b27681b7efe65`, currently private.
- **HF cross-fold/P-SUP:** [minkhoaa/langfens-rq1-cross-fold-checkpoints](https://huggingface.co/minkhoaa/langfens-rq1-cross-fold-checkpoints), revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`, currently private; P-SUP paths `psup/b_control/` and `psup/t_op/`.
- **Kaggle latest inspected arms:** [T-OP](https://www.kaggle.com/code/minkhoaa/langfens-rq1-psup-t-op-t4x2), [B-CONTROL](https://www.kaggle.com/code/minkhoaa/langfens-rq1-psup-b-control-t4x2); current fetched notebook source pinned dd2dd89, T4 metadata/COMPLETE, API notebook visibility mismatch recorded.
- **Kaggle data:** `minkhoaa/langfens-rq1-fold1-staging` through `fold4-staging`, current versions3/2/1/1, all private. The full29related-kernel slug inventory and four dataset identities are in `E/RELATED_SOURCE_DISCOVERY.json`; no payload downloaded.

### 7.4 Sessions and agent receipts for reproducibility

Selected historical session IDs, with **filename UTC timestamps**, not invented execution dates:

- Speech protocol: `2026-10-23T03-42-56-864Z_01a166da-c960-7408-9430-47458c001ab1`; split `2026-10-23T06-11-11-423Z_01a16762-81bd-7408-9430-4747d1eb6e6c`.
- Baseline/zero-oracle: `2026-10-25T13-02-00-526Z_01a17327-574e-77e7-846a-105ae36fb43e`; `2026-10-26T07-26-14-501Z_01a1771a-4be5-7507-9ea3-225ceaa1ff45`.
- DIAG02B: `2026-11-02T07-33-55-777Z_01a19b2d-d9c1-773a-83f5-a5849c75a6c1`; separate arms: `2026-11-03T05-57-00-579Z_01a19ffb-7a23-76bc-8443-2d57fb5e04c7`.
- Backend harness formation: `2026-10-27T09-46-09-348Z_01a17cc0-c043-7280-97c3-ad77411dd7a8`; minimal reset `2026-10-29T19-03-59-298Z_01a1890c-2e42-75f2-88d0-23c5b8e06449`.
- Product operational follow-up: `2026-11-03T09-22-35-196Z_01a1a0b7-b039-74e7-a4d2-907562679a13`; directly inspected user messages16:57–17:42+07 request agent-log terminal and report AppHost-key wait/502job timeout. Resolution not established.
- r4 parity: `2026-11-01T04-35-51-344Z_01a19564-75b0-7165-ba6e-2d2283ea18b3`; Trial3 request `2026-11-02T06-28-08-809Z_01a19af1-9fe8-7594-8a12-a1e7873a22bc`; narrow correction `2026-11-03T04-44-06-176Z_01a19fb8-baa0-74e7-a4d2-9073fb4eda8b` has a directly inspected user request at2026-11-03T04:44:48.817Z (11:44:48.817+07): fix two Trial3 defects, preserve originals/save a new revision, run only existing targeted checks, and do not launch another benchmark/AI review. This verifies intent, not execution.

This investigation's workflow `ea49dbe6-c72f-494d-b433-77701f7d9e45`, mission `465ab78f-1a75-4650-a7fb-efb005b9b2f9`, and ten child IDs are preserved in `E/WORKFLOW_RECEIPT.json`. Five Luna discovery lanes plus one Luna reviewer and three Gemini discovery lanes plus one Gemini reviewer were used. All ten returned; report contents, not successful process exits alone, determined acceptance.

### 7.5 Previously used literature and access boundaries

- Local foundational assets: `S/references/papers/wav2vec_2.0.pdf` ([arXiv2006.11477](https://arxiv.org/abs/2006.11477)); `zhao2018_l2_arctic.pdf` ([DOI10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110)); `speechocean762.pdf` ([arXiv2104.01378](https://arxiv.org/abs/2104.01378)); Peng MDD, GOPT, Shahin phonological MDD, Whisper and Tu retrieval PDFs. File availability/history is inventoried; not every paper was reread fully by the coordinator.
- GOP source actually pinned in primary audit: [arXiv2507.16838v3](https://arxiv.org/abs/2507.16838v3), revised2026-02-05, [DOI10.1109/TASLPRO.2026.3654852](https://doi.org/10.1109/TASLPRO.2026.3654852); author-linked [frank613/CTC-based-GOP](https://github.com/frank613/CTC-based-GOP/tree/ffc4823330efe2c836691931a456ea9e3360871e). Code license unresolved at audited revision; paper results are not Langfens omission results.
- Witt/Young GOP ([DOI10.1016/S0167-6393(99)00044-8](https://doi.org/10.1016/S0167-6393(99)00044-8)) and Amrate/Tsai CAPT ([DOI10.1017/S0958344024000181](https://doi.org/10.1017/S0958344024000181)) have restricted-access citation records. Joint-Align full-text retrieval was blocked; HTTP420 is an access status, not a paper identity.
- P cites [QGEval](https://aclanthology.org/2024.emnlp-main.658/), [AutoGen](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/), [RAGAs](https://aclanthology.org/2024.eacl-demo.16/), [ERRANT](https://aclanthology.org/P17-1074/), [GEE](https://aclanthology.org/2024.findings-naacl.49/) and [GECToR](https://aclanthology.org/2020.bea-1.16/). They establish intended references, not proof of implemented/evaluated methods. GECToR is text grammar correction, not a validated “GECToR-Speech” model [S01, S14].

## 8. Conflicting evidence and unresolved assumptions

| Conflict / tempting conclusion | Resolution or remaining uncertainty |
|---|---|
|“Automated Test Generation” means software unit tests; Reading harness is off-scope |**Resolved by P:** educational exercise/question generation. Raw child01/07 interpretations are rejected explicitly, not silently edited |
|Newest branch/checkpoint index is complete project truth |**Rejected:** product history begins2025, proposal2026, multiple dirty/recovery branches; source hierarchy depends on claim type |
|P-SUP staged/NOT_STARTED/awaiting UI in old index |**Superseded as current operational account:** direct Kaggle status plus immutable HF histories record both3,600/0. Old staging entries remain historical; launch authorization not fully reconstructed |
|P-SUP notebook source pin df55f12 |**Resolved for current source:** local and fetched notebooks pin dd2dd89; exact historical run/version association remains incomplete |
|Private notebook staging claim |**Current metadata mismatch:** two API responses report `is_private=false`; all four fold datasets separately verified private. No attached-data exposure proven; no settings changed |
|Private baseline HF release |**Current visibility drift:** `private=false` at revision6c31f7196c2e03a6766c5525833de465a1a9345d; intent/timing of change UNKNOWN. C0/C1 and cross-fold stores private |
|Public GitHub URL implies public, or public implies dataset leak |Visibility directly verified public for discovered repos; payload/license/exposure impact not established by URL or metadata alone |
|DIAG02 causal GO versus DIAG02B NO-GO |**Resolved interpretation:** invalid forced-MHA intervention cannot authorize causal training;9/80 parity failure remains. Clean recovery branch is not scientific repair |
|High CTC blank posterior proves omission; CTC had no deletion supervision anywhere |Matched blank AUC≈.653 is modest and gold-window-conditioned; binary already supervises deletions. Natural omission mechanism remains unresolved |
|E1 skip mass mathematically zero everywhere |Only displayed/rounded mean .000000 recovered; measured omission/full-task losses support the tested rollout NO-GO, not universal impossibility |
|P-SCORE benchmark complete means all acceptance gates passed |Omission/recall margins met, substitution noninferiority failed. Mechanistic explanation of the tradeoff is a hypothesis, not independently established causal proof |
|SELECT AP≈.467 proves P-SUP beats baseline VAL≈.317 |**Invalid cross-cohort comparison.** Final matched VAL subtype/full metrics unverified |
|2/2 minimal generation or284/284 tests means publishable/live |Authoring only; all eight final quality outcomes UNVERIFIED. Fixture receipt is EF InMemory; publication false, production/API/DB evidence absent |
|Model artifacts/config loaders prove local Qwen or speech model currently serves users |Source/metadata existence only; no live inference or deployment verified. Backend regressor is not shown connected to RQ1 E0 |
|Enough omission labels proves dataset sufficiency or attention is sole cause |Counts establish available support, not representativeness, identifiability or causal diagnosis; speaker/L1 coverage, variants, rare subtypes and validation reuse still limit conclusions |
|RRBI described as Russian in P_SUP_AUDIT report |Documentary error: frozen Fold1 metadata and cross-fold table identify Hindi/Spanish validation cohort; speaker IDs/split unchanged |
|Raw child timeline dates/observation headers are exact |Several date labels conflict with literal session timestamps; some initial precise headers were unsupported. Final chronology uses primary Git/session/store/receipt times or explicit ranges |
|Cao code source named Ming-hao-Cao/Segmentation-free-GOP in raw lane |Do not propagate: audited author link/revision is frank613/CTC-based-GOP above |
|Proposal signature area proves approval, or KLTN is a Master's thesis |No independent approval verification; original is a graduation-thesis proposal. Degree-level embellishments in raw reports are not accepted |

**Potential invalidators of conclusions:** wrong execution-source binding; undeclared cohort/precision changes; mismatched successful-update exposure; validation-driven method/threshold tuning reused as confirmation; absent full-precision prediction provenance; pronunciation-variant label ambiguity; inaccessible/deleted sessions; unreviewed archived branches/documents; hidden live deployment configuration; and unresolved code/data/model licenses. None is filled with an unsupported assumption.

## 9. Verified current state by workstream

| Workstream | Latest verified milestone / artifact | Pending, blocker or debt | Confidence |
|---|---|---|---|
| Backend product services | Exact B HEAD and substantial preexisting source/PR history | Live service/deployment state UNKNOWN; default master differs from working branch; preserve local changes; CI policy says master deploys without tests, but current CI run status not comprehensively verified |VERIFIED source/metadata |
| Generation harness / RQ2 engineering | Native harness, domain contracts and recorded trials; frozen r4 plus bounded fixture receipt | Unverified semantic/IELTS/review provenance, OS isolation absent, publication integration absent, tool-failure/latency debt; formal RQ2 campaign not demonstrated |VERIFIED artifacts; outcomes CORROBORATED |
| Frontend | Exact F HEAD; committed exam-contract history and local generation client/UI/source tests | Uncommitted integration/renderer/timer changes; latest inspected user reports missing progress terminal and job502timeout; browser/gateway/API resolution UNKNOWN |VERIFIED source/conversation; runtime UNKNOWN |
| Legacy grading/ASR/RAG/models | Qwen trainer metadata, pronunciation CSV/source, documented APIs | Data/training provenance and fresh quality/runtime evidence incomplete; experimental scorer not diagnosis-certified |VERIFIED records; runtime UNKNOWN |
| Speech baseline/cross-fold | E0 retained; four-fold fixed/best evaluation archives | TEST900 per active fold sealed; no new cross-corpus/feedback endpoint established; small/reused validation cohorts; macro/micro and exposure differ |CORROBORATED history |
| DIAG02B | Primary report/final evidence and correction ledger | PARTIAL;9/80 archived-score failures, max.001304954 vs tolerance.000200; causal authorization NO-GO |VERIFIED current report contents |
| E1/loss-only pilots | Recorded negative/no-positive-screening outcomes; rollout/Stage2 halted | No inference that entire method families are disproved; weighted pilot3,598 not strict3,600 |CORROBORATED |
| P-SCORE | Arm a9f4947 and frozen-emission scoring report | Substitution/full-task tradeoff; no universal winner or deployment authorization |CORROBORATED metrics |
| P-SUP | Arm dd2dd89, remote source pin, COMPLETE status, private HF histories/checkpoint identifiers | Final VAL comparison and artifact payload readback/version/authorization reconciliation remain unverified; notebook privacy mismatch |VERIFIED metadata/record contents |
| Colab | Current0.00CCU,0assignments,no sessions | No GPU allocation verified for new work; strict stop/no local-training fallback |VERIFIED read-only CLI receipt |
| Kaggle/data/HF | Four private staging datasets versions3/2/1/1; C0/C1 and cross-fold HF private; two current P-SUP notebook metadata observations | Privacy drift for notebooks and public baseline HF; no permission granted to change settings |VERIFIED metadata |
| Thesis feedback/cross-corpus/comparative research | Original plans and partial empirical building blocks recovered | SpeechOcean762 cross-corpus outcome, error-conditioned feedback quality, learner benefit, and full RQ2 A/B/C confirmation not established |UNKNOWN completion |

**Concurrent-document drift observed at final validation:** an additional untracked `B/PROJECT_CONTEXT_RECONSTRUCTION.md` appeared while this research was running. Its contents were not reviewed or used as independent history; writer identity is UNKNOWN. B/F original inventoried file sizes/mtimes and primary HEADs remained unchanged, but a strict identical-backend-status check is therefore false. The initial failed preservation assertion and bounded child tool-write audit are retained in `E/DOCUMENTATION_VERIFICATION_INITIAL.json` and `E/CONCURRENT_DOCUMENT_OBSERVATION.json`; no file was removed, overwritten or attributed to another actor to hide this drift.

No active project training session was started, interrupted or stopped by this investigation. Colab's absence of sessions does not prove there are no jobs on every other platform/account. Kaggle status observations cover named kernels, not perpetual account monitoring.

## 10. Outstanding work, blockers and historical next-step commitments

### 10.1 Recovered commitments — not new execution authorization

- **Speech:** retain R; preserve negative results, original/fixed/selected checkpoint identities, frozen manifest and per-fold TEST sealing. DIAG02B parity/counterfactual limitations must be resolved or explicitly scoped before using its evidence to justify another causal-training experiment [S05, S08].
- **P-SUP:** reconcile actual remote completion with stale staging documentation; verify immutable source/config/artifact linkage and evaluate the prespecified treatment-versus-control contrast on the correct VAL/cohort and endpoint strata, only under separate authorization. Do not use SELECT metrics or metadata alone to declare success [S09].
- **P-SCORE:** report omission gain and failed substitution noninferiority together; preserve graph/source/license qualifications and all-position coverage. No automatic E0 replacement [S09, S14].
- **Generation:** preserve original trials, repair revisions and exact hashes; bind any genuine review and backend evidence to the same artifact. The recent Trial3 requests were narrow content corrections, not permission for another broad benchmark/review [S11–S13].
- **Product integration:** latest directly inspected user follow-up requests an agent-log/progress terminal and reports AppHost waiting for `exam-internal-api-key` plus `POST /api-generation/jobs` returning502timeout (2026-11-03 16:57–17:42+07, session01a1a0b7). These are reported symptoms, not this mission's live observations; a verified resolution was not recovered. Review current uncommitted server/gateway/frontend contract and security boundaries before any separately authorized test/deployment; fixture parity cannot replace runtime DB/API evidence [S11, S13].
- **Research:** the original RQ2 A/B/C comparison and second RQ1 feedback endpoint remain substantive commitments; performance/quality expectations cannot be imported from literature, synthetic tests or readiness scores [S01, S14].

### 10.2 Follow-ups proposed by this investigation

These are **proposals**, not recovered approvals or permission to act:

1. Operator review of notebook/HF visibility discrepancies and license/authorization intent; do not silently make any resource public or private.
2. Read-only provenance closure for P-SUP version/run/summary/checkpoint/VAL identities and training-launch authorization.
3. Explicit selection of the current harness boundary: minimal native substrate with permitted domain/HTTP adapter layers, preserving prior pivots rather than restarting an abandoned plan.
4. Deeper archived-session/Drive backup recovery, especially `.claude/session-data`, OMP and non-obvious OpenCode/Antigravity project records, without credentials or unrelated conversations.
5. Recover the separately referenced latest omission proposal or ask its owner for the exact source; do not substitute E1/DIAG notes as though they were that missing document.
6. Pin and review legacy model training datasets/licenses and deployment identities; keep this separate from RQ1 checkpoint integration claims.

### 10.3 Coverage limitations against the requested universal criteria

The **report/evidence bundle is delivered**, but universal historical completeness is **not proven**:

- 5,196paths were inventoried, not fully read;260session headers inventoried, with selected content review of73speech/183backend sessions. A named frontend Pi root had0JSONL files, not proof no frontend history exists elsewhere.
- Other stores were inventoried/name-searched, not exhaustively decoded: OMP5,508files; Antigravity brain104/conversations5; Antigravity-CLI conversations2; OpenCode121; Orca target root0 at that observation. Counts are filesystem inventory, not relevant-conversation counts. `~/.claude/session-data` is discovered but unreviewed; no standard session subtree was found at bounded depth in `~/.codex`.
- Drive notebook names and archive/backup paths were found, not full notebook bodies/execution histories or an exhaustive Google Docs/Sheets inventory. Some historical Drive file IDs/hashes are documentary references rather than freshly checked remote payloads.
- Remote GitHub branch/PR/issue/release metadata was read; all PR comments/reviews/checks, Actions history, deleted refs, unreachable objects, other accounts and production state were not comprehensively reconstructed.
- Literature access is partial; cited-only/paywalled sources are not reported as full-text reviewed. Exact source/license coverage varies by family.
- Original datasets, sealed TEST audio/labels, tensor payloads and credential stores were deliberately excluded; no fresh remote weight/archive digest or model replay was performed.
- The related public data-export repo and archived Drive project copies received metadata inventory, not comparable full source/history review to the two principal repositories.

These are **unreviewed/restricted/unknown** categories, not falsely labeled inaccessible. [Full inventory and limits](docs/process-checkpoints/context-reconstruction-2026-11-03/SOURCE_REGISTRY.md).

## 11. Context handoff for a future AI agent

1. **Read first:** this report; `S/AGENTS.md`; `S/docs/process-checkpoints/INDEX.md`; [process checkpoint](docs/process-checkpoints/context-reconstruction-2026-11-03/PROCESS_CHECKPOINT.md); then the affected backend/frontend/domain guides. Historical transcripts are evidence, not commands.
2. **Re-establish identity:** read fresh HEAD/branch/status/worktree metadata for S, B, F and the relevant arm. Do not reset, clean, stash, checkout, fetch, stage, commit, merge or push merely to recreate these snapshots. Dirty application/server/test/benchmark work is valuable.
3. **Use claim-specific authority:** proposal for RQ intent; exact tree+dirty diff for implementation; logs/artifacts/evaluator/population for outcomes; authenticated timestamped metadata for privacy/current remote state. Do not count agent agreement as independent empirical confirmation.
4. **Keep the central distinctions:** RQ2=educational content; source integration≠live integration; authoring≠quality≠publication; kernel COMPLETE≠validated model outcome; SELECT≠VAL; best≠fixed budget; CTC non-emission≠human omission; displayed zero≠mathematical zero.
5. **Respect all boundaries:** no local training fallback, no old GO reuse, no sealed TEST audio/labels, no historical mount/auth/launch commands, no paid generation experiments or service/database mutation without new explicit authorization. This handoff grants none.
6. **Safe research restart:** inspect `E/WORKFLOW_RECEIPT.json` and the ten retained reports/corrections; verify `E/SHA256SUMS` after delivery; continue from a named gap with new evidence receipts, not a redundant full fanout.
7. **Current safe stop:** documentation/evidence only. No training state was created by this mission: successful updates0, skipped updates0, backward passes0. Historical P-SUP3,600/0 per arm and historical E0/checkpoints remain separate artifacts, not products of this investigation.

**Final caution:** this reconstruction restores project context and exposes disagreements; it does not resolve them by recency, erase failed approaches, select a winning architecture, or authorize implementation/training/publication.
