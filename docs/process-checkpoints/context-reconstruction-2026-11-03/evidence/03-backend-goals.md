# Forensic Reconstruction Report: Backend Project Documentation, Autonomous-Generation, and RQ2 Intent

**Repository Under Review:** `Project_Langfens_Microservice` (HEAD `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`, branch `generation-harness`)  
**Context Worktrees / Associated Repos:**
- Dataset / Coordinator: `/home/khoa/Projects/KLTN/Dataset` (HEAD `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`, branch `prep/rq1-shared-base`)
- Associated Frontend: `/home/khoa/Projects/langfens/langfens-fe-app` (HEAD `2d9fe94a54e65680eca56aafbe40e691018113da`, branch `autonomous-generation`)
- Isolated Reading SDD Worktree: `/home/khoa/Projects/langfens/worktrees/langfens-reading-sdd-20261029` (HEAD `93d585a5ab4e8bce75215d0b2099a36242b578d8`)  
**Investigator Role:** Independent Read-Only Forensic Researcher (SEAM 3: Backend Goals & Autonomous Generation)  
**Timestamp:** 2026-11-03T23:55:00+07:00 (Asia/Ho_Chi_Minh)  
**Primary Thesis Scope Clarification (VERIFIED):** Per original Master's thesis proposal `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md` (Line 14, sections 2.2(e), 2.3 RQ2, 3(b), 4(e)), **RQ2 is strictly defined as Automated Learning-Content / Educational Test Generation** (IELTS Reading comprehension passages, questions, and grammar exercises with reference answers), **NOT software engineering test generation**. The `generation-harness` subsystem is the primary technical implementation aligned with RQ2.

---

## 1. Scope & Primary Source Inventory

| Artifact / Path | Format / Size | State & Revision | Direct Evidence & Role |
|:---|:---|:---|:---|
| `/home/khoa/Projects/KLTN/DC KLTN0127 23520772...md:14` | Markdown (68 KB) | Historical Proposal | Primary definition of RQ2: Multi-agent generation of educational tests, blinded review, bounded repair. |
| `Project_Langfens_Microservice/README.md` | Markdown (333 lines) | Tracked | Monolithic .NET 10 + FastAPI microservice ecosystem, Qdrant RAG, Qwen2.5-LoRA, Wav2Vec2/Whisper. |
| `Project_Langfens_Microservice/AGENTS.md` | Markdown (40 lines) | Tracked | Microservice boundary rules, Aspire orchestration, database per service, CI no-test warnings. |
| `docs/superpowers/specs/2026-10-27-langfens-harness-design.md` | Markdown (180 lines) | Tracked | Initial design specification: refactor Pi 1.4.0 runtime + pi-subagents 0.83.0 into Langfens Harness. |
| `docs/superpowers/plans/2026-10-27-langfens-harness.md` | Markdown (245 lines) | Tracked | 11-task SDD implementation plan on branch `autonomous-generation-v2`. |
| `.superpowers/sdd/2026-10-27-langfens-harness/progress.md` | Markdown (132 lines) | Tracked | SDD execution ledger: Tasks 1–5 completed (commit `ee44064`); Task 6 split into 6A/6B/6C. |
| `.superpowers/sdd/2026-10-27-langfens-harness/scope-incident-6a/` | Directory (16 files) | Untracked | Recorded failure on Task 6A: build/typecheck breakage leading to workflow halt on 2026-10-27. |
| `services/generation-harness/README.md` | Markdown (138 lines) | Tracked | Standalone Node.js harness architecture, 4 core tools (`read`, `bash`, `edit`, `write`), offline safety. |
| `services/generation-harness/.generation/reading/README.md` | Markdown (140 lines) | Tracked | Reading domain extension: 13 native tools (7 validators + 6 builders), 4 specialist roles, 4 skills. |
| `services/generation-harness/.generation/reading/SYSTEM.md` | Markdown (37 lines) | Tracked (Modified) | Authoritative Exam Designer prompt: orchestrates passage-author, question-author, semantic, examiner. |
| `services/generation-harness/.generation/reading/BLIND_REVIEW_WORKFLOW.md` | Markdown (97 lines) | Untracked | Protocol for blind solving: public-first derivation before opening private keys, SHA-256 bindings. |
| `services/generation-harness/.generation/reading/NATIVE_RELIABILITY.md` | Markdown (108 lines) | Untracked | Forensic replay audit of 790 tool calls across 74 JSONL sessions, MiniMax tool formatting diagnosis. |
| `services/generation-harness/BENCHMARK_MINIMAX_THINKING.md` | Markdown (173 lines) | Untracked | Thinking benchmark report for MiniMax-M2.7 across 8 runs (`off`, `minimal`, `medium`, `high`). |
| `services/generation-harness/benchmarks/metrics/` | 12 JSON files | Untracked | Machine metrics for 8 initial runs, 2 postfix runs, `trial-3-computed.json`, and `cool-islands-computed.json`. |
| `services/generation-harness/benchmarks/runs/` | 11 run directories | Untracked | Full session JSONL, transcripts, and workspaces for all experimental generation runs. |
| `services/generation-harness/benchmarks/final-evidence-wiring/` | 17 files | Untracked | Formal completion gate verification, `actual-completion.json`, .NET backend parity receipts. |
| `FIXES_PROPOSAL.md` | Markdown (377 lines) | Deleted in worktree | Multi-agent audit (commit `1f14aee`) identifying 3 P0, 12 P1, 8 P2 defects across backend and harness. |
| `models/qwen25-lora/` | safetensors + configs | Tracked | Qwen2.5-1.5B-Instruct PEFT LoRA adapter (r=16, alpha=32), trained via TRL SFT (checkpoint-766). |
| `services/ai-service/app/prompts/autogen.py` | Python (62 lines) | Tracked | Older monolithic Python autogen prompt builder (mirror of frontend `llmPromptBuilder.ts`). |
| Inaccessible / Prohibited Sources | Remote / Paid | N/A | Live LLM inference APIs, live database migrations, network fetches, remote cloud mutators. |

---

## 2. Chronological Forensic Event Log

| Event Time (Asia/Bangkok) | Source / Commit / Doc | Goal & Context | Architectural Decision & Rationale | Alternatives Considered | Outcome & Empirical Limitation | Current Validity |
|:---|:---|:---|:---|:---|:---|:---|
| **2026-09-07** 11:48 | `models/qwen25-lora/` commit `424b917` | Local IELTS grading adapter for Speaking/Writing | Fine-tune Qwen2.5-1.5B-Instruct using LoRA (`peft` 0.19.1, TRL 1.2.0, r=16, alpha=32) over 766 steps (1 epoch). | Rely solely on cloud models (Groq `qwen3.8-27b`) or unsloth. | Adapter saved (`adapter_model.safetensors`, 36.9 MB). High CPU latency (~68s) noted in README; used for local grading fallback. | **VALID** (Model artifact preserved) |
| **Pre-2026-10-27** | `services/ai-service/app/prompts/autogen.py` | Initial reading comprehension exercise generation | Monolithic single-turn prompt in FastAPI returning raw JSON schema of questions based on input passage. | Multi-agent collaboration. | Brittle parsing; frequent JSON syntax errors; no independent review or grounded verification gates. | **SUPERSEDED** by Generation Harness |
| **2026-10-27** 16:46 | `AGENTS.md` (root microservice) | Establish microservice boundaries | 11 independent services (.NET 10 + Python), MassTransit RabbitMQ, isolated Postgres per service. | Monolithic database or shared schema. | Strict boundary enforced; AI grading decoupled via asynchronous worker queues. | **VALID** (Core microservice stack) |
| **2026-10-27** 20:05 | `docs/superpowers/specs/2026-10-27-langfens-harness-design.md` | Create autonomous test generation subsystem (RQ2) | Vendor Pi runtime 1.4.0 (`4a08025`) and `pi-subagents` 0.83.0 into local Node.js `services/generation-harness`. | Python/FastAPI agent stack or LangChain/CrewAI. | Chose Node.js native loop with 4 tools (`read`, `bash`, `edit`, `write`), single/ordered delegation, JSONL RPC. | **VALID** (Subsystem architectural baseline) |
| **2026-10-27** 22:39 | Commit `ee44064` (`progress.md`) | Implement harness core and subagents | Completed SDD Tasks 1–5: package branding, runtime isolation (`.runtime/`), local core ownership, subagent extension. | External npm dependency. | 34 offline unit tests passed; verified local child session execution without host Pi dependencies. | **VALID** (Harness foundation verified) |
| **2026-10-27** 23:14 | `.superpowers/sdd/.../scope-incident-6a/` | Consolidate core modules and narrow exports (Task 6) | Split Task 6 into 6A/6B/6C due to model context limits; Task 6A failed typecheck/build (`src/core/tools/index.ts`). | Revert entire branch. | Workflow halted safely with `DONE_WITH_CONCERNS`; worktree dirt preserved; manual intervention required. | **HISTORICAL** (Scope incident documented) |
| **2026-10-29** 18:26 | Frontend commit `2d9fe94` (`langfens-fe-app`) | Update source of truth for IELTS Reading delivery | Align frontend exam contract with canonical schema (`question-types.v1.schema.json`). | Maintain separate frontend question types. | Contract tests updated (`vitest.reading-contract.config.ts`); admin generation UI scaffolded. | **VALID** (Contract parity enforced) |
| **2026-10-30** 14:22 | Commit `1f14aee` (`FIXES_PROPOSAL.md`) | Full-stack audit of Reading extension & backend | 5 independent Gemini subagents audited backend and harness, identifying 3 P0, 12 P1, 8 P2 defects. | Unaudited implementation. | Identified `P0-01` (all-or-nothing backend grading vs authentic IELTS marks) and `P1-04` (lack of OS sandboxing). | **VALID** (Master audit baseline) |
| **2026-10-31** 15:30 | `.generation/reading/STABILIZATION.md`, `PROGRESS.md` | Stabilize Reading domain extension and tools | Implemented 13 native tools (7 validators + 6 builders `create_*`), Exam Designer role, and 4 minimal skills. | Python-based tools. | 124 Reading tests + 87 harness tests passing. Implemented fixes P1-01, P1-03, P1-11, P1-12, P1-06 (SDK exception). | **VALID** (Stabilized extension) |
| **2026-10-31** 22:49 | `BENCHMARK_MINIMAX_THINKING.md` | Evaluate MiniMax-M2.7 thinking levels on 13Q IELTS generation | Benchmarked 8 runs across profiles: `off`, `minimal`, `medium`, `high` (2 runs each) on Mangrove ecology topic. | Single thinking budget. | `minimal` achieved 100% completion (2/2) in 420s ($0.253); `medium` timed out (0/2) due to high turn latencies (>11s). | **VALID** (Definitive benchmark evidence) |
| **2026-11-01** 02:12 | `services/generation-harness/.generation/reading/NATIVE_RELIABILITY.md` | Forensic analysis of MiniMax tool execution failures | Replayed 790 calls / 481 responses across 74 sessions; diagnosed tool schema mismatches and parallel args. | Assume model hallucination. | Found MiniMax emitted `{item: [...]}` for arrays; fixed parameter descriptions and schemas in `creation.ts`. | **VALID** (Root-cause tool repair) |
| **2026-11-01** 04:19 | `benchmarks/runs/postfix-minimal-run-2/` | Execute post-fix minimal thinking benchmark | Evaluated updated tool schemas on 13Q generation with full blind review and repair phases. | Manual inspection only. | Completed through Revision 4 (`exam.r4.json`), producing candidate for backend evidence binding. | **CORROBORATED** by run artifacts |
| **2026-11-01** 12:49 | `benchmarks/final-evidence-wiring/` | Bind offline .NET backend evidence to generation output | Created `record-backend-evidence.mjs` running 284 Attempt tests (11 r4 grading tests) against `exam.r4.json`. | Mock grading only. | Produced `actual-completion.json`: Structure PASS, Backend Parity PASS, Semantic/Review UNVERIFIED, Publication false. | **VALID** (Authoritative completion state) |
| **2026-11-02** 17:50 | `benchmarks/runs/trial-3/` (`trial-3-computed.json`) | Uncapped end-to-end trial (Trial 3) | Executed full multi-agent generation without artificial time or token caps using `MiniMax-M2.7:minimal`. | Bounded 480s timeout. | Ran 294 turns, 18.87M tokens ($1.86), 47 tool failures; produced `exam.corrected-r1.json`, halted at r2 reviews. | **CORROBORATED** (Raw metrics verified) |
| **2026-11-03** 09:59 | `benchmarks/metrics/cool-islands-computed.json` | Evaluation of "Cool Islands" generation session | Forensic metrics extraction from session `2026-11-01T11-30-04-623Z` across 4 revisions (Run 01–04). | Aggregate summary only. | 459 turns, 33.77M tokens ($3.19), 74 tool failures. Produced valid `review-semantic-rev3.json` and `review-examiner-rev3.json`. | **CORROBORATED** (Raw metrics verified) |

---

## 3. Architecture & Intent Evolution

```mermaid
flowchart TD
    subgraph Phase1_EarlyArchitecture [Phase 1: Microservice & Early Autogen (Pre-Oct 2026)]
        A1[FastAPI ai-service] -->|Single Prompt| A2[app/prompts/autogen.py]
        A2 -->|Raw JSON Output| A3[Brittle Frontend Import]
        A4[Qwen2.5-1.5B LoRA] -->|peft SFT| A5[Speaking/Writing Scoring]
    end

    subgraph Phase2_HarnessPivot [Phase 2: Subsystem Pivot (2026-10-27)]
        B1[Pi 1.4.0 Core + Subagents 0.83.0] -->|Refactor to Node.js| B2[Langfens Harness]
        B2 -->|Built-in Tools| B3[read, bash, edit, write]
        B2 -->|Subagent Tool| B4[Isolated Child Sessions]
    end

    subgraph Phase3_ReadingMultiAgent [Phase 3: Multi-Agent Reading Superpowers (2026-10-29 - 2026-10-31)]
        C1[Exam Designer / Parent] -->|Phase 1| C2[passage-author]
        C2 -->|Passage Ground Truth| C3[question-author]
        C3 -->|13 Canonical Builders| C4[exam.json]
        C4 -->|Phase 3 Blind Review| C5[semantic-checker]
        C4 -->|Phase 3 Blind Review| C6[ielts-examiner]
        C5 & C6 -->|Phase 4 Targeted Repair| C3
    end

    subgraph Phase4_VerificationWiring [Phase 4: Evidence Binding & Parity (2026-11-01 - 2026-11-03)]
        D1[exam.r4.json] -->|validate_exam pilot:true| D2[Canonical Gate PASS]
        D1 -->|record-backend-evidence.mjs| D3[.NET Attempt-Service Tests 284/284 PASS]
        D4[Blind Solvers Derivations] -->|Public-First Solving| D5[Review Provenance UNVERIFIED]
    end

    Phase1_EarlyArchitecture -.->|Replaced by| Phase2_HarnessPivot
    Phase2_HarnessPivot --> Phase3_ReadingMultiAgent
    Phase3_ReadingMultiAgent --> Phase4_VerificationWiring
```

### Evolution of Subsystems and Frameworks
1. **Initial Monolithic Python Autogen:** In `services/ai-service/app/prompts/autogen.py`, early generation relied on single LLM completions guided by stringified schemas mirrored from `langfens-fe-app/src/app/admin/_lib/llmPromptBuilder.ts`. This was ungrounded, prone to hallucination, and lacked systematic verification.
2. **The Langfens Harness Subsystem (2026-10-27):** Vendored from Pi 1.4.0 (`4a08025`) and `pi-subagents` 0.83.0. Rebranded and stripped of upstream telemetry, global fallback resolution, and unnecessary tools. Core was consolidated to 4 built-in tools (`read`, `bash`, `edit`, `write`), with delegation handled exclusively through the `subagent` extension.
3. **Reading Domain Extension & Multi-Agent Architecture (2026-10-29 to 2026-10-31):**
   - Transformed the authoring process into a 5-phase multi-agent workflow coordinated by an **Exam Designer** (Parent) using `using-reading-superpowers`.
   - Distinct roles: `passage-author` (academic text generation), `question-author` (canonical question synthesis via 6 `create_*` builders), `semantic-checker` (propositional entailment, factual accuracy, and UTF-16 evidence verification), and `ielts-examiner` (task authenticity, distractor quality, and scoring parity).
   - Enforced strict separation between student-facing content (`inspect_delivery_projection`) and private answer keys.
4. **Backend Evidence Binding & Verification Hardening (2026-11-01 to 2026-11-03):**
   - Implemented `record-backend-evidence.mjs` to execute real `.NET` test fixtures (`ReadingExamR4GradingTests.cs`, 284 passing tests) in `attempt-service.Tests` against generated candidate artifacts.
   - Enforced fail-closed completion gates: `actual-completion.json` refuses to elevate to `PASS` or `publicationReady: true` because independent review provenance and OS-level sandboxing remain unverified.

---

## 4. Intended vs. Implemented vs. Executed Quality Gates

| Gate ID | Target Quality Dimension | Intended Specification (Design Docs / SYSTEM.md) | Implemented Code Seam | Actually Executed in Benchmarks / Evidence | Final Status & Residual Risk |
|:---|:---|:---|:---|:---|:---|
| **GATE-01** | Canonical Hierarchy & Schema | Valid 4-tier Exam tree (`sections[].questionGroups[].questions[]`) matching `question-types.v1.schema.json`. | `validate_exam` in `hierarchy.ts:70`, `validate_question` in `question.ts`. | **EXECUTED** across all runs (`off`, `minimal`, `high`, `trial-3`, `cool-islands`). 0 schema errors. | **PASS** (Mechanical schema validity fully verified). |
| **GATE-02** | 13-Mark Pilot Compliance | Exactly 13 Questions, 13 answer slots, 13 scoring units. 1 backend Question = 1 all-or-nothing mark. | `hierarchy.ts:100` (`pilot: true` qualifier checks `PILOT_13_MARKS`). | **EXECUTED** in `minimal-run-1/2`, `high-run-1/2`, `postfix-2`, `trial-3`, and `cool-islands`. | **PASS** (13-mark constraint strictly enforced). |
| **GATE-03** | Public-First Blind Derivation | Reviewers derive candidate answers from public projection (`inspect_delivery_projection`) BEFORE reading private keys. | `BLIND_REVIEW_WORKFLOW.md`, `SYSTEM.md`, enforced via `printf %s` derivation persistence command. | **EXECUTED** in `postfix-minimal-run-2` (r2/r4) and `cool-islands` (Run-04 rev3). | **PASS** for transcript ordering; **UNVERIFIED** for cognitive independence. |
| **GATE-04** | Mechanical Evidence Spans | UTF-16 code unit offsets (`start`, `end`) match exact substring in passage text. | `verify_evidence` in `evidence.ts:15`. Rejects surrogate splits and offset drifts. | **EXECUTED** (87 calls in `trial-3`, 83 in `cool-islands`). Verified byte-exact quote matches. | **PASS** (Mechanically verified; semanticStatus remains UNVERIFIED by design). |
| **GATE-05** | Semantic Entailment & NOT GIVEN Proof | Factual truth, non-contradiction, and whole-passage absence reasoning for NOT GIVEN items. | `agents/semantic-checker.md`, `skills/reading-quality-review/SKILL.md`. | **EXECUTED** in `cool-islands` Run-04 (`review-semantic-rev3.json`: q1–q13 verified). | **PASS** within specialist scope; unverified by external human judge. |
| **GATE-06** | IELTS Authenticity & Scoring Parity | Authentic rubrics, Section 1 difficulty ceiling, valid distractors, no multi-blank/multi-slot conflicts. | `agents/ielts-examiner.md`, `question.ts:UNVERIFIED_SCORING_CONFLICT`. | **EXECUTED** in `cool-islands` Run-04 (`review-examiner-rev3.json`: keys agreed 13/13). | **PASS** with cosmetic notes (item difficulty calibration, duplicated instructions). |
| **GATE-07** | Backend InMemory Grading Parity | Generated exam entity grades authentically under .NET `attempt-service` grading engine. | `ReadingExamR4Fixture.cs`, `record-backend-evidence.mjs` executing `attempt-service.Tests`. | **EXECUTED** via offline test recorder; 284/284 tests passed (11 r4-specific tests). | **PASS** for EF Core InMemory fixture; **UNVERIFIED** for live PostgreSQL/API. |
| **GATE-08** | OS-Level Filesystem Sandboxing | Blind reviewers restricted from accessing parent filesystem, git history, or private keys out-of-band. | Documented in `P1-04` (`FIXES_PROPOSAL.md`) and `NATIVE_RELIABILITY.md`. | **NOT IMPLEMENTED**. Subagents share host user permissions; CWD isolation is not an OS sandbox. | **UNVERIFIED / NOT_ENFORCED** (Explicitly reported as limitation in completion manifests). |
| **GATE-09** | Production Publication Gate | Automated promotion to live exam catalog and database persistence. | Intended endpoint / workflow. | **NOT IMPLEMENTED** (`publicationReady: false`, `publicationIntegration: NOT_IMPLEMENTED`). | **FAIL-CLOSED** (Intentional safety invariant; unverified exams cannot publish). |

---

## 5. Failed and Inconclusive Experiments

1. **High / Medium Thinking Timeout Collapse (`BENCHMARK_MINIMAX_THINKING.md`):**
   - *Hypothesis:* Higher thinking budgets (`medium`: 8,192 tokens, `high`: 16,384 tokens) on `MiniMax-M2.7` would improve question quality and distractor subtlety.
   - *Empirical Result:* FAILED. Per-turn generation latencies rose to 11.24s (`medium`), accumulating 31,486 reasoning tokens. Because 13 questions are authored sequentially, both `medium` runs timed out at 480 seconds before completing Question Authoring (0/13 questions authored).
2. **MiniMax Tool Argument Serialization Mismatch (`NATIVE_RELIABILITY.md`):**
   - *Symptom:* When delegating via `subagent({ workflow: "parallel", args: { tasks: [...] } })`, MiniMax wrapped the array into `{ item: [...] }`.
   - *Impact:* Harness rejected calls with `args.tasks must be a non-empty array`. Parent spent 4–8 turns in guide-retry loops (~120–160s wasted).
   - *Resolution:* Repaired in `.generation/extensions/reading.ts` and `creation.ts` by adding explicit parameter examples and conditional schema projection.
3. **R3 Delivery Projection Recovery Failure (`bench/recover-r3-projection.mjs`):**
   - *Objective:* Reconstruct the exact byte-identical public delivery projection for historical Revision 3 (`693ed4b879cc...`) from `exam.corrected.json`.
   - *Result:* INCONCLUSIVE / FAILED. Sibling reconstructions differed in key serialization order and trailing whitespace. The script reported: `"Original bytes not recovered. No historical projection file has been created; semantically equivalent reconstruction is NOT historical evidence."`
4. **P0-01 Backend Scoring Asymmetry Blocker (`FIXES_PROPOSAL.md`):**
   - *Defect:* Backend `attempt-service` (`BuildQuestionIdSet.cs:178`) assigns exactly 1 point per Question entity, and `Grader.cs` grades completion and matching tables all-or-nothing.
   - *Consequence:* Authentic multi-slot questions (e.g. 5-blank summary) cannot award partial marks. Authoring tools are forced to constrain IELTS generation to 13 single-slot questions, severely limiting pedagogical task variety.

---

## 6. Claim and Evidence Registry

| Claim Identifier | Statement / Claim Under Audit | Evidentiary Basis & File References | Confidence | Forensic Verdict |
|:---|:---|:---|:---|:---|
| **CLM-01** | RQ2 is automated learning-content/question generation, not software test generation. | Primary Thesis Proposal `/home/khoa/Projects/KLTN/DC KLTN0127...md:14` (sections 2.2(e), 2.3 RQ2, 3(b), 4(e)). | **VERIFIED** | Authoritative primary truth. Child 07 interpretation of RQ2 as software-testing was erroneous. |
| **CLM-02** | Generation Harness achieves 100% completion of authentic 13Q Reading papers with MiniMax `minimal` thinking. | `BENCHMARK_MINIMAX_THINKING.md:20-35`, `minimal-run-1.json`, `minimal-run-2.json`. 896–899 word passages, 13 questions. | **VERIFIED** | Validated empirically on Mangrove ecology prompt. |
| **CLM-03** | Local Qwen2.5-1.5B LoRA adapter is deployed and functional for IELTS grading. | `models/qwen25-lora/adapter_config.json`, `checkpoint-766/trainer_state.json`, `README.md:220-230`. | **VERIFIED** | Trained over 766 steps, loss 1.32, integrated into `ai-service` via PEFT. |
| **CLM-04** | Dual-specialist blind reviews achieve complete agreement on candidate exam items. | `services/generation-harness/.runtime/reading/workspace/run-04/review-semantic-rev3.json`, `review-examiner-rev3.json`. | **VERIFIED** | Both reviewers derived answers from public view first; 13/13 keys agreed with zero material findings. |
| **CLM-05** | Generation harness runs with full OS-level sandbox isolation for child subagents. | `BLIND_REVIEW_WORKFLOW.md:85`, `NATIVE_RELIABILITY.md:100`, `actual-completion.json:35`. | **CONTRADICTED** | CWD isolation is NOT filesystem isolation; subagents have unconstrained access to host filesystem. |
| **CLM-06** | Generated exams are publication-ready for production IELTS candidates. | `actual-completion.json:3`, `PROGRESS.md:95`, `SYSTEM.md:30`. | **CONTRADICTED** | `publicationReady: false` and `publicationIntegration: NOT_IMPLEMENTED` are strictly enforced. |
| **CLM-07** | Generated exams achieve verified .NET backend runtime integration. | `benchmarks/final-evidence-wiring/actual-completion.json:11`, `backend-evidence.ts:25`. | **PARTIAL** | InMemory test fixture parity is PASS (284/284 tests); PostgreSQL, API Gateway, and frontend E2E remain UNVERIFIED. |

---

## 7. Contradictions & Unresolved Seams

1. **Thesis Proposal RQ2 vs. Local Agent Misconceptions:**
   - *Conflict:* Prior subagent notes hypothesized that RQ2 involved automating software engineering unit/integration tests for the microservices.
   - *Reconciliation:* Primary proposal Line 14 explicitly defines RQ2 as automated multi-agent generation of *learning content* (IELTS reading comprehension and grammar exercises). The Reading generation harness is the definitive RQ2 artifact.
2. **IELTS Authentic Multi-Blank Tasks vs. Backend 1-Point Scoring (`P0-01`):**
   - *Conflict:* IELTS Summary Completion and Matching tasks authentically contain multiple blanks/targets worth 1 mark each. `attempt-service` hardcodes 1 Question = 1 point all-or-nothing.
   - *Impact:* The generation harness is forced to artificially author 13 independent single-slot Question entities (e.g. 5 disconnected sentences instead of one cohesive summary paragraph) to prevent candidates from losing all marks on partial matches.
3. **Claimed Verification vs. Filesystem Blindness (`P1-04`):**
   - *Conflict:* Documentation states that reviews are "blind" independent evaluations.
   - *Reality:* The harness isolates working directories (`cwd`), but native tools (`read`, `bash`) execute with full user privileges. No container, chroot, or read-only mount prevents a child LLM from inspecting private artifacts via absolute paths.
4. **Offline Test Parity vs. Live Service Execution:**
   - *Conflict:* `attempt-service.Tests` passes 284/284 checks including `ReadingExamR4GradingTests.cs`.
   - *Reality:* This establishes compatibility only with the C# in-memory domain model and EF Core InMemory provider. It does not prove schema migrations, PostgreSQL triggers, YARP gateway routing, or frontend JSON parsing compatibility.

---

## 8. Unresolved Questions & Exact Safe Restart Instructions

### Unresolved Research & Engineering Questions
1. *Comparative RQ2 Empirical Campaign:* While individual generation runs (MiniMax `minimal`, `high`, `trial-3`, `cool-islands`) exist, has the formal 3-way comparative research campaign planned in thesis proposal Section 4(e) (Condition A: Single-LLM vs. Condition B: Sequential Self-Review vs. Condition C: Multi-Agent Harness across 25–40 prompt groups) been formally executed and analyzed? (Current Evidence: **UNPROVEN / INCOMPLETE**).
2. *P0-01 Backend Fix Disposition:* Will backend grading in `BuildQuestionIdSet.cs` and `Grader.cs` be updated to support fractional/per-slot scoring, or will the 1-Question-per-slot authoring constraint remain permanent for thesis submission?
3. *Frontend Projection Integration:* Has `langfens-fe-app` successfully rendered a generated 13Q exam from `exam.delivery.json` without layout regressions in `QuestionRenderer.tsx`?

### Safe Restart Protocol (Preserving All Pre-existing Dirt)
1. **Preserve Repository Dirt:** Under NO circumstances execute `git checkout -- .`, `git clean -fd`, or `git reset --hard` in `Project_Langfens_Microservice` or `langfens-fe-app`. Both working trees contain valuable uncommitted test fixtures, benchmarks, and configuration fixes.
2. **Environment Verification:**
   ```bash
   node -v # Must be >= 22.19.0
   dotnet --version # Must be .NET 10 SDK
   ```
3. **Execute Offline Test Verification (Read-Only):**
   ```bash
   cd /home/khoa/Projects/langfens/Project_Langfens_Microservice/services/generation-harness
   GENERATION_HARNESS_OFFLINE=1 npm test
   GENERATION_HARNESS_OFFLINE=1 node .generation/reading/bench/tool-replay.mjs
   ```
4. **Inspect Prior Generation Outputs:**
   Read completed artifacts in `services/generation-harness/benchmarks/runs/postfix-minimal-run-2/runtime/reading/workspace/` and `.runtime/reading/workspace/run-04/`.
5. **No Unauthorized Inference:** Do not trigger live API runs (`run-benchmark.mjs`, `run-trial-3.mjs`, or `npm run start:reading`) without explicit operator compute and financial authorization.
