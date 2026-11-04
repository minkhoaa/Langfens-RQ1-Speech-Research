# Coordinator direct verification notes

## C-01 — RQ2 means educational exercise/test generation, NOT software-test generation

- Confidence: VERIFIED for original intended scope from primary proposal; later actual implementation coverage remains separately assessed.
- Primary source: `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, line 14 (very long Markdown table cell), sections 2.2(e), 2.3 RQ2, 3(b), 4(e), 4(g).
- Original Vietnamese RQ2 explicitly asks whether specialized agents coordinating generation, review, validation and correction improve *learning-content quality* over single-LLM and sequential self-review, with time/cost tradeoffs.
- Proposed representative tasks: passage-grounded reading comprehension questions and grammar exercises with reference answers; Coordinator/Generator/Reviewer/Validator, bounded repair, independent solving blinded to generator answer.
- Planned comparison: A Single LLM, B Single LLM with Sequential Self-Review, C Proposed Multi-Agent; initially approximately 25–40 requests per representative exercise group, retaining failures in denominators.
- Parent directly read full line 14 and proposal header on 2026-11-03 during this investigation. Title is `Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation`.
- Interpretation of English `Test Generation` as *software-test generation* is unsupported and contradicted by original proposal. Root AGENTS scope sentence alone cannot settle translation; proposal does.
- Child 07 integration report incorrectly inferred a separate RQ2 software-test workstream and asserted IELTS Reading harness is not RQ2. Preserve that report as raw research evidence, but DO NOT propagate that interpretation into synthesis. Reading harness is scope-aligned with RQ2; formal comparative research completion is still not established.
- Do not silently edit child history. This explicit correction is authoritative for synthesis because it uses the original intent source.

## C-02 — Manifest integrity

Computed by parent (metadata/text only, no sealed audio/labels):
- `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- Matching historical digest does not validate experimental metrics or current remote artifact existence.

## C-04 — Parent authenticated Hugging Face metadata resolves part of P-SUP gap

- Existing CLI authentication was verified without exposing credentials (`hf auth whoami`); `HfApi.model_info(files_metadata=True)` and authenticated GET of small JSON at immutable revision were then used. No weights/datasets downloaded or loaded; no remote mutation.
- Receipts: `HF_METADATA_OBSERVATION.json` and `P_SUP_REMOTE_METADATA.json` in this directory (exact observation timestamps embedded).
- `minkhoaa/langfens-rq1-cross-fold-checkpoints` is private, revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`, with 63 files at observation. BOTH `psup/b_control/` and `psup/t_op/` have best, last and fixed-budget checkpoint file metadata plus history/training_summary JSON.
- Each remote training summary reports successful_updates=3600, skipped_updates=0, attempts=3600, fixed_budget_saved=true; objective binary_collapsed vs operation_ce. History has eight epochs of 450 updates each (FIT=1,799, not baseline's 600-update epochs).
- Remote selection AP (NOT final Fold-1 VAL comparison): control best 0.46760702566135065 at 2,250 updates; treatment best 0.46497542138006975 at 2,700. Do not compare these SELECT values against baseline's VAL AP or use them to declare treatment superiority. Final VAL subtype/AP/recall/noninferiority results remain unverified.
- Current source notebooks pin dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f, not older documented df55f125. Remote model metadata does not itself prove Kaggle notebook source/version or authorization.
- Baseline HF repo `minkhoaa/langfens-rq1-fold1-baseline-r` currently returns private=false at revision `6c31f7196c2e03a6766c5525833de465a1a9345d`; old reports call it private. Flag visibility drift without changing it or asserting a dataset leak. C0/C1 and cross-fold stores currently private.
- Child05's anonymous-HF failure and P-SUP unknown outcome reflect that lane's method/time, not universal lack of access. Preserve the child report; use parent receipts to update synthesis.

## C-05 — Raw child session timeline contains date-normalization errors

- Child01's dated table has rows labelled Oct30 for sessions whose literal filenames begin Nov02/Nov03, and a row labelled Oct31 that embeds an actual Oct27 harness session. These are synthesis/date-label errors, not historical events.
- Prefer exact session filename/message timestamp over that table's date label. Parent final timeline must normalize directly from the referenced timestamp or explicitly use a date range/UNKNOWN.
- Parent directly verified main Git root commits: backend init `62072d506164eb2e7b7b4bac5c5022acd6fb38b5` at 2025-09-15T02:54:06+07:00; frontend init `b7403a454ab2cbc47015e6097a8e42225031f245` at 2025-09-16T13:23:29+07:00; speech local root `085bfcd4e935ed708722cd161dd1892ae16eb3d8` at 2026-10-23T15:01:33+07:00. Extra root objects titled untracked files are stash-history artifacts, not separate project inception.

## C-06 — Direct GitHub metadata closes lane access omissions

- `GITHUB_METADATA_OBSERVATION.json` preserves GET metadata, current branches, PRs (merged_at + merge_commit_sha), non-PR issues and releases for all three repositories.
- At observation all three repositories are public. Speech default main / 44 remote branches / 3 PRs (one open); backend default master / 6 branches / 14 PRs; frontend default master / 6 branches / 44 PRs. No non-PR issues or releases returned by these endpoints (one page of 100; all returned counts <100).
- Current branch tips match initial primary HEADs. Speech public-origin privacy concern is now directly observed visibility, not inferred from URL. No restricted-data leak established and no remote settings changed.

## C-07 — Kaggle current source and metadata, and remote persistence time

- Parent official read-only `KaggleApi.kernels_status` + `kernels_pull(metadata=True)` retrieved only small notebook source/metadata into a temporary research directory, not outputs/weights/datasets. Receipt `KAGGLE_METADATA_OBSERVATION.json`, observations 2026-11-03T23:49:30+07 and 23:49:32+07.
- BOTH remote notebooks currently pin `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` and contain `--remote-execution`; both status COMPLETE, no failureMessage; `NvidiaTeslaT4`, GPU enabled, Fold-1 staging dataset attached. Thus the local-vs-remote source-pin gap is resolved for the current fetched notebook source (not proof that every historical execution used that version).
- Both returned server metadata `is_private=false`, contradicting local deploy metadata and privacy policy. Parent checked CLI implementation `kaggle/api/kaggle_api_extended.py:7442–7463`: exported is_private is directly response.metadata.is_private. This is API-reported notebook visibility, not an audit of attached dataset accessibility or proof of raw-data exposure. No privacy settings changed.
- HF commit metadata in `HF_SOURCE_REGISTRY.json` timestamps current control artifact persistence to 2026-11-03T23:08:03+07 (UTC16:08:03), including training summary at 23:07:53+07. This predates mission discovery, and is artifact commit time, not exact training event time. Only latest 22 HF commits are recorded; earlier treatment upload history not exhaustively reconstructed.

## C-08 — Historical backend model lineage is distinct and incompletely documented

- Direct primary metadata inspected: `models/qwen25-lora/README.md`, `adapter_config.json` references from search, `checkpoint-766/trainer_state.json` header and final section, `models/pronunciation-scorer-v2/training_history.csv`, and `services/ai-service/app/services/pronunciation_scorer.py:18–39,64–82`.
- Qwen SFT/LoRA uses `Qwen/Qwen2.5-1.5B-Instruct`; state says global_step=max_steps=766, epoch1, train_batch_size1, best_metric=null, best checkpoint=null. Training-loss log is not a held-out grading-quality evaluation. Platform, original dataset, timestamp and run authorization were not recovered from these records.
- Backend pronunciation scorer is Wav2Vec2Model with mean pooling and four-output regressor (pronunciation/accent/fluency/prosody), not RQ1 phoneme-level CTC+binary architecture. Five-epoch CSV has minimum val_loss 0.004361358 at epoch2; it is a local historical record, not fresh verified predictive accuracy. Do not equate it to RQ1 E0 or claim acoustic diagnosis readiness.
- Backend README:230–305 contains dictionary/ASR/embedding/LoRA latency and n=5 RAG pilot claims; treat them as documented historical claims pending raw benchmark verification, not current operational PASS. No model was loaded here.

## C-09 — Parent direct generation-evidence checks constrain child03 overclaims

- Read `services/generation-harness/benchmarks/final-evidence-wiring/actual-completion.json`: overallStatus=UNVERIFIED, publicationReady=false, publicationIntegration=NOT_IMPLEMENTED; structure and backendFixtureParity PASS are historical artifact statuses, semantic/IELTS/reviewProvenance/backendRuntimeEvidence UNVERIFIED. Receipt says unsigned local .NET execution, 284/284 tests, 11 r4-specific, EF InMemory only. No tests rerun now.
- Frozen r4 artifact SHA `4958e156c7e4b0183f55633dedccbab2807166d0fd4f322cdd424303079c6f54`; delivery projection `c2cc490d88ef72ee8fac9f57b63f098ad7e922552eeeb31c4b05021334960932` from that manifest (parent hashes still to be checked separately).
- Read `BENCHMARK_MINIMAX_THINKING.md`: 8 runs / 2 per profile. Minimal2/2 and high2/2 refer to AUTHORING only; off1/2 and medium0/2. The same report's quality table lists ALL eight final outcomes UNVERIFIED. Do not propagate child03's 'publication-grade', '100% authentic papers', 'definitive benchmark', or actual live LoRA functionality claims. Source integration/trainer metadata are not live deployment proof.
- Directly read computed metrics JSON headers: Trial3 294 turns / 18,865,628 total tokens (including17,141,220 cacheRead) /47 tool failures /modeled cost1.8609801; Cool Islands459 turns /33,773,831 total (31,229,239 cacheRead) /74 failures /modeled cost3.19117014. These are file-recorded aggregates, not independently recomputed or provider billing attestations. Later specialist agreement is artifact-specific and cannot upgrade unrelated r4 completion manifest.
- Current mission's output must not label the thesis a Master's thesis; original proposal is a KLTN graduation-thesis proposal, not evidence of postgraduate degree level.
- Child06's claims that GOP failure is eliminated, dataset census proves full sufficiency, omission failure is definitely architectural, or ProtocolZ proves deployment readiness exceed the measured evidence. Retain as hypotheses/rationales, not verified causal conclusions.

## C-10 — Additional discovered data repository and Kaggle registry

- `RELATED_SOURCE_DISCOVERY.json`: GitHub account repository listing discovered public `minkhoaa/langfens-ielts-dataset`, main `fcc18c97eee4b93a25ff464fe912399ce1c98e99`, created2026-03-16T06:29:31Z. Description connects Phase2 processed essays/grammar/speaking and RAG/fine-tuning exports. Root path metadata inventoried only; data payloads NOT opened; licensing and exact mapping to local Qwen training UNKNOWN. No README content returned by endpoint; do not claim readme reviewed.
- Official Kaggle `kernels list --mine --page-size200 --formatjson` returned31 items,29 Langfens-related; datasets list returned4 matching fold1–4 staging packages. Related source receipt lists slugs/titles/raw platform times. Non-project2 kernels not retained. Counts<page size, no apparent truncation. API timezone-naive lastRunTime fields must be labelled platform raw/assumedUTC unless verified; do not silently equate with exact execution boundaries.

## C-11 — Dataset privacy verified separately from notebook privacy

- `KAGGLE_DATASET_METADATA.json`: official SDK metadata GET verifies all four staging datasets currently private, versions Fold1=3, Fold2=2, Fold3=1, Fold4=1; each metadata license CC BY-NC4.0. No dataset payload/list of data files accessed.
- Thus P-SUP notebook `is_private=false` must NOT be reported as proof that attached raw datasets are public. Privacy mismatch is specific to notebook metadata; public baseline-HF model visibility is a separate direct observation.

## C-12 — Bibliographic and cohort-detail corrections from primary records

- Primary source audit `manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness/SHARED_GOP_PRIMARY_SOURCE_AUDIT.md` identifies segmentation-free GOP as arXiv2507.16838v3 (revised2026-02-05), DOI10.1109/TASLPRO.2026.3654852, author-linked code `frank613/CTC-based-GOP` at `ffc4823330efe2c836691931a456ea9e3360871e`. Raw child06's Ming-hao-Cao/Segmentation-free-GOP link is not the verified source identity here; do not propagate it.
- The unrestricted independent `ctc-union-language-v1` differs from author's neighbor-restricted graph and occupancy approximation; no exact author reproduction or verified author-code reuse license established.
- `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md` section1.1 labels RRBI Russian. Parent read frozen split manifest lines3390–3410: Fold1VAL L1 distribution Hindi1/Spanish1; existing four_fold_metrics_summary.csv also identifies RRBI Hindi. Treat Russian label as documentary error, not a change in speaker identity/split. No raw audio/labels opened.

## C-13 — Final reviewer adjudication; no automatic acceptance of reviewer prose

- Workflow completed with10 successful child returns, not10 verified experiments. Reviewer09 checked18 platform claims; reviewer10 supplied12 material research checks. Read their matrices as evidence reviews, not authority.
- Reviewer10 repeats `CTC excludes deletions, providing zero loss gradient on omissions`: narrow this to no explicit canonical-deletion target in realized-phone CTC. Binary already supervises deletions as errors; acoustic frames/parameters can still receive gradients from the total objective. Closure source/data audit explicitly warns against claiming deletion supervision was absent.
- Reviewer10/raw06 calls E1 skip mass exactly zero. Original closure states rounded reported mean `.000000`, not proof every token is mathematically zero. Use displayed mean and measured AP loss only.
- Reviewer10 says P-SUP final VAL evaluation is unperformed. Evidence only establishes no final VAL comparison verified in sources examined; absence of all such evaluations is not proven.
- Review matrices do not establish fresh Drive/HF payload readback. Current model artifact LFS identities are server metadata only; bit-identical tensor replay remains historical evidence. Some reviewer date/approval prose also overstates source precision; final synthesis uses exact direct receipts and labels proposal date, not independently verified signature/approval.
- Research subagent model calls were explicitly requested/authorized. Claims of no paid/live calls mean no new PROJECT experiment/benchmark/provider generation beyond those research-agent calls, not zero LLM usage for this investigation.

## C-14 — Final primary-history and latest-intent corrections

- Direct Git path history places Qwen model inclusion at `1cd4ec7711279bc19b676e62198e4ebdb24d2a98` / duplicated lineage `23266fbdf7b69d2bca998e439e81a7e4b6a30d2e`, 2026-05-31T13:44:50+07, not child03's September7. This is Git inclusion, not verified training time. Final timeline corrected.
- Directly inspected user messages in `2026-11-03T04-44-06-176Z_01a19fb8-baa0-74e7-a4d2-9073fb4eda8b.jsonl`: at04:44:48.817Z (11:44:48.817+07), only two Trial3 content defects were authorized to fix, preserving originals/new revision, targeted existing validation, no new benchmark/AI reviewers. Nearby01a19fb2 session is adversarial review, not that exact correction request.
- Directly inspected `2026-11-03T09-22-35-196Z_01a1a0b7-b039-74e7-a4d2-907562679a13.jsonl`: user requests progress/log terminal at16:57:06+07, reports terminal missing17:01, AppHost waiting on exam-internal-api-key17:38:54, and job POST502timeout17:42:44. These are historical user-reported symptoms, not current reproduced failure or verified fix. Final report records them as latest inspected operational commitments.

## C-03 — Source inventory accounting

`SOURCE_FILE_INVENTORY.tsv` inventories tracked + nonignored untracked files in the three primary repositories (5,196 rows at capture), NOT all ignored data/cache/model/session files and NOT full content review of every row. `SOURCE_DISCOVERY_SNAPSHOT.json` captures Git metadata without fetching; remote-tracking refs can be stale. This process's own documentation is incidental in speech untracked status and must not be mistaken for pre-existing changes.
