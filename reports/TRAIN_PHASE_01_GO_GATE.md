# TRAIN-PHASE-01 GO gate — orchestration closure

## Verdict: GO for controlled-training infrastructure preflight

**All three previously listed engineering blockers are resolved.** This is readiness of the frozen controlled-run implementation—not a completed formal run, deployment/safety approval, or automatic authorization to start training. **Formal training has NOT_RUN.** The task stops at this verified gate.

Previous CONDITIONAL_GO reports and machine gate are preserved in `experiments/rq1_train_phase_01/preflight/orchestration_v2/{prior_reports/,PRIOR_GO_GATE.json}`. Scientific architecture/configuration/budgets were not tuned or amended from outcomes.

| Acceptance criterion | Final disposition | Evidence |
|---|---|---|
| Valid forward/backward, optimizer updates, full checkpoint/restart, N/N+1 | **VERIFIED** | Real TRAIN branch smoke; actual integrated fit-loop tests; strict inference; new archival/terminal/resume engine tests |
| Speaker grouping, event identity, inference inputs | **VERIFIED within cached/supplied-phone scope** | Full TRAIN WAV/cache audit, exact inner exclusions, all4 UNKNOWNs, strict keyed exports |
| Eligible reproducible joint S/O/I policy | **VERIFIED** | Prior12 inner+4 outer gap fits,48 frozen policy pairs, independent official evaluation and checkpoint replay; no refits in this closure |
| Frozen architecture/loss/seeds/budget/selection and feasible bounded execution | **VERIFIED implementation/governance; full-run duration NOT_TESTED** | Complete driver, fixed recipes, every-epoch archive, terminal-only loading, cumulative2700s/seed budget including population initialization, safe stop/resume |
| Blocking findings resolved with reproducible receipts | **VERIFIED** |91 tests pass; full generated-data CLI rehearsal; independent reviewer confirms its P1/P2 fixes; no remaining reviewed engineering blocker |

## Three blockers that are now closed

1. **Full driver implemented:** `experiments/rq1_train_phase_01/train.py` defaults to plan mode. It schedules official seeds101/202/303 and four outer folds, three local inner fits plus outer refit, with separate target/CTC/gap training. All eligible inner scores precede immutable policy fitting; outer refit/application follows. Full-key predictions, per-fold/speaker/pooled official evaluation, direct/E4/fixed comparisons and cached-acoustic controls are wired end to end.
2. **Checkpoint/seed lifecycle implemented:** each branch resets Python/NumPy/Torch before model construction; fit-only scalers/weights/CTC competitors and exact IDs bind provenance. Every epoch has an immutable indexed full-state checkpoint; fixed terminal12/5/8 is the only selection path and is reloaded for inference. Partial-epoch updates restore optimizer, RNG, sampler and cursor exactly. Historical archive/index mutation and terminal/latest disagreement fail closed.
3. **Whole-seed stopping implemented:** persistent cumulative2700s active wall cap includes population initialization, hashing/loading, fitting/scoring/evaluation. Resume uses remaining allowance, not a fresh45minutes. SIGINT/SIGTERM and cap stops preserve committed safe points and release the writer lock. Checks are cooperative; an in-flight kernel/atomic commit finishes at the next safe point. No further update/utterance/stage starts after the detected stop. Cap increases are not CLI overrides.

A separate fresh-context native `openai-codex/gpt-6-luna` reviewer initially found population loading outside the budget, a two-file policy commit gap and permissive historical-index repair. All were fixed and regression-tested: population factory runs inside the budget, policy+digest is one atomic file, and only a just-completed archive commit can be repaired after exact equality. Final review reports no residual blocker among those findings.

## Measured verification receipts

Evidence under `preflight/orchestration_v2/`:

- **26 new tests passed**; combined suite **91 passed** (one benign warning on the existing empty-N std fixture). The final added fixture numerically matches new-backend E4 competitors and S/O LLRs to the verified MB07 producer at1e-12 on generated inputs.
- CLI plan: official three-seed schedule, **no corpus loading/fits**.
- CLI rehearsal: generated12-utterance/36-target/48-gap population, explicitly tiny fixture networks, all four outer/12 inner fits,48 independent branch fits, **96 epoch archives**, policy/evaluation/controls and complete pooled coverage. **10.236s process wall /703,240KiB child maxRSS**, of which8.865s was the persistent synthetic seed execution budget; common import startup is process overhead. This is not a formal-seed speed estimate.
- Selected real architectures exercise the same lifecycle engine on tiny generated-input unit fixtures. Existing eligible real TRAIN smoke remains24 utterances/738 targets/762 gaps, one epoch; no real experiment was restarted.
- Unauthorized formal command was **rejected with exit2 before population loading**; no formal output directory was created.
- Source/config freezes, initial/final review, full logs/commands and raw generated keyed artifacts are retained.

## Scope limits, not concealed blockers

Full formal3-seed/epoch training, its actual wall/RSS distribution, external validation and deployment safety remain **NOT_TESTED**. Hard-cap enforcement is verified without running a full budget. Existing small real pilot, compact frozen-feature heads and historical component measurements support bounded feasibility; they do not establish full-run completion under45minutes. A later authorized run must stop/report unfinished work at its cap; it may not retune budgets after outer outcomes.

The earlier valid mixed-source policy result is unchanged: fixed50 at nominal B=.2 has Macro3 F1 .161780, Macro3 AP .302723 and jointFWR .202222; speaker maximum.36. That is not a hard-rate or safety guarantee. Intended phones are explicitly supplied; prompt-text-only lineage remains INCONCLUSIVE. No raw encoder rerun or shared/staged architecture result is claimed.

## Stop and authorization boundary

**No formal training was launched.** No GPU use, protected/TEST_SEALED reads, DEV_SEARCH tuning, destructive changes or remote push. Formal mode requires BOTH an explicit flag and a separately operator-provided authorization file binding exact source/config/environment/seeds. This GO report does not create that authorization.

Reproduction: `experiments/rq1_train_phase_01/README.md`; closure receipts: `preflight/orchestration_v2/`; complete evidence hashes: `preflight/EVIDENCE_MANIFEST_SHA256.json`. Scientific design remains the frozen MB08 specification. Stop here, before formal training.
