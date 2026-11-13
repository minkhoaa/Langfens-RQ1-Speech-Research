# LANGFENS RQ1-TRAIN-PHASE-01 — completed bounded preflight

## Final decision: GO for controlled-run infrastructure

The three earlier engineering blockers—formal driver, coordinated seed/checkpoint lifecycle and whole-seed stopping—have been implemented and verified. **No formal training has run or been automatically authorized.** This GO is a preflight decision, not full-run completion, external validation or deployment/safety approval. See [GO gate](TRAIN_PHASE_01_GO_GATE.md).

The earlier CONDITIONAL_GO bundle is preserved under `experiments/rq1_train_phase_01/preflight/orchestration_v2/prior_reports/`, with prior machine gate and evidence manifest. The operator requested completing the unfinished engineering; this closure did not restart any completed MB03–MB08 experiment or refit real-data models.

## Phase0 and scientific freeze

Read MB08 final, readiness and training specification before new evaluations. HEAD `aa4e6ea2147180a4a820d77a71911f333fcb6721` plus dirty-source hashes defines provenance. Existing dirty/untracked user work was preserved; no stash, commit, destructive worktree operation or remote push.

The original acceptance/design freeze recorded283 source/config/report hashes at2026-11-08T13:26:28Z. All136 archived MB08 artifact hashes and all283 initial source hashes remain unchanged. Selected architecture/config did not change: independent H-002 target + independent CTC/E4, fixed50/50 S/O fusion, independent MB03 exact-gap insertion; direct H-002 fallback. The original machine config hash remains `6206c19db6cad9d99727a16cd9c220d077e09a9662a0229ee0971945bf1cbcb1`.

Initial execution used three parallel native `openai-codex/gpt-6-luna` agents in disjoint lanes plus supervisor; only unfinished corrections were resumed. The engineering closure was implemented by supervisor and independently reviewed by a fresh native `gpt-6-luna` reviewer. Historical audit-reconciliation provider timeout, partial state and one same-backend retry remain recorded; no alternate backend or concealed failed outcome.

## Completed evidence

| Area | Status | Evidence / claim boundary |
|---|---|---|
| TRAIN lineage/topology | **VERIFIED** |1,800 WAV paths/hashes/16kHz headers;12 official speakers;59,562 targets/61,362 gaps;four zero39 UNKNOWNs |
| Initial runner repair | **VERIFIED** | N-versus-N+1/shifted-context defect repaired; current source freeze matches; prior faulty code/tests retained |
| Actual real TRAIN branch smoke | **MEASURED** |24 utterances/six fit speakers,738 targets/762 gaps, one epoch, three full-state checkpoints;61.955s/865,252KiB; in-fit plumbing only |
| Branch restart/strict inference | **VERIFIED** | Actual target/gap/CTC interrupted fit-loop equality; all four real unknown utterances; strict per-utterance full-key boundary |
| Joint policy readiness | **MEASURED / VERIFIED** |12 inner+4 outer necessary gap fits;48 frozen policy pairs; official EVAL-V3.1 agreement for12 method/budget outputs and16 gap checkpoint replays |
| Complete controlled driver | **IMPLEMENTED / VERIFIED** | `train.py` plus protocol/runtime/engine/datasets/backend/workflow/pooled modules; all official seeds/folds; inner-policy-outer ordering; per-fold/speaker/pooled exports |
| All-epoch/seed lifecycle | **VERIFIED** | Independent branch reset; immutable epoch hash index; fixed-terminal loading; full optimizer/RNG/sampler/cursor; corruption rejection and exact resume |
| Whole-seed resource control | **VERIFIED implementation** | Budgeted population factory; cumulative2700s active wall/seed; remaining-time resume; safe-point timeout/signal stop; no silent budget extension |
| Full formal training / actual full-seed timing | **NOT_TESTED** | Deliberately not run; no claim that a complete seed has finished within45minutes |

## New orchestration closure: what actually ran

`preflight/orchestration_v2/SOURCE_FREEZE_{INITIAL,REVIEW_FIXES}.json` precede their synthetic tests. No architecture/loss/hyperparameter adjustment or real outer-score selection occurred.

- **26 new regression tests** (including missing/malformed budget-ledger rejection), including exact interrupted/resumed optimizer/RNG/sampler state at batch/epoch boundaries, per-branch initial seed independence, all epoch archives, terminal selection, cap persistence, signal handling, source/scaler/archive/policy/decision corruption rejection, budgeted data initialization, and numerical E4 competitor/LLR parity against the verified MB07 producer on generated inputs (1e-12 tolerance).
- **91 combined tests pass** in12.11s (one benign existing empty-N std warning).
- Safe CLI plan covers all3 seeds, four outer folds and48 branch fits/seed without loading corpus features or fitting.
- Safe CLI rehearsal executes the complete four-fold orchestration on **generated12 utterances/36 targets/48 gaps and tiny fixture models**:12 inner+4 outer bundles,48 branch fits,96 epoch archives, sealed policies, acoustic controls, official per-fold/speaker/pooled evaluator and full-key exports. Accepted-source process wall10.236s/max child RSS703,240KiB; synthetic seed budget8.865s. These are engineering profile values, not model quality or a formal-seed performance estimate.
- Selected full-size architecture heads also use the new engine in tiny generated-input unit fixtures. No real-data formal fit was launched.
- Formal mode without external authorization is rejected with exit2 **before population construction**; its requested output directory does not exist.

Independent reviewer found and then confirmed fixes for: (1) data initialization originally outside the wall budget; (2) policy/seal two-file crash window; (3) over-permissive missing archive-index repair. Final review reports no remaining blocker among those findings. Sources, commands, logs, profile,96 generated epoch checkpoints, keyed outputs and reviews are retained.

## Prior valid joint S/O/I result — unchanged

This is an archived-target/new-gap **mixed-source audit**, not formal Phase01 training. All policies use eligible inner scores only, allocated80% target/20% gap at predeclared B=.2/.5/1/2. Wrong S↔O type counts false; outer labels do not fit thresholds.

| Target stream + shared new gap | AP-S / AP-O / AP-I | Macro3 AP | Macro3 F1 at B=.2 | Joint FWR | Warnings |
|---|---|---:|---:|---:|---:|
|Direct H-002|.384080/.265655/.174642|.274792|.110711|.143333|624|
|E4-only|.272225/.229009/.174642|.225292|.140457|.280000|887|
|Fixed50 neural/E4|.402742/.330783/.174642|.302723|.161780|.202222|920|

Fixed primary FP364 includes29 wrong-type warnings; FWR=364/1,800, not920/1,800. Speaker FWR spans.086667–.360000; conditional95% speaker-bootstrap interval.158306–.248347. No hard-rate, external-population or safety claim. These values were not rerun/tuned during orchestration closure. Full details: [joint policy](TRAIN_PHASE_01_JOINT_POLICY_RESULTS.md).

## Corrections and claim limits

Initial source-absence claims by agents were incorrect: MB03 producer and eligible target inner streams existed and were recovered/verified. Initial gap-helper defect was real and repaired. Intermediate C runner findings were reconciled against final source/tests/smoke; all historical handoffs remain rather than being relabeled successes.

Supplied intended phones trace to canonical TextGrid annotations, not certified prompt-text dictionary lineage. This scope is explicitly authorized; prompt-only inference remains INCONCLUSIVE. Four neural/gap zero39 vectors remain; CTC aliases/SPN and alignment-only SIL behavior stay distinct. No raw encoder rerun, universal alignment claim, unified network or new independent cohort is asserted.

**MEASURED:** bounded real smoke/policy fits and evaluations; generated-data orchestration/profile/tests. **VERIFIED:** integrity, event/exclusion/provenance, checkpoint/scaler replay, exact restart, lifecycle and stop controls. **INCONCLUSIVE:** text-only lineage/external safety. **NOT_TESTED:** full formal3-seed execution/duration/RSS, external validation and shared/staged architecture ablations.

## Deliverables and stop

All six requested reports are current. Executable driver and modules: `experiments/rq1_train_phase_01/`; all receipts/configs/scores/checkpoints/tests/reviews: `preflight/`; commands: runner README and `preflight/orchestration_v2/IMPLEMENTATION_CLOSURE.md`; hashes: `preflight/EVIDENCE_MANIFEST_SHA256.json`.

**Stop at GO preflight. No formal training, GPU use, protected/TEST_SEALED reads, DEV_SEARCH tuning, destructive changes or remote push.** Formal execution requires separate explicit operator authorization bound to the exact source/config/environment/seeds; a GO report does not create that file.
