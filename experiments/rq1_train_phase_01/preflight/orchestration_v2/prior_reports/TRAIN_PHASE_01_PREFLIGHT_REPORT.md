# LANGFENS RQ1-TRAIN-PHASE-01 bounded preflight

## Decision

**CONDITIONAL_GO; formal training remains blocked.** Bounded runner/model/inference contracts and joint-policy provenance now have reproducible receipts. Complete controlled seed/fold orchestration, all-epoch/terminal-selection lifecycle and whole-seed hard-stop governance remain unfinished. See [GO gate](TRAIN_PHASE_01_GO_GATE.md) for the exact three launch conditions. No formal run was launched.

## Recovery and freeze

Read MB08 final/readiness/training specification and original machine config before new outcomes. HEAD `aa4e6ea2147180a4a820d77a71911f333fcb6721`, with substantial dirty/untracked user artifacts preserved. No stashing, commits, cleanup, remote push or managed worktree creation was attempted. Three native `openai-codex/gpt-6-luna` workers used disjoint sole-writer lane directories; supervisor alone writes root reports. No alternate CLI backend or child fanout.

Acceptance/design froze at **2026-11-08T13:26:28Z**, with283 source/config/report hashes, before new fits/evaluations. Selected architecture stayed independent H-002 target + independent CTC/E4, fixed50/50 S/O fusion, independent MB03 exact-gap head; direct H-002 fallback. No unified-network substitution or search. Formal design/config is byte-identical to MB08, detailed in [frozen protocol](TRAIN_PHASE_01_FROZEN_PROTOCOL.md).

Evidence root: `experiments/rq1_train_phase_01/preflight/`. `supervisor/phase0_receipt.json` records Git status/worktree/environment/source hashes; `ACCEPTANCE_FREEZE.md` sets measurable scope and limits. All136 archived MB08 artifact hashes and283 frozen sources were independently verified unchanged. Authoritative TRAIN manifest/support/topology and historical AP/P@560 reproduce. Initial workflow receipt and corrective continuation IDs are retained; no completed MB03–MB08 experiment was restarted.

## Workstream results

| Workstream | Evidence | Status / boundary |
|---|---|---|
| A runner | Independent heads, fit-only scaling/weights, grouped plan, optimizer/RNG/sampler checkpointing; repaired exact N+1 gap construction; MB07 E4 producer path | **VERIFIED primitives**, not a complete formal driver |
| A real TRAIN smoke |24 utterances, six fit speakers,738 targets/762 gaps; one epoch; three checkpoints;61.955s/865,252KiB CPU | **MEASURED**, in-fit plumbing only, not OOF/generalization |
| Supervisor runner verification | Actual interrupted/resumed target/gap/CTC fit-loop equality, partial final batches and epoch boundaries; strict gold-free per-utterance inference; four real UNKNOWN utterances | **VERIFIED**,24 runner/policy tests pass |
| B joint policy |12 eligible inner gap fits+4 outer gap refits, seed101 schedule, five fixed epochs;48 frozen method/fold/budget policy pairs;33.57s combined phase wall | **MEASURED**, mixed archived-target/new-gap audit, not formal training |
| C scientific/data audit |1,800 TRAIN WAV hashes/16kHz headers, manifest/cache/event/reference lineage; independent baseline AP and inner-policy reconstruction | **VERIFIED**, supplied intended phones only; prompt-only unproven |
| Supervisor joint evaluation | All12 method/budget pooled outputs match official EVAL-V3.1;48 policy pairs rederived;16 gap checkpoint/feature inference replays | **VERIFIED**, raw full-domain keyed outputs retained |
| Formal3-seed protocol/architecture controls | No formal seed/fold driver execution, encoder rerun, shared-trunk/staged training, full-budget feasibility or external validation | **NOT_TESTED** |

### Corrections and audit history

Initial A implementation had a real N-row/shifted-neighbor gap defect and tests that expected the wrong contract. C found it; A repaired it and supervisor reverified exact values/edges. Initial A/B/C claims that data, the MB03 producer or target inner provenance were absent were incorrect. Supervisor recovered the existing producer and score sources; retained lanes completed only unfinished work. Historical handoffs/failure logs and source snapshots remain; none are silently relabeled successful.

Intermediate C runner findings were taken while A was still completing corrections. Final source/config/test hashes, smoke receipt and supervisor integrated tests supersede stale two-failure/absent-freeze assertions. The original A freeze intentionally describes its prior version; the amended BCF2 freeze matches accepted source. C's final reconciliation explicitly accepts this correction and the conditional gate. One reconciliation continuation failed with `Codex SSE response headers timed out after 300000ms`; its partial state/output were captured, and exactly one same-model native retry completed the read-only reconciliation without repeating tests/fits. No alternate backend was used. Audit development errors (schema assumption, exact-text edit failures, verifier NPZ decompression timeout) are preserved separately and never treated as model outcomes.

## Critical measured operating results

TRAIN:1,800 utterances/12 speakers/59,562 targets (C50,356/S7,381/O1,825),61,362 gaps (I560), four official speaker groups. Outer fits exclude their three held speakers; inner fits exclude both inner-held and outer-held speakers. Thresholds use inner OOF only, before outer application; target wrong-type warnings count false.

| Target stream + shared new gap | AP-S / AP-O / AP-I | Macro3 AP | Macro3 F1 at B=.2 | Joint FWR | Warnings |
|---|---|---:|---:|---:|---:|
|Direct H-002|.384080/.265655/.174642|.274792|.110711|.143333|624|
|E4-only|.272225/.229009/.174642|.225292|.140457|.280000|887|
|Fixed50 neural/E4|.402742/.330783/.174642|**.302723**|**.161780**|**.202222**|920|

Fixed primary FP364 includes29 wrong-type S/O warnings. JointFWR is364/1,800, not920/1,800. Threshold budgets `.2/.5/1/2`, allocation `.8/.2`, ties and all outputs were frozen, not selected from outer outcomes. Full class/fold/speaker/budget tables, coverage, uncertainty and exploratory matched-achieved-FWR caveats: [joint policy report](TRAIN_PHASE_01_JOINT_POLICY_RESULTS.md).

At nominal B=.2 fixed speaker FWR spans .086667–.360000. Conditional whole-speaker bootstrap95% interval is .158306–.248347 for FWR and .131321–.183223 for Macro3 F1. This repeatedly explored TRAIN cohort is not independent confirmation; no deployment or safety claim. The primary architecture was not changed because an audit metric improved or missed a budget.

## Evidence labels and claim boundaries

- **MEASURED:** bounded smoke, necessary gap refits, eligible threshold fitting/application, official metric replay, runtime/RSS and executed tests.
- **VERIFIED:** source/artifact hashes, raw WAV/cache and event joins, four official groups, inner exclusions, checkpoint/scaler/ID provenance, full topology, corrected FWR and actual fit-loop restart equality.
- **INCONCLUSIVE:** raw prompt-text→phone lineage and external-population safety. Supplied canonical phones are authorized; canonical TextGrid origin is disclosed, not disguised as dictionary-derived text.
- **NOT_TESTED:** formal target12/gap5/CTC8 per-seed execution, complete orchestration/hard stop, full-seed time/RSS, new waveform extractor run, formal acoustic/reference controls, architecture ablations and external cohort.

Historical MB03 gap AP-I .162124897/P@560 .242857 and selected MB08 Macro3 AP .298550 reproduce unchanged. New seed-scheduled gap AP-I .174642 is a bounded refit result, not historical checkpoint replay or a new method. Historical direct neural checkpoints were not saved; eligible archived score/source provenance is verified, not invented checkpoint replay.

## Deliverables and reproducibility

Six root reports: preflight, runner verification, joint policy results, scientific audit, frozen protocol and GO gate. Executable code/tests,19 new checkpoints (3 accepted smoke+16 policy fits), earlier smoke engineering bundles, configurations, source/input hashes, raw keyed predictions, cutoffs and CSVs remain under the preflight root. Full policy-key NPZs explicitly include dataset/speaker/utterance/domain/index; compact IDs are never joined across target/gap domains without disambiguation.

Verification commands/exit notes: `supervisor/COMMANDS.md`. The final combined suite reports **65 passed** (41 baseline/EVAL plus24 branch/inference/policy), with one benign empty-N tensor std warning. Additional standalone C lineage/boundary/policy receipts are retained, not added as if disjoint tests. Orchestration/failure recovery: `supervisor/orchestration_receipt.json`; final C reconciliation: `agent_c/FINAL_HANDOFF.0eb6c855.0e1f2305.061fc1f1.md`. Final evidence manifest indexes reports and artifacts by SHA256.

**Stop:** no formal training, GPU use/lease, protected/TEST_SEALED reads, DEV_SEARCH tuning, destructive changes or remote push. Preserve this bundle; close the three explicit orchestration conditions in a separately authorized bounded continuation before reconsidering GO.
