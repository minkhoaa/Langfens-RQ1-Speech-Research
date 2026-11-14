# TRAIN-PHASE-01 joint S/O/I policy preflight

**MEASURED and independently VERIFIED: eligible joint policy is estimable.** This is a bounded **mixed-source audit**: archived MB07 seed101 target/CTC scores plus newly fitted MB03 PLAIN gap heads. It is **not formal Phase-01 training, an independent cohort, or deployment readiness**. No target/CTC experiment was restarted.

Evidence root: `experiments/rq1_train_phase_01/preflight/` (paths below relative to it).

## Recovery, eligibility and freeze

The initial B handoff incorrectly reported missing producer/inner scores. Supervisor recovered `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py` and independently reconstructed all eligible inner target streams. Initial handoffs remain historical, not current blockers.

- **VERIFIED:** 12 inner target partitions, six fit speakers, three inner-held speakers, three outer-excluded speakers. Exact event joins, CTC checkpoint hashes/metadata and fit-only E4 competitors checked in `supervisor/inner_target_verification.json`.
- **MEASURED:** 12 necessary inner gap fits and four outer gap refits, fixed five epochs, phase01 seed schedule, two CPU threads. Combined phase wall time **33.57s**, including source feature preparation and score/checkpoint export; this is elapsed wall time, not CPU process time or full protocol runtime. No GPU used.
- Gap producer preserves the historical `Detector`, including dormant speaker branch initialization, fit-only mean/std floor, square-root class weights, weighted CE, AdamW and shuffle semantics. Dormant speaker logits are not used as input or loss. This is source-faithful replay under the new seed schedule, not historical checkpoint replay.
- All fit/held IDs, model/scaler hashes, 16 checkpoints, 184,086 inner gap rows and 61,362 outer gap rows are retained in `agent_b/results/`.
- `continuation_freeze.json` precedes fits; `policy_freeze.json` precedes threshold fitting; `thresholds_frozen.json` precedes outer gap fits; `evaluation_freeze.json` precedes outer application. Four outer folds are untouched by their respective fits/policy selection—not blind to historical TRAIN exploration.

**Policy:** B∈{.2,.5,1,2} false warnings/utterance, 80% target/20% gap; primary B=.2. Eligible inner OOF only. Target type=argmax(S,O), tie S; descending confidence then exact first-eight-byte SHA256 tie key. Largest prefix within floor(false budget×1,350 utterances). A wrong S/O type counts as FP. Frozen thresholds apply without outer labels; no rank-to-budget operation on outer data. There are **48 method/fold/budget policy pairs**, each with target+gap cutoff; gap cutoffs are shared across the three target methods.

## Complete outer operating results

All rows cover 1,800 utterances, 59,562 targets (S7,381/O1,825) and 61,362 gaps (I560), with exact N/N+1 IDs, no missing events and full coverage. Every method shares the same new gap scores and gap policies.

| Target method | Nominal B | Macro-3 Error F1 | Joint FWR | FP incl. wrong type | Total warnings |
|---|---:|---:|---:|---:|---:|
| Direct H-002 | .2 | .110711 | .143333 | 258 | 624 |
| Direct H-002 | .5 | .169229 | .378333 | 681 | 1,469 |
| Direct H-002 | 1 | .222239 | .780000 | 1,404 | 2,779 |
| Direct H-002 | 2 | .283400 | 1.743889 | 3,139 | 5,590 |
| E4-only | .2 | .140457 | .280000 | 504 | 887 |
| E4-only | .5 | .186450 | .604444 | 1,088 | 1,737 |
| E4-only | 1 | .206695 | 1.090556 | 1,963 | 2,895 |
| E4-only | 2 | .204288 | 2.075556 | 3,736 | 5,008 |
| Fixed .5 neural/E4 | .2 | **.161780** | **.202222** | **364** | **920** |
| Fixed .5 neural/E4 | .5 | .227474 | .498333 | 897 | 1,980 |
| Fixed .5 neural/E4 | 1 | .274982 | .975556 | 1,756 | 3,500 |
| Fixed .5 neural/E4 | 2 | .329746 | 1.870556 | 3,367 | 6,153 |

At primary B=.2:

| Method | S P / R / F1 | O P / R / F1 | I P / R / F1 |
|---|---|---|---|
| Direct | .711599 / .030755 / .058961 | .608696 / .030685 / .058425 | .389671 / .148214 / .214748 |
| E4-only | .502924 / .011652 / .022775 | .425447 / .117260 / .183849 | .389671 / .148214 / .214748 |
| Fixed | .757764 / .033058 / .063352 | .594805 / .125479 / .207240 | .389671 / .148214 / .214748 |

Shared insertion counts: TP83/FP130/FN477. Fixed target+gap FP364 includes **29 wrong-type S/O warnings**; warnings920 are **not** the FWR numerator. Exact class values and counts are authoritative in CSV/official JSON, not rounded prose.

| Method | AP-S | AP-O | AP-I | Macro-3 Error AP |
|---|---:|---:|---:|---:|
| Direct | .384080 | .265655 | .174642 | .274792 |
| E4-only | .272225 | .229009 | .174642 | .225292 |
| Fixed | .402742 | .330783 | .174642 | **.302723** |

New gap AP-I .174642 differs from historical MB03 .162125 because the preregistered seed schedule differs; this is not tuning or a new architecture claim. AP is threshold-independent. Historical MB08 selected Macro3 AP .298550 remains unchanged and separately labeled.

## Matched budgets, speaker dispersion and uncertainty

The table compares **matched nominal inner false budgets**, not equal achieved outer FWR. Fixed primary FWR .202222 slightly exceeds .2; no hard-rate guarantee is supported. Per-speaker fixed FWR spans **.086667–.360000**, sample SD .083959. Low S/O recall and this dispersion preclude safety claims.

Supervisor whole-speaker bootstrap: 1,000 paired draws, seed8808; conditional 95% intervals at B=.2 for fixed Macro3 F1 **[.131321,.183223]**, FWR **[.158306,.248347]**. These intervals condition on twelve repeatedly explored TRAIN speakers; they are not independent-population assurance. All method/budget intervals and paired differences are retained.

**Exploratory only:** already-frozen outer curves share achieved-FWR support [.28,1.743889]. Linear interpolation at FWR .5 gives Macro3 F1 direct .185286 / E4 .171644 / fixed .227640; at FWR1, .236199/.202924/.276477. Sparse four-point interpolation is not an executable threshold, does not use outer labels to choose a deployment policy, and cannot establish exact deployable matched-rate superiority. There is no common all-method support at FWR .2; no such comparison is manufactured.

## Independent verification and artifacts

- `supervisor/verify_joint.py`: independently recalculates all 48 policy pairs' inner target/gap cutoffs with integer SHA ties; verifies all 16 checkpoint/scaler/fit/held-ID records; applies without outer-label fitting; invokes **UnifiedEvaluatorV3 (EVAL-V3.1)** for all 12 pooled method/budget outputs. Class TP/FP/FN, Macro3 F1/AP, wrong-type FP and FWR match B to numerical precision. Receipt: `joint_independent_verification.json`; compact CSV: `joint_independent_metrics.csv`.
- `supervisor/verify_gap_checkpoints.py`: reconstructs held leading/interior/trailing gap features independently from H002 cache and replays all 16 checkpoints; max absolute probability difference **1.20e-7**. No refits. Initial verifier timed out due to repeated NPZ decompression in its own audit loop; array preloading corrected this, timeout log retained, no model/data change.
- `supervisor/joint_keyed_verified/`: independently regenerated NPZs for all 12 method/budget outputs, with explicit full `(dataset,speaker,utterance,domain,index)` keys, original continuous scores, decisions and gold in separate target/gap arrays; their hashes are in the official verification JSON. This is the unambiguous keyed policy export.
- `agent_b/results/joint_policy_raw_keyed.csv`: 1,451,088 keyed decision rows across 3 methods×4 budgets×120,924 events. Its target/gap domain is recoverable from finite/missing secondary score and type; do not join this legacy-format CSV by compact event_id alone across domains.
- All-budget pooled/fold/speaker CSVs, raw score NPZs, immutable cutoff fit IDs and freeze receipts: `agent_b/results/`.
- Bootstrap/interpolation CSVs and JSON: `supervisor/joint_{speaker_bootstrap,paired_bootstrap,matched_fwr_exploratory}.csv`, `joint_uncertainty.json`.

Reproduce verification only (CPU environment in `supervisor/COMMANDS.md`):

```bash
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_inner_targets.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_joint.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_gap_checkpoints.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/joint_uncertainty.py
```

**NOT_TESTED:** formal three-seed retraining, external speakers, deployment calibration/safety and architecture ablations. No further refitting/tuning is needed to verify this audit's joint-policy provenance.
