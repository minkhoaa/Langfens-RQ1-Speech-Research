# Agent C final independent audit — RQ1-TRAIN-PHASE-01

**Scope:** final read-only review of the completed Agent A/B preflight lanes, plus bounded reruns of their tests and keyed-order audit. No formal fit, outer policy application, protected/TEST_SEALED or DEV_SEARCH access, CUDA, or changes to A/B/root artifacts. Supervisor owns final gate and reports.

**Conclusion:** preflight is **NOT a formal-training readiness pass**. A's synthetic primitives and B's policy utility are useful, but critical gaps remain: A's gap feature builder fails the specified N+1 topology, full E4/inference integration and cross-branch resume exactness are unverified, and B could not produce eligible joint inner scores/thresholds because exact MB03 gap fit provenance and complete target inner stream provenance are unavailable. Keep joint metrics null/NOT_ESTIMABLE. Supplied intended phones remain safe and explicitly allowed; prompt-text-only lineage remains inconclusive but is not a blocker if the intended-phone input contract is stated honestly.

## Lane artifacts and independent validation

Reviewed A: `runner.py`, `test_runner.py`, `phase01_config_frozen.json`, `SOURCE_FREEZE.json`, `test_profile.log`, `TRAIN_PHASE_01_RUNNER_VERIFICATION.md`, `HANDOFF.md`.

Reviewed B: `joint_policy.py`, `test_joint_policy.py`, `verify_keyed_order.py`, `freeze_receipt.json`, `HANDOFF.md`.

Independent SHA-256 checks matched the lane receipts:

- A runner `09ec3ab086995076637abd975301f3821825acbfceb26eee2463b1b6fdf51edc`; tests `c36e1f68dbf25b0aa2b01d8d4728e58f9933ddd59554b9789ac5eba511722b6e`; frozen config `6206c19db6cad9d99727a16cd9c220d077e09a9662a0229ee0971945bf1cbcb1`.
- B policy `63976ea3adfd27cfd35c853ab91d57f51af9a413d0fc7061ff656e90b1741917`; tests `1852188fe6ecbb51f1cb178055eeaa99d8d876a2c05f053b823d3664c874ae57`; keyed-order audit `558bcac5c158a5297520ec2eddfe67a6335c0f3b6c805c803203fdd3c8926fda`.
- B's pre-outcome freeze records spec/config authority and reused artifacts; no gap fits or policy application were started.

**Reran (MEASURED):** A's five synthetic tests passed (`5 passed in 0.66s`); B's four `unittest` policy tests passed; B's read-only keyed-order verification passed for 1,800 utterances, 59,562 targets, 61,362 gaps, with fixed50 target hash `d08b3f240f5d9b2442222d0a2f1c6b5c95627979b9cc425c375b2fbd0e852058` and keyed MB03 hash `8a5692e43f64b08d60252b9f6a8c63d8be7f8c5e6d6dcfa4d0e26ef7e65bb42b`. These verify synthetic contracts and ordering, not trained model outputs or policy eligibility.

### Material runner defect: N+1 gap topology

**MEASURED by bounded direct invocation:** `runner.gap_input(crops, refs)` given three target crops returns `(3,1614)`, whereas the specification requires four gap rows. The implementation creates arrays of length N, not N+1. It also places `right[0]` from target crop 1 (rather than target crop 0 for the leading gap) and `left[-1]` from crop N-2 (rather than crop N-1 for the trailing gap). Thus it both omits an event and shifts adjacent-context inputs. This is a **blocking identity/input defect**, not merely a missing test; do not use this helper for gap fitting until repaired and retested. Existing `export_keyed` correctly rejects a gap count other than N+1, so current helper and export contract do not compose.

### Checkpoint/restart audit

A implements optimizer/model/RNG/sampler/cursor serialization after updates. Synthetic code proves a direct target-head next-update equality from a saved state with dropout and optimizer state. It does **not** run interrupted `fit_target` versus uninterrupted `fit_target` end-to-end. Neither gap nor CTC was tested for next-update/restart equivalence; CTC batches/permutation/cursor also lack an executed interrupted-resume fixture. The implementation is promising but **complete restart exactness remains INCONCLUSIVE/NOT_TESTED**, especially interruption between differently sized last batches / across epoch boundaries. No real checkpoints or formal training receipt exist. The one-epoch smoke functions do not constitute an enforced CLI/run guard.

### E4, unknowns, speaker and event identity

- A separates target, gap and CTC parameters and implements spec-shaped heads and fixed `softmax([0,SLLR,OLLR])`/0.5 fusion arithmetic. The test compares only this simplex arithmetic; exact competitor selection, CTC sequence likelihood, canonical-mean E4 aggregation and parity against the verified producer are not implemented/tested. No complete audio + supplied-phone inference adapter or label-free dataflow exists.
- A's unknown helper enforces zero39 only when called with one of four hard-coded event IDs. There is no demonstrated full manifest-to-reference map, automatic UNKNOWN tagging, or real event stream coverage. Preserve known zero vectors; do not silently map unsupported phones to phone0/AA. Full real-data unknown handling is **NOT_TESTED**.
- `grouped_plan` uses the official groups and visibly excludes each validation group from its inner fit. Synthetic coverage checks the outer held group but does not assert all inner fit/held partitions, fit-derived scaler provenance, or predictions' per-event model provenance. No A fit actually exercised speaker exclusions.
- B's read-only audit verifies manifest order/metadata/shape for reused fixed50 targets and keyed MB03 N+1 gaps. This is **VERIFIED identity/order** only; it does not establish newly fitted prediction lineage or inner OOF eligibility. MB03 prediction row reconstruction remains source-order/prediction reproduction, not checkpoint replay.

### B joint policy and FWR

The utility implements the specified target FP rule (`predicted S/O type != gold S/O type`, including S↔O) and gap FP rule (predicted insertion on gold-negative gap), .8/.2 allocation, largest prefix under floor budget, descending score then descending SHA tie hash, immutable threshold application, duplicate-ID rejection, and basic fit-vs-held speaker checks. The implementation is **VERIFIED at code level and MEASURED by four synthetic tests**, not as an executed policy.

Limitations relevant to eligibility: `freeze_joint` receives held-speaker labels and held speaker IDs but does not receive/validate each prediction's model-fit speaker IDs; it cannot establish that every prediction's entire pipeline excluded that speaker. It does not check complete inner fold coverage/one prediction per event or verify fitted checkpoint/normalizer provenance. Tests check overlap, held membership, wrong-type errors and duplicate IDs, but do not test a partial SHA-tie cutoff that selects some tied rows, all score edge cases, or complete OOF provenance. B froze code before new outcomes but executed no thresholding or outer application. Consequently no eligible thresholds/raw policy predictions/false-warning counts or full-system metrics exist; all such values are null/NOT_ESTIMABLE, not zero.

B reports no exact MB03 gap-head training source/checkpoint/normalizer sufficient to replay the historical recipe. It also could not certify complete fixed50/E4 target inner score provenance across partitions. Do not reconstruct missing optimizer/sampler/initialization semantics, fit a substitute gap model, or use historical outer gap scores as joint inner OOF. The freeze's maximum 12 inner and 4 outer seed101 gap fits/20-minute CPU budget were **not exercised**.

## Independent scientific/data audit carried forward

The inspected authority remains `reports/METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md`, phase0 config and acceptance freeze. Primary remains independent H-002 target + CTC/E4 and fixed 50/50 fusion, direct H-002 fallback, independent MB03 exact-gap model; no shared trainable trunk. Seeds/folds/optimizers/losses/fixed terminal epochs and budgets remain as specified. Do not report architecture ablations as tested. Existing independently recomputed keyed MB03 AP-I `.1621248969942254`, P@560 `.24285714285714285` remains **VERIFIED/MEASURED from preserved scores**, not checkpoint replay. TRAIN seeds do not represent independent cohorts. Existing source audit says audio features are predicted-alignment, intended-phone-conditioned, not realized-phone/gold-timing inference; the exact raw prompt-to-phone lineage is unproven. Supplied intended phones are **SAFE/IN-SCOPE** under this explicit protocol. The four unsupported tokens remain explicit zero-vector UNKNOWNs.

Legacy qualifications remain: historical MB07 `realized_target_FWR` is total alert rate, not false-warning rate; its prior slice includes three acoustic dimensions; learned meta crossfit has a base-fit exclusion limitation. These do not turn into evidence for this new joint policy. No complete-system F1/FWR is estimable without eligible frozen inner target+gap thresholds.

## Blocking and nonblocking risk register

| ID | Finding | Evidence/status | Severity / disposition |
|---|---|---|---|
| F1 | `gap_input` produces N rows, shifted neighbor features, not N+1 edge-correct rows | Direct invocation: 3 crops → `(3,1614)`; source inspection | **BLOCKER**: repair before any gap fit; assert exact N+1 leading/interior/trailing reference and acoustic inputs. |
| F2 | End-to-end audio/intended-phone inference, E4 producer parity, full keyed export and label-free path absent | A handoff; synthetic simplex test only | **BLOCKER**: wire actual frozen feature/CTC/E4 producer or validated equivalent; parity test before outcomes. |
| F3 | Exact restart only tested by direct target next-step fixture; no integrated gap/CTC interrupted resume | A tests/source | **BLOCKER**: interrupted vs uninterrupted training equivalence for target/gap/CTC, sampler cursor, RNG/dropout, optimizer, partial/final batch and epoch boundary. |
| F4 | Unknown handling is a hard-coded validator, not end-to-end mapping | A source; known four rows from frozen audit | **BLOCKER** for data inference: verify all four retained zero39 UNKNOWN event IDs, N/N+1 and no AA/phone0 fallback. |
| F5 | No real eligible TRAIN smoke, full CLI/run guard, checkpoint or event stream | A handoff | **BLOCKER**: implement enforced one-epoch/small-subset guard and bounded real TRAIN-only smoke; freeze first. |
| F6 | No eligible joint inner OOF threshold set; MB03 gap refit provenance absent; target inner streams incomplete | B handoff/receipt | **BLOCKER** to full-system readiness: recover verifiable exact source/checkpoint or do not fit; validate complete target+gap inner fold exclusions and freeze cutoffs before outer application. Keep metrics null until then. |
| F7 | B policy does not validate per-row model-fit exclusion, complete coverage, or checkpoint/normalizer lineage | B code | **BLOCKER** for claiming eligible policy, though utility arithmetic is sensible. Add provenance-aware validation and fixtures. |
| F8 | Prompt-text → phone lineage inconclusive | Prior source audit | **NONBLOCKING** if supplied intended phones remain the declared input; never claim prompt-only. |
| F9 | Synthetic profile (1.429s / 614,488 KiB) is not real fit/full-budget feasibility | A profile | **BLOCKER before outer outcomes**: record eligible TRAIN small-smoke wall/RSS and batch/step evidence; not necessarily a reason to cancel if budget feasible. |
| F10 | B tie tests do not exercise partial tied cutoff; joint policy has no actual data outcome | B code/tests | **NONBLOCKING to utility design**, required before threshold gate: add boundary test and apply only after eligible source records exist. |

## Candidate frozen protocol update (candidate only; supervisor authoritative)

Retain the existing immutable architecture, training hyperparameters, official speaker groups, seeds, score formulas, thresholds, and CPU-only limits without adjustment. Candidate gate update: **no formal folds/seeds or outer outcomes under the current runner** until (a) correct N+1 gap context builder; (b) hash-frozen A/B source/config and complete end-to-end input/event-ID contract; (c) exact producer E4 parity and all four UNKNOWN rows preserved; (d) integrated interrupted-resume equivalence for target/gap/CTC; and (e) bounded eligible TRAIN smoke with model-fit speaker exclusions, timing and maxRSS. For complete-system policy evaluation, also require (f) eligible target and gap inner OOF score sources with per-event fit IDs/checkpoint/normalizer hashes, complete coverage and policy cutoffs frozen before outer application. If historical MB03 gap producer cannot be recovered, stop the gap-refit branch and report the gap policy NOT_ESTIMABLE; do not substitute unexplained implementation. The acceptance freeze permits supervisor-approved narrow staged-training deferral if the runner/data/inference infrastructure is valid; such a deferral must keep joint metrics null and make no full-system readiness claim. It does not cure the current N+1/inference/restart blockers. Continue to allow explicitly supplied intended phones; text-only lineage is not a mandatory gate under frozen scope.

## Exact bounded commands run

Environment for Python commands: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; A tests also set `torch.set_num_threads(2)`.

Exact commands and outcomes:

```bash
cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m unittest -v test_joint_policy
# passed: 4 tests

cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest -q test_runner.py -p no:cacheprovider
# passed: 5 tests

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/verify_keyed_order.py
# passed: 1,800 / 59,562 / 61,362 verified

sha256sum /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/runner.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/test_runner.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/phase01_config_frozen.json /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/joint_policy.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/test_joint_policy.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/verify_keyed_order.py
# passed: all six hashes match receipts

git -C /home/khoa/Projects/New-Method-Research diff --cached --name-only
# passed: empty, no staged files
```

For the direct N+1 check, invoking `runner.gap_input(torch.zeros(3,17,768), torch.zeros(3,39))` returned `(3,1614)` instead of required `(4,1614)`. This command established the blocker; it is not a passing contract test.

No tests/files were added in this auditor lane beyond this requested final handoff; no A/B code was edited. Root/reports/legacy files remain read-only.

### Acceptance report
```acceptance-report
{
  "criteriaSatisfied": [
    {"id":"criterion-1","status":"satisfied","evidence":"Reviewed A/B code and receipts independently, reran bounded tests/order validation, and wrote findings only to the requested lane output."},
    {"id":"criterion-2","status":"satisfied","evidence":"Identified direct N+1 gap-input defect, exact restart/E4/identity/policy evidence boundaries, risks, blocker list, source hashes, commands and candidate frozen protocol."}
  ],
  "changedFiles":["experiments/rq1_train_phase_01/preflight/agent_c/FINAL_HANDOFF.md"],
  "testsAddedOrUpdated":[],
  "commandsRun":[
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m unittest -v test_joint_policy","result":"passed","summary":"4 tests passed."},
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest -q test_runner.py -p no:cacheprovider","result":"passed","summary":"5 synthetic tests passed in 1.39s."},
    {"command":"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/verify_keyed_order.py","result":"passed","summary":"Verified manifest order for 1,800 utterances, 59,562 targets and 61,362 gaps."},
    {"command":"sha256sum /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/runner.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/test_runner.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a/phase01_config_frozen.json /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/joint_policy.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/test_joint_policy.py /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b/verify_keyed_order.py","result":"passed","summary":"All six hashes match receipts."},
    {"command":"git -C /home/khoa/Projects/New-Method-Research diff --cached --name-only","result":"passed","summary":"No staged files."}
  ],
  "validationOutput":["A tests 5/5; B tests 4/4; keyed-order audit VERIFIED; gap builder violates N+1 topology."],
  "residualRisks":["Gap context builder defect blocks use.","Exact producer E4 parity and full inference absent.","Gap/CTC integrated resume exactness not tested.","No eligible TRAIN smoke or joint inner policy predictions; thresholds and full-system metrics NOT_ESTIMABLE.","Prompt-only lineage inconclusive; supplied intended phones remain allowed."],
  "noStagedFiles":true,
  "diffSummary":"Added only FINAL_HANDOFF.md in Agent C lane; no other files modified.",
  "reviewFindings":["blocker: agent_a/runner.py:gap_input - returns N rows and shifts edge context instead of required N+1 gap rows.","blocker: no eligible joint inner policy and no exact MB03 gap refit provenance; full-system metrics remain null/NOT_ESTIMABLE."],
  "manualNotes":"Do not start formal training from these partial preflight results. Supervisor owns disposition."
}
```
