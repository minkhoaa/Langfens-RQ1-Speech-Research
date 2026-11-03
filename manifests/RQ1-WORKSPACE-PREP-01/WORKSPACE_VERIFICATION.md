# RQ1 P-SCORE / P-SUP setup — final process checkpoint

**SETUP_COMPLETE_WITH_ACCOUNTED_SOURCE_LIMITATIONS; BOTH_LANES_DOCUMENTATION_READY; IMPLEMENTATION_AND_EXPERIMENTS_NOT_AUTHORIZED.** This final checkpoint advances the earlier preparation-pending state without erasing it. Actual verification receipt: [WORKSPACE_VERIFICATION.json](WORKSPACE_VERIFICATION.json). Hash/Git/document checks certify setup/isolation only, not scientific performance or remote artifact readiness.

## Exact identities

| Item | Path / branch | Commit SHA |
|---|---|---|
| Frozen shared base | `prep/rq1-shared-base` at freeze | `0d268ff8c444e16fd76aca2f4ad6ece966908edc` |
| P-SCORE | `/home/khoa/Projects/KLTN/langfens-rq1-p-score` · `prep/rq1-p-score` | `de1db5cc2f0b4ed8b750ccc6120a314679dc3b6f` |
| P-SUP | `/home/khoa/Projects/KLTN/langfens-rq1-p-sup` · `prep/rq1-p-sup` | `d6ef18443752b0add3f3dfcfff42f7fa7be967c1` |

Both arm commits are direct children of the exact frozen shared base; their merge-base is that SHA. This final coordinator receipt is committed **after** the freeze on `prep/rq1-shared-base` in `/home/khoa/Projects/KLTN/Dataset`. That branch tip advances for receipts only; neither arm is rebased or advanced to it. Exact receipt-commit SHA is resolved with `git log -1 --format=%H` on the coordinator branch, avoiding a self-referential digest.

Shared path relative to each worktree: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`.

**SHA-256: `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.** Same bytes at frozen base and all three checkouts. V1/addendum/source manifest are committed before branch creation; do not edit historical contracts or independently diverge shared V1.

## Proposal conformance / source state

Original UIT proposal found, read, hash verified; [addendum](PROPOSAL_CONFORMANCE_ADDENDUM.md) reconciles selected acoustic error detection **and** feedback, non-oracle reference practice, fair comparison, uncertainty and integration. The distinct latest acoustic-grounded omission proposal is still **NOT LOCATED** in documented local scopes. Latest-proposal conformance cannot be claimed until exact source/version is supplied or operator explicitly accepts that limitation. This accounted gap does not block workspace setup but blocks unconditional proposal-conformance/execution claims. Joint-Align full text and several source-code licenses remain unresolved if those assets are pursued.

## Per-lane readiness and next tasks

### P-SCORE

Committed isolated files under `manifests/RQ1-P-SCORE/`: experiment spec, provenance manifest, output/schema/task/CPU-test plan, process checkpoint; additive lane registry entry. Frozen E0 S-vs-SD primary contrast, self-aligned/segmental and greedy-CTC+DP controls, numerical versus occupancy normalization, graph/repeat/neighbor requirements. No scoring implementation or synthetic/real-data execution.

Pending: approved graph/path/occupancy/neighbor restriction and license/source disposition, prospective practical/noninferiority/false-alarm margins, separately authorized synthetic CPU implementation/tests, future actual model/package/cache verification and baseline parity qualification. Frozen E0's calibration speakers overlap historical acoustic training, explicitly disclosed. No learned calibrator/fusion approved; no population inference from G=2 reused VAL.

### P-SUP

Committed isolated `manifests/RQ1-P-SUP/`: experiment spec, label/collator/output design, Kaggle T4 deployment design, provenance manifest, process checkpoint; additive lane registry entry. Natural-operation/existence options stay compact; ONE parameterization/control requires later registration, not an architecture selected by this setup. Matched independent initialization/update schedule and fixed/best/state durability designed; no model/training/collator code changes.

Pending: explicit distinct-objective/two-head-policy amendment; treatment and parameter-matched binary control choice; frozen label/mask/loss/initialization/control inventory, support check and effect margins; separately authorized synthetic CPU no-training extension; private package/source/weight/remote allocation/AMP/mirror/recovery verification before separately authorized remote training. No quota/auth/secret/kernel/deployment checked or invoked. P-SCORE is not a dependency.

## Verification actually executed

- Current branch/HEAD/dirty state/worktrees/branches inspected before mutation; local closure/revalidation bytes newer than local commit were explicitly preserved and committed, not replaced by older remote. Original `research-phase` remains `b25a7595360294798d8d06b37592dd65dd4c9c33`; recovery worktree ref unchanged.
- `git worktree add -b` created both sibling paths from the same immutable SHA. Existing requested paths/branches were absent. No reset/clean/stash/force/push/merge/rebase or existing-worktree overwrite; stale E1 worktree record left untouched.
- Both arm statuses clean after commits. Exact parents, branch names, merge-base, contract bytes and per-arm provenance identity verified. P-SCORE's documents absent from P-SUP and vice versa; coordinator retains only shared documents and final receipts, no arm-doc copies.
- 57 source/input/archive hashes verified in each of three workspaces (**171 checks**), including model/trainer/collator/parser/evaluator/configs/split/prompt/dictionary and E0/E1 archives. Original **639/640** fingerprinted regular tracked files byte-unchanged relative to session entry; only global registry receives new additive text. Pre-existing parent-checkpoint and research closure/revalidation contents retained.
- E0/E1 metadata/checkpoint references unchanged; model weights were **not** read/loaded/copied/modified by any setup command. No current durable-store/weight readback claim. Original ignored model artifacts remain outside all arm checkouts; a Git/source audit cannot certify remote checkpoint availability.
- No `.env`, datasets or checkpoints directories in either new arm; nested repo not initialized/copied. New staged artifacts all regular `.md`/`.json` or checksum text under100,000 bytes; no newly tracked data/audio/TextGrid/weight/secret/large binary. Existing tracked prediction audit tables/dictionaries/papers are inherited history, not new uploads, and remain subject to pre-existing privacy/publication constraints.
- JSON/provenance/hash/metadata selectors checked; new registry links verified (7 including shared pointers in P-SCORE; 8 in P-SUP). Scoped common and both arm Git whitespace checks clean. Whole common commit diff check has **12 inherited Markdown trailing-space findings** in original closure B/C/E reports; checksummed historical bytes intentionally preserved, not silently cleaned or reported PASS.
- One precommit audit assertion initially counted new staged arm docs as pre-existing tracked modifications and failed before either commit. Corrected predicate reran all checks successfully; failure recorded in arm checkpoints/JSON receipt. No fallback or experiment execution followed.
- Original nested checkout status and binary diff exactly match initial snapshots (14 mode changes, two deleted paper symlinks); original coordinator status retains `m repo`, intentionally not globally clean. Ephemeral audit aids `/tmp/langfens-two-workspaces-audit/` are not restart dependencies; durable receipt/source hashes and Git history are authoritative.

**Not run:** scorer/model/training implementation; unit/regression/model tests; dataset/audio/TextGrid loading; inference/replay; calibration/model fitting; backward/optimizer/training; GPU/API/remote staging/deployment; sealed TEST data access. Only split metadata read for membership/exclusion. No historical evaluator semantics or thresholds modified; no scientific PASS. DIAG-02B PARTIAL (9/80 archive parity) and E1 cross-fold rollout NO-GO remain.

Current process: training **NOT_STARTED**, **0 successful updates, 0 skipped updates/no attempts**, no new model artifact. Last valid model references are archived E0 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` and E1 fixed `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`, not newly certified weights.

## Exact read-only next commands / STOP

P-SCORE:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
git status --short
git rev-parse HEAD
# Expected de1db5cc2f0b4ed8b750ccc6120a314679dc3b6f
git merge-base HEAD prep/rq1-p-sup
# Expected 0d268ff8c444e16fd76aca2f4ad6ece966908edc
(cd manifests/RQ1-WORKSPACE-PREP-01 && sha256sum -c CONTRACT_SHA256)
```

P-SUP:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-sup
git status --short
git rev-parse HEAD
# Expected d6ef18443752b0add3f3dfcfff42f7fa7be967c1
git merge-base HEAD prep/rq1-p-score
# Expected 0d268ff8c444e16fd76aca2f4ad6ece966908edc
(cd manifests/RQ1-WORKSPACE-PREP-01 && sha256sum -c CONTRACT_SHA256)
```

Read local AGENTS/index/shared contract/addendum/provenance and `manifests/RQ1-P-SCORE/PROCESS_CHECKPOINT.md` or `manifests/RQ1-P-SUP/PROCESS_CHECKPOINT.md`. Confirm SHA/status/hash, resolve source/registration gates and request the next bounded authorization. **STOP after setup/documentation. Do not implement or launch either arm without separate operator authorization.** No smoke-training exception, local training fallback or TEST unsealing is implied.
