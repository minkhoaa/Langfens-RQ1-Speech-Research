# TRAIN-PHASE-01 GO gate

## Verdict: CONDITIONAL_GO — formal launch remains blocked

**Bounded preflight executed; stop here.** Runner branch tests, real TRAIN smoke, strict cached inference contracts and eligible joint policy are verified. A complete controlled seed/fold orchestration driver and its archival/budget governance are still missing. This is **not full-system/formal-training readiness** and does not authorize training now.

| Acceptance criterion | Disposition | Evidence |
|---|---|---|
|1. Finite forward/backward, optimizer updates, checkpoint/restart, N/N+1 | **VERIFIED at branch/smoke level** | A7 tests; supervisor actual fit-loop interruption/restart equality for all3 branches; strict per-utterance export;24 total runner/policy tests |
|2. Speaker grouping, event identity, leakage-safe inference inputs | **VERIFIED within explicit cached/supplied-phone scope** | Full TRAIN WAV/cache/manifest audit; exact keys; all4 zero39 UNKNOWNs; fit exclusions; strict gold-free inference facade; no raw-text-only claim |
|3. Eligible reproducible joint S/O/I policy | **VERIFIED, bounded mixed-source audit** |12 inner+4 outer gap fits;48 frozen policy pairs; independent C and official EVAL-V3.1 replay of12 pooled outputs;16 checkpoint inference replays |
|4. Frozen architecture/loss/seeds/budget/selection/controls, feasible governed execution | **PARTIAL** | Scientific design unchanged and smoke/resources measured; full driver, all-epoch archive, coordinated seed reset/terminal selection and enforced45-minute stop not integrated/tested |
|5. All launch blockers resolved with receipts | **NOT SATISFIED** | Three precise engineering conditions below remain |

## Exact blockers before any later formal launch

1. **Controlled-run driver missing:** planned `experiments/rq1_train_phase_01/train.py` is not implemented. Current `preflight/agent_a/runner.py` provides reusable fit primitives; `smoke.py` provides a deliberately bounded launcher, not3-seed×4-outer×3-inner orchestration.
2. **Selection/provenance lifecycle not coordinated:** the driver must reset the prescribed seed before **each** branch construction, bind fit/held IDs/scalers/competitors/source hashes, archive every epoch, select only fixed terminal12/5/8, and use the strict per-utterance inference/export boundary. Rolling resumable branch checkpoints alone are not the complete formal-run lifecycle. Validate with mocks/tiny synthetic fixtures, not a full-budget launch.
3. **Budget/stopping enforcement missing at whole-run level:** enforce45-minute wall cap per seed with graceful save/resume, uniform fold budgets and no outcome-directed omissions. The24-utterance pilot and bounded gap refits are measured; full3-branch seed wall/RSS is **NOT_TESTED**. Historical component times are indicative, not a measured complete-seed feasibility guarantee. A timing-based uniform amendment, if necessary, must freeze before new outer outcomes.

These are incomplete orchestration/governance requirements. No current accepted fit-loop failure, material primary leakage or invalid measured event mapping remains. Thus **NO_GO for a disproven method/broken accepted runner is not supported**, but **GO would overstate readiness**. Retain CONDITIONAL_GO with the above launch block. No additional method search, DEV_SEARCH tuning, protected data or formal training is needed to implement/test these controls.

## Resolved MB08 conditions / bounded evidence

- Initial gap helper's N-versus-N+1/shifted-input defect was repaired; valued edges and full-domain keyed outputs verified. Initial passing tests had the wrong expectation; historical failure evidence preserved.
- Initial agent source-absence claims were false. Supervisor recovered MB03 producer and all eligible seed101 inner target streams; independently verified exclusions and CTC checkpoint hashes.
- Complete wrong-type-inclusive joint thresholds now exist. Fixed50 at B=.2: **Macro3 Error F1 .161780, Macro3 Error AP .302723, jointFWR .202222,364 false warnings,920 total warnings**. FWR slightly exceeds .2; per-speaker range .086667–.360000. This is not deployment/safety readiness.
- Strict cached inference rejects arbitrary gap IDs/features, duplicate utterances, malformed/nonfinite inputs, unregistered tokens and gold dictionaries. All four known zero39 events remain covered. Prompt-text-only lineage is **INCONCLUSIVE**, nonblocking only because supplied intended phones are explicitly authorized.
- Three smoke checkpoints,16 bounded gap-fit checkpoints, raw inner/outer scores, full-key policy exports, configurations, cutoffs, input/source hashes, CSVs and independent receipts are preserved. Archived MB03–MB08 artifacts are unchanged.

## Scope and stop

No formal controlled training, GPU allocation, DEV_SEARCH tuning, protected/TEST_SEALED reads, destructive changes or remote push. New work is restricted to preflight artifacts and six requested reports. The operator's existing dirty/untracked work remains preserved. Three native `openai-codex/gpt-6-luna` agents ran in isolated lane directories plus supervisor; only unfinished corrective/audit work was resumed.

**Next authorized action under this result: none automatically.** This task ends at this gate. A later continuation may close the three listed engineering conditions and rerun the gate; it must not silently turn this report into permission for formal training.

Evidence index: `experiments/rq1_train_phase_01/preflight/EVIDENCE_MANIFEST_SHA256.json`. Supporting reports: [preflight](TRAIN_PHASE_01_PREFLIGHT_REPORT.md), [runner](TRAIN_PHASE_01_RUNNER_VERIFICATION.md), [joint policy](TRAIN_PHASE_01_JOINT_POLICY_RESULTS.md), [scientific audit](TRAIN_PHASE_01_SCIENTIFIC_AUDIT.md), [frozen protocol](TRAIN_PHASE_01_FROZEN_PROTOCOL.md).
