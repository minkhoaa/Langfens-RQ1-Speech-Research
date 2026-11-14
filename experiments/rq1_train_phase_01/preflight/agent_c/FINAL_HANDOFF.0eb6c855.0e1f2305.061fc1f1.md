# Final reconciliation — Agent C

This brief reconciliation supersedes the earlier Agent C “two failing tests / missing A freeze” status. No repository/root or A/B files were changed in this reconciliation; no training, smoke, audit replay, or tests were run here. One same-backend continuation had previously failed with `Codex SSE response headers timed out after 300000ms`; the supervisor preserved its failure snapshot. No model/backend switch or further retry was made.

## Corrected A and mapping status

- **A7 is now VERIFIED:** `SOURCE_FREEZE_BCF2.json` freezes the current runner `19e590e565d509b72a76bad2d3271a5fa1e3d243e0e49f0fa4925291e65a78a7`, current `test_runner.py` `6698130085b93d1c6ab8b721ec09a1083513e3ab7a3df54bf8f7f63de4333495`, and config. This current test hash matches the A hash already recorded in Agent C’s independent audit freeze. The old pre-correction `SOURCE_FREEZE.json` is historical, not the current authority; no audit-freeze amendment is needed.
- A’s actual in-fit 24-utterance TRAIN smoke receipt is **MEASURED plumbing only**: 738 targets / 762 gaps; CPU, two threads, one epoch; 61.955s and 865,252 KiB max RSS. Checkpoint/prediction/input hashes are retained in `agent_a/smoke_artifacts_final/receipt.json`. It is not a performance estimate or formal run.
- Supervisor’s accepted `validated_inference.py` provides a strict per-utterance cached inference/export boundary: it constructs event IDs and N+1 gap features itself, rejects malformed/nonfinite/gold-dictionary inputs, checks phone aliases, and preserves explicit unknown status. `unknown_inference_keyed.jsonl` covers all four actual unknown utterances. These predictions use smoke/random weights and are identity/topology evidence only, not scored model results. This resolves the generic mapping/export boundary **for the accepted facade**, not for arbitrary low-level helper use or a raw-waveform frontend.
- Supervisor’s `runner_policy_integration_tests.log` records **24 passed** (A7+B6+integration8+strict-facade3), with one benign empty-N standard-deviation warning. In the preceding bounded reconciliation, Agent C independently reran A’s seven tests and the eight actual-loop integration tests; both passed. The final strict-facade run is supervisor evidence, not rerun in this reconciliation.

## Scientific/data evidence carried forward

The prior false assertion that MB03 gap producer / eligible target inner sources were unavailable remains corrected: MB03 `train_speaker_adversary.py` has the exact PLAIN recipe; MB07 nested neural and CTC/E4 source, payload, receipts and exclusion IDs provide eligible target inner OOF. B completed 12 inner and 4 outer bounded gap fits with retained IDs/checkpoints/normalizers, froze 48 policy pairs from eligible inner OOF, and applied the joint policies. Agent C independently recomputed MB03 AP-I `.162124897` / P@560 `.242857`, validated all 1,800 TRAIN WAV hashes, four zero39 references and target-inner joins, and independently recomputed/applied the frozen thresholds. Prior long reports, CSVs, source hashes, and pre-freeze exploratory snapshots are preserved.

**Phone contract:** explicitly supplied intended phones are safe/in-scope; prompt-text-only derivation remains INCONCLUSIVE because canonical phones trace to TextGrid annotation, not a verified prompt dictionary. The target reference vectors for `ER)`, `D_`, `SPN`, `Z_` remain zero39. E4’s explicit `ER)`→`ER`, `D_`→`D`, `Z_`→`Z` aliases and SPN inventory class are distinct from that zero-vector target representation. H-002 uses the documented `[SIL]` alias only for OOV acoustic forced alignment. Do not claim text-only inference or use AA/phone0 fallback.

## Remaining launch blockers and disposition

The remaining honest gaps are **controlled-run orchestration**, not a material baseline leakage finding:

1. A full formal outer-fold/seed driver that integrates target, CTC/E4, gap, frozen policies and required artifacts is not implemented/validated end-to-end.
2. All-epoch checkpoint archival, branch-specific seed reset and lifecycle controls are not integrated together in a single controlled driver.
3. The 45-minute global stop/resume contract is not enforced end-to-end, and complete formal-seed wall/RSS feasibility remains unmeasured. A small one-epoch smoke is not that measurement.

The accepted strict facade, independent source/exclusion audit, B joint-policy evidence and smoke support **CONDITIONAL_GO for bounded implementation/launch-contract completion**, not immediate formal training. No NO_GO-level baseline leakage or identity failure was found. Full formal phase01 training remains NOT_TESTED; deployment, safety, prompt-only and independent-cohort claims remain outside evidence.

### Acceptance report
```acceptance-report
{
  "criteriaSatisfied": [
    {"id":"criterion-1","status":"satisfied","evidence":"Reconciled against A BCF2 and supervisor strict-facade/integration receipts without changing A/B/root files or rerunning training/audits."},
    {"id":"criterion-2","status":"satisfied","evidence":"This handoff identifies corrected source/test status, accepted mapping boundary, preserved provenance, and exact remaining launch blockers."}
  ],
  "changedFiles":["experiments/rq1_train_phase_01/preflight/agent_c/FINAL_HANDOFF.0eb6c855.0e1f2305.061fc1f1.md"],
  "testsAddedOrUpdated":[],
  "commandsRun":[
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest -q test_runner.py -p no:cacheprovider","result":"passed","summary":"Final A7 suite passed in prior bounded reconciliation; not rerun in this final pass."},
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/supervisor && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest -q test_runner_integration.py -p no:cacheprovider","result":"passed","summary":"8 actual-loop integration tests passed in prior bounded reconciliation; not rerun in this final pass."}
  ],
  "validationOutput":["A BCF2 runner SHA256 19e590e565d509b72a76bad2d3271a5fa1e3d243e0e49f0fa4925291e65a78a7; test_runner SHA256 6698130085b93d1c6ab8b721ec09a1083513e3ab7a3df54bf8f7f63de4333495; BCF2 freeze binds both.","Supervisor final integration log: 24 passed, one benign warning; strict unknown output retained."],
  "residualRisks":["Full formal fold/seed driver absent.","All-epoch archive and branch seed/lifecycle controls not integrated end-to-end.","Global 45-minute stop/resume enforcement and full-run feasibility remain unverified."],
  "noStagedFiles":true,
  "diffSummary":"Added one concise Agent C reconciliation handoff only.",
  "reviewFindings":["resolved: prior A test/freeze status superseded by BCF2 and final integration receipts.","nonblocking scope: supplied intended phones allowed; prompt-only lineage remains inconclusive.","blocker to immediate formal launch: orchestration/lifecycle/resource contract gaps remain."],
  "manualNotes":"Supervisor owns verdict; CONDITIONAL_GO is appropriate for completing bounded launch infrastructure, not formal-launch authorization."
}
```