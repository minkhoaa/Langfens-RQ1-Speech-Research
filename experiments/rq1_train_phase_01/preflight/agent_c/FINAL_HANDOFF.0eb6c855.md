# Final Agent C audit handoff

**Disposition:** The previous Agent C statements that MB03's producer and eligible target inner score provenance were absent were wrong. The exact MB03 PLAIN source and eligible seed101 target inner sources are now independently verified; B completed the allowed bounded gap continuation and froze/applied the joint policies. Those old handoffs are preserved as historical audit trail and superseded by this final review. No A/B/root files were changed by Agent C.

## Independent evidence produced

- `TRAIN_PHASE_01_SCIENTIFIC_AUDIT.md` — scientific/data lineage, source-claim correction, metrics, policy results, and residual blockers.
- `TRAIN_PHASE_01_FROZEN_PROTOCOL.md` — candidate protocol summary; supervisor specification remains authoritative.
- `risk_register.csv` — scoped blockers and nonblockers.
- `independent_train_audit.py`, frozen by `INDEPENDENT_AUDIT_FREEZE.json` — executable TRAIN-only audit. It verified all 1,800 manifest WAV paths/hash/16-kHz headers, full target/gap topology, H6 reference/event order, and the four all-zero 39D references. Per-WAV SHA256s are in `TRAIN_WAV_SHA256.csv`.
- The same script reconstructed all 178,686 seed101 target-inner event joins across the four outer pools from archived direct OOF NPZs and CTC/E4 nested payloads; six-speaker fits exclude inner-held and outer-held speakers. It verified CTC checkpoint hashes and reproduced MB03 keyed AP-S `.4232088151`, AP-O `.2767063392`, AP-I `.1621248970`, Macro3 Error AP `.2873466837`, P@560 `.2428571`.
- `independent_policy_audit.json` and `independent_joint_metrics.csv` — independently rederived 96 cutoff objects for 48 fold/method/budget pairs and applied the frozen inner-only policies to outer TRAIN predictions. Fixed50 neural+E4 at B=.2: AP-S/O/I `.402742/.330783/.174642`, Macro3 Error F1 `.161780`, achieved joint FWR `.202222`; fold joint FWR `.25778/.22889/.14889/.17333`. These are bounded mixed evidence (archived target + new MB03-style gap refits), not formal phase01 or an independent cohort.
- `INDEPENDENT_AUDIT_RECEIPT.json`, `independent_audit_run.log`, and `audit_development_log.md` preserve command outcomes. Preliminary pre-freeze outputs are retained under `pre_freeze_exploratory/`.

## Exact corrections and current blockers

1. **MB03 gap source: resolved correction.** `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py` (SHA256 `dadc7c…7e1ec2`) contains `get_arrays`, `Detector`, `fit_predict` and `main`, including the PLAIN task-only arm, fit scaler, class weights, AdamW, batch, five epochs and dormant speaker branch. B used PLAIN semantics for 12 inner and 4 outer fits; checkpoints, normalizers, fit/held IDs and receipt hashes are retained. This is reproducible source recipe evidence, not replay of historical MB03 checkpoints.
2. **Target inner provenance: resolved correction.** MB07 direct inner OOF, CTC/E4 nested payload and fit receipt, plus its fusion join source and supervisor `inner_target_verification.json`, give complete seed101 eligible target inner scores. Prior Agent C claims of unavailable streams/checkpoints were overbroad; historical neural checkpoint replay remains unavailable but score/source provenance is verified.
3. **B joint policy: bounded evidence complete.** B's 12 inner/4 outer gap fits ran CPU-only within the 20-minute lane cap; thresholds use only eligible inner OOF. Agent C independently checked the budget allocation, score/SHA tie rule, wrong-type S↔O FPs, gap false warnings, speaker exclusions and all cutoff values. B's older `HANDOFF.md` and initial freeze receipt are stale; later `continuation_freeze.json`, `policy_freeze.json`, receipts and `thresholds_frozen.json` are the corrective evidence.
4. **A runner: still not accepted.** Current `gap_input` now constructs N+1 and Agent C's exact first/interior/trailing edge check passed. Current A pytest nevertheless fails 2/6: one fixture gives 12 labels to 13 gap logits; another expects the trailing gap's left-neighbor phone reference to be zero. Current runner SHA256 `19e590…65a78a7` does not match its original `SOURCE_FREEZE.json` runner hash `09ec3a…f51edc`; no amended current A freeze was present at review. A's end-to-end input/E4 parity, real TRAIN runner smoke, label-free keyed inference, and gap/CTC integrated restart exactness remain unverified. Do not edit A; supervisor should resume final runner review after A issues a corrected test/source freeze.
5. **Input lineage:** all TRAIN WAV hashes and H6/reference order are verified. Prompt text does not establish phones: parser uses TextGrid phone-tier canonical entries. Supplied intended phones are safe/in-scope; prompt-only lineage remains INCONCLUSIVE. `ER)`, `D_`, `Z_` are explicit E4 aliases and SPN is in inventory; target reference remains zero39. H002 alignment separately maps OOV to SIL for alignment only. Test these distinct behaviors end-to-end; no AA/phone0 fallback.
6. **Compute/formal scope:** B's bounded gap fits measured 21.74s inner + 11.83s outer on CPU/two threads. Historical CTC inner receipt is 614.95s. Full three-branch, per-seed 45-minute runtime/RSS remains NOT_TESTED; component times are not a full-run feasibility proof. Formal controlled phase01 training remains NOT_TESTED.

**Unresolved blockers:** repair A's failing test fixtures and create a freeze receipt for the exact current runner; validate full audio/feature/CTC/E4 inference parity and keyed output; verify integrated restart equivalence for target/gap/CTC; run an eligible small TRAIN smoke with timing/RSS before formal outer outcomes. Full protocol runtime feasibility remains unmeasured. Prompt-only lineage is a claim-scope limitation, not a blocker when supplied intended phones are explicitly used.

**Recommended next step:** supervisor should use the completed B policy evidence and this source correction, then request A's corrected frozen runner/test bundle for final review. Do not authorize full formal training from A's current failed test state. No independent-cohort, deployment, safety, or unconditional GO claim is made.

### Acceptance report
```acceptance-report
{
  "criteriaSatisfied": [
    {"id":"criterion-1","status":"satisfied","evidence":"Read-only final audit; corrected MB03/target provenance from code and receipts; wrote requested audit, protocol, risk register, script, and receipt artifacts only in Agent C lane."},
    {"id":"criterion-2","status":"satisfied","evidence":"Independent full-TRAIN WAV/reference/event audit, baseline AP recomputation, target-inner provenance joins, B cutoff rederivation, and frozen outer policy metrics are preserved with hashes and commands."}
  ],
  "changedFiles": [
    "experiments/rq1_train_phase_01/preflight/agent_c/FINAL_HANDOFF.0eb6c855.md",
    "experiments/rq1_train_phase_01/preflight/agent_c/TRAIN_PHASE_01_SCIENTIFIC_AUDIT.md",
    "experiments/rq1_train_phase_01/preflight/agent_c/TRAIN_PHASE_01_FROZEN_PROTOCOL.md",
    "experiments/rq1_train_phase_01/preflight/agent_c/risk_register.csv",
    "experiments/rq1_train_phase_01/preflight/agent_c/independent_train_audit.py",
    "experiments/rq1_train_phase_01/preflight/agent_c/test_gap_boundary.py",
    "experiments/rq1_train_phase_01/preflight/agent_c/INDEPENDENT_AUDIT_FREEZE.json",
    "experiments/rq1_train_phase_01/preflight/agent_c/INDEPENDENT_AUDIT_RECEIPT.json",
    "experiments/rq1_train_phase_01/preflight/agent_c/independent_train_audit.json",
    "experiments/rq1_train_phase_01/preflight/agent_c/independent_policy_audit.json",
    "experiments/rq1_train_phase_01/preflight/agent_c/independent_joint_metrics.csv",
    "experiments/rq1_train_phase_01/preflight/agent_c/TRAIN_WAV_SHA256.csv",
    "experiments/rq1_train_phase_01/preflight/agent_c/independent_audit_run.log",
    "experiments/rq1_train_phase_01/preflight/agent_c/audit_development_log.md",
    "experiments/rq1_train_phase_01/preflight/agent_c/pre_freeze_exploratory/"
  ],
  "testsAddedOrUpdated": ["experiments/rq1_train_phase_01/preflight/agent_c/test_gap_boundary.py"],
  "commandsRun": [
    {"command":"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_c/independent_train_audit.py (stdout redirected to independent_audit_run.log)","result":"passed","summary":"Full TRAIN lineage and independent frozen-policy audit; exit 0."},
    {"command":"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_c/test_gap_boundary.py","result":"passed","summary":"Exact N+1 edge/interior phone/acoustic boundary test passed."},
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_b && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m unittest -v test_joint_policy","result":"passed","summary":"6 tests passed."},
    {"command":"cd /home/khoa/Projects/New-Method-Research/experiments/rq1_train_phase_01/preflight/agent_a && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest -q test_runner.py -p no:cacheprovider","result":"failed","summary":"4 passed, 2 failed due test fixture/expectation issues; details documented."},
    {"command":"git -C /home/khoa/Projects/New-Method-Research diff --cached --name-only","result":"passed","summary":"No staged files reported."}
  ],
  "validationOutput": ["All 1,800 TRAIN WAVs verified as 16 kHz and SHA256 indexed; 59,562 targets and 61,362 gaps exactly keyed.","MB03 PLAIN independently reproduced AP-I .1621248969942254 and P@560 .24285714285714285.","48 frozen joint policy pairs / 96 thresholds independently reproduced and applied; full metrics/CSV/JSON receipts preserved."],
  "residualRisks": ["A current runner source freeze is stale and its pytest suite has two failing test assertions/fixtures.","End-to-end E4 producer parity, inference/export and gap/CTC integrated restart exactness are not verified.","Complete phase01 45-minute per-seed wall/RSS feasibility remains NOT_TESTED.","Prompt-text-only phone lineage remains INCONCLUSIVE; supplied phones are allowed."],
  "noStagedFiles": true,
  "diffSummary":"Created Agent C scientific audit, candidate protocol, risk register, independent TRAIN/policy audit script, hashed WAV list, receipts, metrics and final handoff. No A/B or repository-root artifacts modified.",
  "reviewFindings": ["blocker: Agent A current source differs from initial freeze and pytest has two failing fixtures; defer runner acceptance.","corrected: MB03 producer and eligible target inner source are present; B completed bounded 16-gap-fit policy preflight and frozen policy application."],
  "manualNotes":"Full phase01 formal training was not performed. Supervisor owns verdict."
}
```