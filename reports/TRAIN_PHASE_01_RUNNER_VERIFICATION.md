# TRAIN-PHASE-01 runner verification

**Outcome: branch-level infrastructure AND complete controlled-run orchestration VERIFIED by bounded tests; GO preflight.** No formal training was launched. The earlier conditional snapshot is preserved under `preflight/orchestration_v2/prior_reports/`. See `TRAIN_PHASE_01_GO_GATE.md` for the authoritative scope.

Evidence root: `experiments/rq1_train_phase_01/preflight/`.

## Implemented and executed

`agent_a/runner.py` implements separate H-002 target, CTC and MB03 exact-gap networks, fit-only scalers/weights, deterministic seeding, grouped split plans, model fitting and checkpoint restore. Target/gap have no shared trainable parameters. CTC/E4 uses the existing MB07 competitor/event-scoring implementation and MB05 exact sequence-likelihood kernel; fixed S/O fusion remains .5 neural/.5 E4, direct H-002 fallback.

The initial gap helper was **broken**: N rows instead of N+1 with shifted contexts, and an initial test expected the wrong shape. It was repaired before accepted use. Valued N=0/1/3 fixtures now check leading/interior/trailing acoustic/reference contexts. Local inner indices were corrected to 0/1/2. Initial code/tests/failure logs remain in `agent_a/prior_version/` and `tests_bcf2*.log`.

`agent_a/SOURCE_FREEZE_BCF2.json` freezes the accepted runner, tests, smoke launcher and configurations before the final smoke; all hashes match. The original `SOURCE_FREEZE.json` intentionally describes the preserved prior version, not current code. Intermediate C prose about absent amended freeze/two failing tests is superseded by these receipts and supervisor tests.

### MEASURED: eligible TRAIN smoke

- 24 utterances, four from each of six fit speakers: HQTV/LXC/NJS/PNV/SKA/SVBI; outer0 ABA/ASI/BWC and inner-held EBVS/HJK/HKK excluded.
- One epoch only, seed101; independent target/gap/CTC fits. Target C/S/O counts635/89/14; gap negative/I754/8.
- **738 targets + 762 exact gaps = 1,500 keyed rows**. CPU/two threads; **61.955s wall, 865,252 KiB process maxRSS**. This includes smoke scoring/export, not just training. The smoke exporter repeatedly recalculates target logits; its wall time must not be scaled as an optimized full-run estimate.
- Three persisted checkpoints contain model, AdamW optimizer, Python/NumPy/Torch RNG, sampler state/order/cursor, epoch, fit IDs, scalers, seed and frozen-config hash.
- `smoke_artifacts_final/receipt.json` records source/input/checkpoint/prediction hashes, per-utterance continuous H6 feature hashes and phone provenance; JSONL/NPZ and validation log are adjacent. Earlier smoke attempts are retained as engineering history, not separate experiments or favorable-outcome selection.
- The same subset was used for fitting and E4 plumbing. **No generalization metric is inferred from this smoke.** It is not outer OOF and is never substituted for B's eligible policy predictions.

### VERIFIED: restart through actual fit loops

A's seven tests pass. Supervisor added `supervisor/test_runner_integration.py` because A's original next-update fixtures alone did not establish integrated restart.

Six executed fixtures interrupt the **actual** `fit_target`, `fit_gap` and `fit_ctc` after update1 or update3 (an epoch boundary), then resume through two tiny synthetic epochs. Official MLP batch sizes are used, with partial final batches; CTC has33 variable-length sequences. Uninterrupted and resumed final model, optimizer, Python/NumPy/Torch RNG, sampler state, order, cursor, epoch and metadata are **exactly equal**. No formal fold or budget was trained. Additional tests verify all four real zero39 references, every official fit/held split, local inner indexing and smoke-budget rejection.

`supervisor/runner_policy_integration_tests.log`: **21 passed** (A7+B6+supervisor8). `supervisor/final_runner_policy_tests.log`: **24 passed**, adding the strict inference boundary below. The independent EVAL/policy suite separately passed41 tests. Passing synthetic tests do not imply full-budget execution.

### VERIFIED: strict label-free inference and event contract

The accepted export boundary is **`supervisor/validated_inference.py`**, not A's permissive low-level multi-utterance helper in isolation. It accepts typed per-utterance acoustic crops, intended-phone vectors/tokens and identity metadata; derives all N target and N+1 gap rows itself, resets edge contexts at every utterance, and creates explicit dataset/speaker/utterance/domain/index keys. Speaker metadata never enters a network tensor. Callers cannot inject arbitrary gap features or IDs; gold dictionaries, duplicate utterances, inconsistent dimensions, nonfinite features and unregistered phones are rejected.

Tests cover N=0/1/3 and all **four complete real TRAIN unknown-bearing utterances**. `supervisor/unknown_inference_keyed.jsonl` retains their full-domain keyed predictions using random weights for identity/unknown verification only—not model-quality evidence.

Four neural/gap zero39 references remain unchanged. CTC/E4 separately uses explicit legacy aliases ER)→ER, D_→D, Z_→Z (also Y_→Y, AH)→AH), with SPN actual class32; unregistered tokens fail. No AA/phone0 fallback. Audited H002 alignment's OOV→SIL rule is alignment-only and does not rewrite reference vectors/events.

## Orchestration closure — all three former launch blockers resolved

`experiments/rq1_train_phase_01/train.py` now implements the complete frozen official schedule, with `runtime.py`, `protocol.py`, `engine.py`, `datasets.py`, `backend.py`, `workflow.py` and `pooled.py`. Default is plan-only. Formal execution needs an external operator authorization bound to exact source/config/environment/seeds; this report does not authorize execution.

1. **Driver:** all3 inner target/CTC/gap fits and exact eligible score unions precede one atomic sealed policy; outer refit/scoring follows. Direct/E4/fixed, controls, full keys and official fold/speaker/pooled metrics are wired end to end.
2. **Lifecycle:** reset each branch seed before construction; persist fit/held IDs, scaler/source identities and full optimizer/RNG/sampler state; archive/hash every epoch; reload only fixed terminal12/5/8. Missing historical archive indexes, changed archives, terminal/latest mismatch and changed score/decision provenance fail closed. Only a just-completed archive commit can recover via exact full-state equality.
3. **Whole-seed stop:** cumulative2700s active wall includes the population factory, input hashing/loading and all fit/scoring/evaluation. Resume retains used allowance; no cap-reset/budget override. Signals/cap stop at safe update/utterance boundaries with resumable committed state and released lock. In-flight kernels/atomic writes finish before the next check; no new work starts after detected stop.

**MEASURED:**26 new tests; full combined suite **91 passed** in12.11s, one benign pre-existing empty-N warning. Generated-only CLI rehearsal completed four outer/12 inner bundles,48 branch fits,96 epoch archives, controls/policies/official pooled outputs:10.236s process wall and703,240KiB child maxRSS. This used tiny fixture networks and36target/48gap generated events, NOT a formal run or a model performance estimate. Selected real architecture heads also run the new engine on tiny generated-input unit fixtures. New-backend E4 competitor selection/S/O LLRs match the verified MB07 producer within1e-12 on generated inputs.

Fresh-context native `gpt-6-luna` review found a budget-init gap and policy/index crash-integrity issues; all three were corrected and regression-tested. Final review accepts those fixes. `preflight/orchestration_v2/` retains freezes, reviews,26-test source, CLI commands/exit/profile receipts and raw generated artifacts. The final accepted freeze also adds explicit rejection of missing/malformed used-time ledgers, preventing accidental allowance resets on resume. Full formal-seed wall/RSS remains NOT_TESTED, not falsely inferred from this rehearsal.

## Reproduction (verification only)

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m pytest -q -p no:cacheprovider \
 experiments/rq1_train_phase_01/preflight/agent_a/test_runner.py \
 experiments/rq1_train_phase_01/preflight/agent_b/test_joint_policy.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_runner_integration.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_validated_inference.py
```

Executed real smoke command is retained in A's report: run from `agent_a`, `smoke.py --outer-fold 0 --inner-local 0 --utterances-per-speaker 4 --epochs 1 --out smoke_artifacts_final`. **Do not rerun into the completed bundle.** It documents an executed bounded test, not permission for a formal run. Frozen source/config and evidence hashes are indexed in the preflight manifest.

Additional safe commands and complete26-test reproduction: `experiments/rq1_train_phase_01/README.md` and `preflight/orchestration_v2/IMPLEMENTATION_CLOSURE.md`. The default `train.py --mode plan` does not load corpus features or fit models; `--mode rehearse` uses generated arrays/tiny fixture models only. Unauthorized formal execution was tested and rejected before population loading.

**NOT_TESTED:** actual full formal pipeline/epoch-budget execution, full seed duration/RSS, raw waveform re-extraction and external validation. **INCONCLUSIVE:** prompt-text-only phone provenance; supplied intended phones are explicitly in scope. Engineering gate is GO; formal execution remains separately authorized and NOT_RUN.
