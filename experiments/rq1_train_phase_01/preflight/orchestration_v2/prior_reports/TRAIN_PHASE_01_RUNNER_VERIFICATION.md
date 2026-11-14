# TRAIN-PHASE-01 runner verification

**Outcome: branch-level infrastructure VERIFIED; complete controlled-run orchestration remains incomplete.** No formal training was launched. See `TRAIN_PHASE_01_GO_GATE.md` for the authoritative disposition.

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

## Remaining launch conditions (not concealed as passed)

1. There is **no complete formal seed/fold command driver** implementing the whole 3-seed, 4-outer/3-inner protocol. The planned root `experiments/rq1_train_phase_01/train.py` is still absent. Reusable fit functions and bounded `smoke.py` are not that driver.
2. Coordinated per-module seed reset before construction, all-epoch archival, fixed-terminal checkpoint selection, scorer/normalizer/fit-ID plumbing and strict inference boundary must be integrated in that driver. Existing fit loops save a resumable rolling checkpoint; they do not archive every epoch automatically.
3. The protocol's45-minute per-seed hard stop, graceful save/resume and uniform-budget enforcement are specified but **not jointly implemented/tested** by a formal orchestration driver. No full-run wall/RSS measurement is claimed. Resource estimates from historical runs are not a replacement for this control.

These are incomplete orchestration/governance requirements, not unresolved corruption of the accepted branch tests or measured policy event mapping. No formal launch is authorized by this report. No new method search or metric tuning is needed to complete them.

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

**NOT_TESTED:** complete formal pipeline/terminal-budget execution, full seed feasibility, raw waveform re-extraction and external validation. **INCONCLUSIVE:** prompt-text-only phone provenance; supplied intended phones are explicitly in scope.
