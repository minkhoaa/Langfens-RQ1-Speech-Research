# Controlled-run orchestration closure — GO preflight

## Authority and scope

Operator requested completing the three earlier blocked engineering items. Supervisor implemented them rather than launching formal training. Scientific architecture/losses/epochs/seeds/policy budgets remain the original MB08 configuration. Historical conditional reports/machine gate/risk register/manifest are retained in this directory; no MB03–MB08 scientific experiment was restarted. All new executions are plan-only, generated-data fixtures or authorization-denial checks.

## Code and completed contracts

New executable `../../train.py` and modules `runtime.py`, `protocol.py`, `engine.py`, `datasets.py`, `backend.py`, `workflow.py`, `pooled.py`:

1. Default plan only. Full official3seed/4outer/3inner schedule;48 independent branch fits/seed. CPU/two threads. Formal mode needs a separate exact-source/config/environment/seed authorization file and explicit flag; neither is generated here.
2. Seed reset before each branch; fit-only scalers/weights/realized CTC supervision/competitors. Typed gold-free inference and full-domain stable keys. All inner scores and provenance validated before one atomic policy+digest; outer refit/application follows.
3. Rolling update safe point plus immutable checkpoint at every epoch; source/split/scaler identities, optimizer, Python/NumPy/Torch RNG, sampler order/cursor and epoch state retained. Fixed terminal only, loaded for inference. Archive/index/terminal/score/decision/CSV tampering fails closed. Only the just-completed archive commit can repair its missing index, after exact complete-state equality.
4. Persistent cumulative2700s active wall per formal seed, including population construction/hashing/loading through evaluation. Signal/cap stop at safe boundaries; resume uses remaining time. Missing/malformed ledgers fail instead of resetting allowance. No CLI budget extension. Common library import startup is separate process overhead, reported in process wall profile.
5. Direct/E4/fixed comparisons, cached acoustic zero/shuffle/reference-only diagnostics, official EVAL-V3.1 fold/speaker/pooled outputs, raw continuous/full-key decisions, class counts, wrong-type-inclusive FWR, AP/F1/calibration/NLL/top-K diagnostics. Tiny populations correctly mark unavailable P@K null rather than invent a metric.

## Review and amendments

Fresh native reviewer `openai-codex/gpt-6-luna`, runs `301e36c6-5d41-48e1-b010-0d7f877e7f96` and retained final `98403bd6-ad10-4cac-b3e2-b8a133c263d3`:

- Initial P1 population constructor outside budget: fixed by budgeted factory and metered initialization; slow-loader regression stops before fitting.
- P2 policy/seal crash gap: payload and digest now one atomic file; threshold mutation rejected before outer fitting.
- P2 historical epoch-index repair: missing historical entries rejected; only a current completed commit may recover by exact equality.
- Final review: all three resolved, no remaining blocker among reviewed findings. Source was not altered by reviewer.
- Supervisor additionally tested missing/malformed budget ledger and terminal/cached-decision consistency. Numerical E4 competitors/S/O LLRs match the verified MB07 producer within1e-12 on generated inputs.

Freeze history: `SOURCE_FREEZE_INITIAL.json`, `SOURCE_FREEZE_REVIEW_FIXES.json`, `SOURCE_FREEZE_FINAL.json`; **current authority `SOURCE_FREEZE_ACCEPTED.json`**. Earlier test/CLI receipts remain historical, not quietly overwritten.

## Actually executed — MEASURED

- `accepted_all_tests.log`: **91 passed**,12.11s; one benign pre-existing PyTorch std warning on empty-N fixture. Existing65 plus26 orchestration/parity/integrity tests.
- `ACCEPTED_CLI_RECEIPT.json`: exact commands, exits/profile. Accepted-source plan exit0, generated rehearsal exit0, unauthorized formal guard expected exit2 before population construction; no formal output directory created.
- `accepted_cli_rehearsal/seed_101/`: **generated**12utterances/36targets/48gaps, explicitly tiny models,12 inner+4 outer bundles,48 branch fits,96 epoch archives, sealed policies,16 outer original/control streams, complete official per-fold/speaker/pooled evaluation. Process wall **10.236s**, child maxRSS **703,240KiB**; persistent seed budget used **8.865s** of a120s synthetic allowance. This is **not** a45minute formal-seed estimate.
- Selected full-sized H-002/gap/CTC heads exercise the same engine in small generated-input unit fixtures; no corpus fitting.
- Mid-update/end-epoch resume equals uninterrupted model+optimizer+RNG+sampler state. Pause after policy freeze creates no outer fit; resume preserves the exact policy and pooled decisions. Cap resume cannot grant more time; signals preserve safe checkpoints.

Original real TRAIN smoke, full data audit and eligible mixed-source joint-policy artifacts remain the scientific evidence. They were not recomputed/tuned here. Formal full-budget timing/RSS/quality, raw encoder extraction, text-only lineage and deployment safety remain NOT_TESTED/INCONCLUSIVE as appropriate.

## Safe reproduction

From repository root:

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python experiments/rq1_train_phase_01/train.py --mode plan
.venv/bin/python -m pytest -q -p no:cacheprovider \
 experiments/rq1_train_phase_01/preflight/orchestration_v2/test_orchestration.py
```

For a full generated-only CLI replay, choose a **new** directory:

```bash
.venv/bin/python experiments/rq1_train_phase_01/train.py --mode rehearse \
 --seed 101 --output /tmp/phase01-fresh-generated-rehearsal
```

Do not overwrite accepted artifacts or run real formal mode to reproduce the gate. A real formal command was NOT authorized/executed. Final evidence index includes driver source, tests, all six reports, scientific receipts and generated replay artifacts. Stop at GO infrastructure preflight.
