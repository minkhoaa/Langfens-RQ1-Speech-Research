# Phase-01 bounded preflight acceptance freeze

Frozen before new fits/evaluations in this phase. Authority: reports/METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md and its machine config, hash-indexed in phase0_receipt.json. Existing dirty work is preserved; HEAD alone is not source provenance.

## Hard scope
- No formal controlled training, DEV_SEARCH tuning, protected/TEST_SEALED reads, gold inference features, destructive changes or remote push.
- CPU only, CUDA_VISIBLE_DEVICES='', two threads per process; no GPU lease requested.
- Three native openai-codex/gpt-6-luna children, each sole writer in its own lane cwd. Existing repository is read-only to children outside their lane. Supervisor alone integrates reports. No managed git worktrees because dirty user/untracked work must not be stashed or committed.
- Agent A owns preflight/agent_a; B owns preflight/agent_b; C owns preflight/agent_c. No child fanout. New source/config freeze receipts precede each lane's new outcomes.

## Immutable intended design
Independent target H-002 (2343→256→128→3) + independent continuous-frame CTC (768→43, blank42) / canonical-mean E4 and fixed .5/.5 S/O fusion. Direct H-002 fallback. Independent MB03 exact-gap head (1614→128→64→2). No shared trainable trunk. Exact losses, scalers, optimizers and seeds are those in MB08 training specification. Unknown intended tokens remain explicit zero vectors with event coverage; never silently AA/phone0. No claim of text-only inference without lineage certification.

TRAIN only: 1800 utterances, 12 speakers, 59562 target events and 61362 gap events; official 4 speaker-grouped folds. Formal seeds 101/202/303, fixed terminal epochs target12/gap5/CTC8; formal runs NOT executed here.

## Bounded new execution
A: synthetic and small eligible TRAIN-subset smoke only (finite forward/backward, parameter change, full optimizer/RNG restart equivalence, keyed N/N+1 coverage, unknown handling and CPU time/RSS). Never execute all formal seeds/folds/budgets.
B: reuse eligible target inner scores; only necessary MB03 gap refits for 12 inner six-speaker fits at historical fixed 5 epochs, seed101, plus at most four nine-speaker gap fits when necessary for a consistently seeded untouched outer assembly. Bound CPU wall to 20 minutes for this policy lane; retain partial state and mark missing metrics NOT_ESTIMABLE if exceeded. No architecture/hyperparameter search. Freeze fit IDs/config/source before fits and policy cutoffs before outer application. Reused outer insertion scores may be valid historical baseline evidence but must not be relabeled a newly consistently seeded pipeline.
C: independent read-only inspection of legacy/current lanes, own bounded lineage/metric tests, risk register and candidate frozen protocol. Reports distinguish source audit from executed tests.

## Policy and gate
Joint B∈{.2,.5,1,2} false warnings/utterance; .8 target/.2 gap allocation; deterministic spec score/SHA tie rule. Thresholds use eligible inner OOF only. Wrong-type S↔O is FP, not total warnings. Preserve raw keyed predictions and fit/exclusion/checkpoint provenance. No outer label feedback.
GO requires validated runner/restart/topology, leakage-safe IDs/inputs, reproducible eligible joint policy OR an explicitly narrow staged-training deferral without full-system readiness claim, frozen feasible formal design, and no material blocking findings. Retain CONDITIONAL_GO for missing joint policy if training infrastructure is valid. NO_GO for material leakage, broken identity/runner or irreproducible primary evaluation. Smoke is not full training; extrapolated compute is not measured full-budget feasibility.

Evidence labels: MEASURED / VERIFIED / INCONCLUSIVE / NOT_TESTED; invalid/missing joint metrics are null / NOT_ESTIMABLE, never zero. No deployment/safety or independent-cohort claims. Stop after integrated gate.
