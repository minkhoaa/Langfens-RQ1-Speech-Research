# RQ1-UNIFIED-02 Agent B — Q4/Q5/Q7 CPU investigation

## Scope and execution limits

This run was read-only analysis of existing feature caches, source, receipts, and frozen predictions. No CUDA training was started (the required coordination with the supervisor/Agent A has not occurred); no shared cache was modified; no DEV_SEARCH training or new weight sweep was done. Reproduction command: `python experiments/rq1_unified_02/agent_b/diagnostics.py`. Full machine-readable output is `diagnostics.json`.

## Q4 — Rare positive learning and diagnostics

TRAIN gaps have **560 insertion positives / 61,362 (0.9126%)**, so an ordinary random 128-example batch has 1.17 positives in expectation; under an illustrative iid draw with replacement, probability of a batch with zero positive is **30.93%**. DEV_CAL is 102/10,343 (0.9862%; 1.26 positives per 128 and 28.12% zero-positive probability); DEV_SEARCH is 125/20,354 (0.6141%; 0.79 positives per 128 and 45.45% zero-positive probability). The zero-positive estimates are not the exact baseline schedule: training shuffles all rows and takes contiguous minibatches, so they diagnose rarity rather than estimate its realized empty-batch rate. Existing `balanced_gap.py` proposes 64 positive + 64 negative examples per batch, hence 64x as many positives per batch as prevalence sampling, but this is a proposed CUDA training arm and **was not run**.

This supports the mechanism that sparse positives expose the gap head to very few positive examples per update, but does not establish that resampling improves ranking or generalizes. Existing U2 uses class-weighted CE, not unweighted prevalence sampling, and AP is insensitive to a monotone class-prior shift. A balanced sampler tests update composition / repeated positive exposure, not a different class-weight sweep. Need a matched control, same seed/initialization, update count, feature data, optimizer, and DEV_CAL checkpoint rule before interpreting any treatment result.

## Q5 — Independent vs joint heads; parameter-sharing check

Source `experiments/rq1_unified_01/agent_a/train_twohead.py` defines `UnifiedTwoHead` with separate `self.target` and `self.gap` modules. The target consumes 768D frame contexts plus 39D phone reference; the gap consumes the 1,614D edge/adjacent-phone representation. Neither module's forward uses the other's activations. The source's optimizer is constructed over `model.parameters()`, and each minibatch adds target loss and gap loss to a scalar before one backward/update. Thus the two losses are jointly optimized in the same training loop, but there are no shared trainable parameters through which one loss can directly change the other head's weights.

CPU replay in `optimizer_equivalence.py` uses the actual U2 module, identical initialization, per-head CE losses, frozen synthetic inputs, identical optimizer settings, and four identical batch pairs. With dropout disabled to ensure identical effective inputs, joint AdamW versus separate target/gap AdamW produced max absolute parameter difference **0.0 at every step**. This empirically verifies the expected equivalence for disjoint parameter blocks when task batches/step cadence and optimizer state match. It is not a full replay on actual acoustic features and does not test asynchronous task-specific schedules. There are shared frozen frontend-derived feature origins and common step/data schedules, not shared trainable parameters; thus there is no classic shared-representation gradient interference in this architecture.

## Q7 — Existing matched flat U2 vs hierarchical U3

Existing seed-101 arms are matched by receipt: 8 epochs, 59,562 targets, 61,362 gaps; same hashed input caches (train frame `eb8b23a2…`, labels `8e2be711…`, references `4a03dacc…`, train gap contexts `1e4f4727…`), same seed, frozen frontend, target/gap dimensions and training schedule. Parameter counts are U2 653,893 and U3 653,764 (U3 has 129 fewer). Target mechanism differs: flat three-way C/S/O cross entropy versus U3 omission BCE on all targets plus conditional C/S BCE only on non-omissions. Gap mechanism is architecturally identical between arms. Hence this is a useful matched objective/mechanism comparison, but not exactly parameter-count-matched.

Published report values from existing EVAL-V3.1 comparison: hierarchical U3 minus flat U2 AP-S **−0.00895**, AP-O **−0.00920**, AP-I **+0.00583**, macro-3 AP **−0.00411**; CAL20 Macro-3 F1 **0.07234 vs 0.12542**. It does not support a benefit from the hierarchical conditional target objective in this one seed. It is only one seed.

Independent CPU replay of saved seed-101 SEARCH exact-gap predictions shows identical gap labels/metadata and AP **0.1179922 U2 vs 0.1238220 U3** (delta **+0.0058298**), consistent with existing EVAL-V3.1 headline AP-I and the reported U3 increment. The replay computes non-interpolated AP from `p_gap[:,1]` and `gap_Y`; EVAL-V3.1 remains the authoritative headline evaluator.

## Q4 balanced screen evidence

The bounded balanced sampler seed-101 run already exists in this directory; it was not repeated. It used 8 epochs / 480 updates per epoch, 64 positive + 64 negative gaps per batch. Its DEV_CAL insertion AP was **0.0940394**, versus **0.1381110** from the matched flat U2 seed-101 saved predictions; CAL BCE was 0.11898. This is not a matched architecture/seed-initialization comparison because the balanced screen trained only a gap MLP, but its CAL ranking evidence does not warrant repeating balanced sampling. Its exploratory SEARCH AP is in the saved receipt and is not a selection criterion.

## Prepared follow-up (not launched)

Seed-202 flat and hierarchical commands are documented in `SEED202_GPU_COMMANDS.md`. `train_twohead_local.py` is a byte-identical source copy with outputs directed to this owner directory. Agent A owns the GPU slot currently, so neither command has been launched. Request slot separately before each run after Agent A release.

## Interpretation and limits

Evidence now includes severe TRAIN imbalance, structurally disjoint target/gap modules, exact optimizer equivalence under matched per-task updates, and one existing seed-101 U3 comparison. The single-seed U3 contrast argues against this particular hierarchical objective, not hierarchical models generally. No causal intervention on rare-positive exposure or head independence is available yet. DEV_SEARCH has been exposed during prior development and remains descriptive, not an untouched final test.
