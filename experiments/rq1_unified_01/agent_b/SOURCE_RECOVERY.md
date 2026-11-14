# RQ1-UNIFIED-01 Agent B: source recovery and distillation readiness

## Scope

This note records the available Phase 4 expert interfaces and whether their predictions can safely supervise TRAIN examples. It separates the source compositions' development scores from training targets. No neural training or GPU work was performed here.

## Recovered Phase 4 recipes

The reproducible omission-specialist score composition is implemented and reported in `experiments/omission_specialist_score_fusion/`:

- **Omission:** EXP-24's omission probability.
- **Correct/substitution:** residual non-omission probability is distributed using the original EXP-36 and matched Gold-Only conditional ratio, with weights 0.8 and 0.2.
- **Insertion:** 0.8 EXP-43 gap score plus 0.2 matched Gold-Only gap score.
- Thresholds are selected per system and seed on DEV_CAL; DEV_SEARCH is development-exposed.
- Event order was checked across the three expert sources and matching Gold-Only predictions. DEV_SEARCH has 19,755 target rows and 20,354 exact-gap rows.

The selected fixed blend is separately specified in `experiments/teacher_fourway_headwise_blend/`: target weights EXP-36 0.8 + Gold-Only 0.2; gap weights EXP-43 0.8 + Gold-Only 0.2. It is not neural distillation and not the omission-specialist composition.

The Phase 4 macro-3 AP five-seed means in `experiments/omission_specialist_score_fusion/REPORT.md` are 0.3018 (omission-specialist) and 0.2955 (fixed blend); macro-3 F1 means are 0.3178 and 0.3296. These are score-composition benchmarks only.

## Artifact and interface availability

| Source | Available evidence | TRAIN supervision readiness |
|---|---|---|
| EXP-24 | Checkpoint `experiments/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINTS/jsac_monotonic_model.pt`; raw search/rep target and gap prediction JSON; training receipt/spec says it fit TRAIN. | **Not safe.** No speaker-held-out TRAIN outputs were found. Forwarding the fitted checkpoint over TRAIN would create in-sample teacher scores. |
| EXP-36 | Canonical component-composition replay scores and frozen recipe in `experiments/evaluation_v3/closure/composition_replays/EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION/`; exact row counts, hashes, and composition checkpoint/source evidence. | **Not safe.** The replay is explicitly DEV_CAL/DEV_SEARCH CPU score composition, not TRAIN OOF output. |
| EXP-43 | Canonical component-composition replay scores and frozen recipe in `experiments/evaluation_v3/closure/composition_replays/EXP-43-CONTINUOUS-AUTONOMOUS-FRONTIER-EXPANSION/`. | **Not safe.** The replay is explicitly DEV_CAL/DEV_SEARCH CPU score composition, not TRAIN OOF output. |
| Phase 3 teacher OOF | Provenance in `reports/PHASE3_TEACHER_OOF_PROVENANCE.md` and V2 under `research/student/kd/`. It documents three-fold speaker-held-out champion/EXP01 replica predictions. | Available as a **different teacher family**, not an EXP-24/36/43 substitute. Its DEV_CAL isotonic map was extrapolated onto TRAIN and documented as shifted; prefer raw OOF logits/scores if those arrays are restored. |

The canonical EXP-36/43 receipts identify score-only replay (`training_or_optimizer: false`, `no_new_neural_forward: true`) and score hashes for development partitions. This makes the recipe reproducible, but does not create OOF TRAIN targets. Existing EXP-24 dev JSON predictions likewise do not supply TRAIN OOF supervision.

The replay `.npz` interface for EXP-36 and EXP-43 has `p_target` `(N,3)`, `p_gap` `(M,)`, `target_meta` `(N,2)`, and `gap_meta` `(M,2)` arrays. On DEV_SEARCH these have shapes `(19755,3)`, `(20354,)`, `(19755,2)`, and `(20354,2)`. EXP-24's historical DEV_SEARCH JSON target rows expose `prob_correct`, `prob_substitution`, and `prob_omission`; its separate gap file contains gap probabilities. Exact event alignment was validated by the fusion run; the raw historical JSON schema itself has no TRAIN OOF rows.

## Distillation decision and executable design

Until held-out TRAIN predictions for the specific expert heads exist, K1-K3 should remain **unrun**. Fitting a student to an expert's in-sample predictions would leak the expert's TRAIN fit into the student targets and would not test transfer to unseen examples. Development predictions are evaluation evidence and must not be copied into TRAIN.

Once Agent A's K0 checkpoint and an OOF target set are both available, use a matched gold-only control and one head at a time:

1. **K1 omission:** add soft binary BCE/KL on `pO_teacher` for the omission-specialist head; retain gold target loss and all C/S and gap losses unchanged.
2. **K2 conditional C/S:** only if the OOF teacher exports a valid 3-way vector, normalize its non-omission mass as `qS = pS/(pC+pS)` and distill conditional logits on non-omission targets. Omission examples contribute no conditional gold loss; keep the omission loss active there.
3. **K3 insertion:** add BCE/KL against OOF `pI_teacher` at exact N+1 gap indices; retain gold insertion BCE.
4. **K4 joint:** combine only interventions whose individual matched runs improve their intended class without material regression in the other S/O/I APs. The dependent correlation between teachers is unknown; the Phase 4 composition weights are not probabilities of independent evidence.

For each head, retain direct gold supervision, report the soft-target coefficient and temperature, use identical samples/initialization/seed/epoch budget to K0, and pick checkpoints on DEV_CAL. Do not use DEV_SEARCH repeatedly for selection. The Phase 3 OOF student path is a possible separate proxy experiment, but it cannot validate transfer of the specific EXP-24/36/43 Phase 4 advantages.

## Current blocker and recommendation

The supervisor reports the CUDA driver is currently unavailable in the project environments. Source preparation can proceed, but no substantial training is authorized until the GPU is available. There is also no valid EXP-24/36/43 OOF TRAIN target artifact. Proceed with Agent A's gold-only two-head architecture when the assigned CUDA slot is available; do not add distillation based on development vectors or in-sample checkpoint forwards. The smallest later transfer experiment is K1 alone after producing OOF omission predictions from speaker-fold EXP-24-compatible models, followed by independent K2/K3 only if the matched K1 result is informative.

## Exact source interfaces, fixed formulas, and CPU preparation

The recovered prediction interfaces are aligned by `(utterance_id, slot_idx)` for target rows and `(utterance_id, gap_idx)` for exact gaps. EXP-24's frozen replay NPZ uses `p_target (N,3)`, `p_gap (M,)`, `target_logits (N,3)`, `gap_logits (M,)`, and both metadata arrays `(rows,2)`; class order is `[CORRECT, SUBSTITUTION, OMISSION]`. EXP-36 and EXP-43's composition replay NPZs expose `p_target (N,3)`, scalar `p_gap (M,)`, and the two metadata arrays. Development shapes are 10,043/10,343 (CAL) and 19,755/20,354 (SEARCH). These are score interfaces, not model training interfaces.

The EXP-36 object is a seven-component score composition, not a standalone EXP-36 neural specialist. Its DEV_CAL-fitted normalized target weights in the frozen evidence order `(EXP-22, EXP-17, EXP-21, EXP-10, EXP-24, EXP-28, EXP-34)` are `(0.10, 0.15, 0.10, 0.10, 0.20, 0.15, 0.20)`; gap weights are `(0.08, 0.12, 0.12, 0.18, 0.12, 0.18, 0.20)`. EXP-43 is a nine-component score composition in order `(EXP-22, EXP-17, EXP-21, EXP-10, EXP-24, EXP-28, EXP-34, EXP-38, EXP-42)`; target weights are `(0.08, 0.12, 0.08, 0.08, 0.12, 0.14, 0.14, 0.16, 0.08)` and gap weights `(0.06, 0.08, 0.08, 0.12, 0.10, 0.14, 0.16, 0.14, 0.12)`. Both fit only on DEV_CAL and their receipts explicitly say `training_or_optimizer: false`, `no_new_neural_forward: true`, and CPU frozen score composition. They have no TRAIN rows.

The fixed omission-specialist score formula (not a distillation recipe) is:

```text
base = 0.8 * EXP36.p_target + 0.2 * matched_gold_only.p_target
q_omission = EXP24.p_target[:, OMISSION]
residual = 1 - q_omission
q_correct = residual * base[:, CORRECT] / (base[:, CORRECT] + base[:, SUBSTITUTION])
q_substitution = residual * base[:, SUBSTITUTION] / (base[:, CORRECT] + base[:, SUBSTITUTION])
q_insertion = 0.8 * EXP43.p_gap + 0.2 * matched_gold_only.p_gap
```

The separate fixed head-wise comparator uses the full target vector `0.8*EXP36 + 0.2*Gold-Only` and the same exact-gap formula. For EXP24's model interface, the source checkpoint is `experiments/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINTS/jsac_monotonic_model.pt`; its spec says 30 epochs of MonotonicLocalAttention. This checkpoint was fit using TRAIN, so forwarding it over TRAIN is in-sample and is explicitly excluded as teacher supervision.

Added `agent_b/distillation.py` as CPU-side preparation for a later valid OOF handoff. It requires separate EXP-24 target, EXP-36 target, and EXP-43 gap NPZs plus receipts labeled `TRAIN_OOF`; checks prediction hashes, exact canonical TRAIN ID/order, probability shapes/ranges, unique IDs, and per-speaker held-out fold provenance; then writes only the head targets needed for K1-K3. It also provides soft omission BCE, conditional C/S KL (masked on gold omission rows), and insertion BCE helpers. No arrays were produced because no qualifying OOF source predictions exist. The script does not run inference or training.

Candidate comparison remains gold-only K0 versus one intervention at a time, with identical seed, initialization, schedule, samples, and direct gold losses. Suggested first candidate is K1: EXP-24 OOF omission soft loss at coefficient 0.25, retaining target gold cross-entropy and the complete insertion gold loss. Only if K1 supports transfer should K2 (EXP-36 conditional C/S soft loss, same coefficient, non-omission gold rows) and K3 (EXP-43 exact-gap insertion soft loss) be run. Report each intended class AP and both untargeted error-class APs, select on DEV_CAL, and treat exposed DEV_SEARCH as exploratory. This is score-head transfer; it does not repeat H-004 feature-cosine distillation.
