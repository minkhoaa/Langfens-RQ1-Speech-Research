# Controlled experiment proposals — cycle 1

These are research designs only, not approvals to modify Phase 4 or train. Any training requires separate operator authorization under the supplied research-only rules.

## E-001 — Offline deletion-aware alignment feasibility

**Question:** does a reference-phone null path separate omitted phones from genuinely short realized phones using current frozen acoustic frame evidence?

1. Freeze acoustic encoder outputs and intended phone sequence; align once with current forced topology and once with a predeclared optional-phone DP. No neural training.
2. Null transition must have a fixed penalty chosen using TRAIN only; report acoustic-only and transition-prior contributions separately.
3. Run target windows through the same fixed downstream scoring transform if possible; if a learned scorer is unavoidable, this becomes a later approved experiment.
4. Controls: forced aligner, all phones optional at fixed penalty, no-audio lexical prior, short-phone correct subset, matched phone-family/context subset.
5. Evaluate AP O/S/I, O precision at predeclared warning budgets, duration/null rates by class, false nulls among correct phones, per-speaker paired effects. SEARCH results exploratory only.
6. Success: O-tail ranking improves across held-out speaker groups while short legitimate phones are not disproportionately nulled and S/I remain within predeclared noninferiority margins. Failure: apparent gain comes from prior-only cues, short-phone false skips, or one SEARCH speaker.

## E-002 — Representation trajectory ablation (future authorization)

Frozen frame embeddings from selected intermediate layers vs current 12-summary features, same lightweight head, identical TRAIN data, seed, steps and DEV_CAL selection. Layer choice must be fixed from TRAIN/OOF, not SEARCH. Compare local Gaussian pooling, uniform pooling and current EXP-24. Primary outcomes: AP S/O/I and low-FP partial PR; report speaker-level bootstrap. GTX 1650 plan: freeze encoder/cache to disk, train only head; avoid end-to-end finetuning initially.

## E-003 — Exact-gap boundary trajectory (future authorization)

For every utterance with N intended phones create precisely N+1 slots indexed 0..N. Pool windows on both sides of each inferred boundary; leading/trailing windows use explicit silence/padding masks. Train a small scorer on TRAIN gap labels only. Compare EXP-28, energy-only, CTC blank-only, lexical-only and combined models. Audit count equality and event IDs before metrics. Primary metrics exact-gap AP, P@K/recall at fixed FP budget, leading/trailing breakdown. Any missing slots invalidate the run.

## E-004 — OOF multi-teacher transfer (later cycle)

Only if E-001/E-002 show information can be captured cheaply: compare Gold-only, scalar KD-All, specialist-head multitask, and disagreement-weighted feature/relation transfer. Construct speaker-grouped OOF teacher targets; no in-sample teacher predictions. Match initialization, data, optimization budget, checkpoint rule, and seeds. Include acoustic-only/prior-only checks and retain insertion head. Acceptance requires paired-seed S/O AP-tail gains, no meaningful I loss, and provenance audit pass. Historical KD does not license an assumption of superiority.

## E-005 — Calibration/selective risk (non-discrimination study)

Fit calibration only on grouped TRAIN-OOF or DEV_CAL. Compare raw, temperature, isotonic and class-conditional maps. Evaluate Brier against prevalence-constant baseline, reliability, risk-coverage, per-speaker error and abstention coverage. Verify rank invariance where mappings are monotonic. With only two CAL speakers, all generalization claims remain provisional; conformal coverage claims require defensible exchangeability across speakers and enough independent groups.
