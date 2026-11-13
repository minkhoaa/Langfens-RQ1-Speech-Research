# LANGFENS — Consolidated Experiment-First Research Report

**Status:** Active research; substantive exploratory capability gain demonstrated through score composition. This is not a blind-test result or a declaration that the research program is complete. All DEV_SEARCH results below are development-exposed. No protected/sealed holdout was accessed.

## Executive summary

The strongest newly demonstrated result is a **fixed-formula, head-wise score fusion** using EXP-24 for omission, EXP-36 plus a matched Gold-Only student for correct/substitution conditional scores, and EXP-43 plus the student for exact-gap insertion. Across five matched student seeds it reached **Macro-3 Error AP 0.3018**, versus **0.2892 for EXP-36** and **0.2316 for Gold-Only**. Its F1 trade-off is important: Macro-3 F1 was 0.3178, below EXP-36's 0.3311 and the simpler head-wise blend's 0.3296.

The more conservative head-wise blend (EXP-36 for target S/O, EXP-43 for exact-gap I, each blended 0.8/0.2 with matched Gold-Only) replicated across five seeds and averaged Macro-3 AP/F1 **0.2955/0.3296**. Confidence-aware gating selected the constant blend (gate slope 0); adding EXP-24 to a whole target vector did not help, but substituting its O score specifically improved Macro-3 AP.

Several TRAINed objective and architecture screens were negative. The successful finding is currently an inference-time composition, not a newly trained model that surpasses the established ensemble on every metric.

## Data, evaluation, and safety

- Authoritative split (`data/split_specs/L2_ARCTIC_SPLIT.yaml`): TRAIN 1,800 utterances / 12 speakers; DEV_CAL 300 / 2; DEV_SEARCH 599 / 4.
- DEV_SEARCH EVAL-V3 population: 19,755 targets and 20,354 exact gaps; supports S=1,768, O=209, I=125 (and 17,778 correct targets).
- Intended-reference, audio-derived inference contract retained for the newly trained students and score compositions. Target probability vectors were simplex-preserving; insertions were scored on exact N+1 gaps.
- TRAIN-only fitting/normalization; DEV_CAL used for checkpoint/weight/policy selection as declared per experiment. DEV_SEARCH was assessment only in each run, but it is repeatedly exposed across this campaign and must be treated as exploratory.
- No FWR<=0.20 gate was enforced. FWR is reported unfiltered. Calibration budgets constrain DEV_CAL only and do not guarantee the same DEV_SEARCH FWR.

## Development benchmarks and five-seed results

Official exploratory development benchmarks supplied for comparison: EXP-24 O AP **0.225092**; EXP-36 Macro-3 Error AP **0.289233**; EXP-43 exact-gap I AP **0.285056**; EXP-29 Macro-3 F1 **0.342914** (CAL_MAX_ERROR_MACRO_F1; Macro-4 F1 0.482871).

The following are five-seed means on the same exposed DEV_SEARCH population. Threshold-based F1 uses DEV_CAL-fitted CAL_MAX_ERROR_MACRO_F1 per system/seed. FWR is unfiltered.

| System | AP S / O / exact-gap I | Macro-3 AP | F1 S / O / I | Macro-3 F1 | Macro-4 F1 | FWR |
|---|---:|---:|---:|---:|---:|---:|
| Gold-Only GRU-64 | .3645 / .1503 / .1801 | .2316 | .3872 / .2518 / .2567 | .2986 | .4447 | 5.65 |
| EXP-36 | .3871 / .2047 / .2759 | .2892 | .4214 / .2712 / .3007 | .3311 | .4705 | 5.53 |
| Fixed head-wise blend | .3978 / .2001 / .2887 | .2955 | .4077 / .2911 / .2900 | .3296 | .4668 | 5.98 |
| **Omission-specialist score composition** | .3917 / **.2251** / .2887 | **.3018** | .4098 / .2535 / .2900 | .3178 | .4584 | 5.85 |

**Interpretation:** the omission-specialist composition raised Macro-3 AP by +0.0126 over EXP-36, with S/O/I AP deltas of approximately +0.0046/+0.0204/+0.0128. However, its Macro-3 F1 was -0.0133 versus EXP-36 and -0.0118 versus the fixed head-wise blend. It is the current AP-leading development candidate, not a universally best operating model. Five seeds vary the student component; the teacher score arrays and DEV_SEARCH speakers are shared, so this is not five independent evaluation cohorts.

## Operating-point analysis

The fixed omission-specialist composition's five-seed mean results across DEV_CAL-selected risk budgets:

| DEV_CAL budget | DEV_SEARCH Macro-3 F1 | DEV_SEARCH FWR | Mean O precision / recall / F1 |
|---:|---:|---:|---:|
| 0.2 | .1587 | .1523 | .4000 / .0957 / .1544 |
| 0.5 | .2179 | .3509 | .3197 / .1866 / .2356 |
| 1.0 | .2858 | .9015 | .3197 / .1866 / .2356 |
| 2.0 | .2917 | 1.5760 | .1718 / .4833 / .2535 |
| Unconstrained | .3178 | 5.8531 | .1718 / .4833 / .2535 |

These are policy trade-offs, not candidate filters. AP is threshold-invariant. At the .2 budget, S/O recall is especially low; at the unconstrained operating point, warning volume is very high. No FWR cutoff is being treated as mandatory.

## TRAINed screening experiments and negative findings

All times are local GTX 1650 CUDA wall-clock unless noted. Metrics use DEV_SEARCH and are exploratory. Paths contain scripts, checkpoints, scores, and receipts.

| Experiment | Change/configuration | DEV_SEARCH result | Decision |
|---|---|---|---|
| EXP24-inspired gated residual | New small per-stream gated-residual acoustic model; one seed, ~12 s | AP S/O/I .3461/.1070/.0141; Macro-3 AP .1557 | Clear negative; do not promote. F1 reported at fixed .5, so not comparable to CAL-selected F1. `experiments/phase4/acoustic_dependence/exp24_gated/` |
| Focal GRU, gamma 1.5 | GRU-64, 12 epochs, seed 101; 47.8 s | AP .3585/.1610/.2002; Macro-3 AP .2399; Macro-3 F1 .3012 | Small/mixed single-seed gain; substitution and Macro-4 regressed. `experiments/agent_b_training_optimization/` |
| Focal GRU, gamma 2.0 | Matched seed/config; 46.4 s | Macro-3 AP/F1 .2348/.2971; Macro-4 F1 .4377 | Worse than gamma 1.5; stop gamma escalation. `experiments/agent_b_focal_gamma2p0/` |
| Exact-gap positive-weight sweep | GRU-64; weights 1/3/10/20; 203 s | DEV_CAL selected weight 3; seed101 Macro-3 AP/F1 .2414/.3059; I AP/F1 .2019/.2492 | Modest seed101 signal did not persist in seed202; not promoted. `experiments/exact_gap_insertion_weight_sweep/`, `..._seed202/` |
| Gap-only focal, gamma 1.5 | Same GRU; seeds 101 and 202 | Seed101 Macro-3 AP/F1 .2412/.3016, I AP .1996; seed202 Macro-3 AP/F1 .2239/.2978, I AP .1505 | Seed101 I ranking gain did not replicate. `experiments/gap_only_focal_screen/`, `..._seed202/` |
| Omission binary specialist | GRU-64, class-weighted BCE, ~42 s | Corrected all-target composition: AP S/O/I .3529/.1452/.2759; Macro-3 AP .2580 | Worse than EXP-36/head-wise blend; binary O-vs-correct-only first attempt excluded S and was not a valid full-population comparison. `experiments/omission_specialist_student/`, `experiments/omission_all_negative_student/` |
| Omission class-weight sweep | Factors 1/2/4; 188 s | DEV_CAL selected baseline factor 1; SEARCH Macro-3 AP/F1 .2369/.3001 | Higher weights rejected on CAL; no joint improvement. `experiments/omission_weight_sweep/` |
| Pairwise-ranking auxiliary loss | GRU-64, lambda .1; ~178 s | Correctly selected on DEV_CAL error Macro-3 AP; SEARCH Macro-3 AP/F1 .2330/.2980 | Below Gold-Only .2369/.3001. Original run mistakenly selected target macro including CORRECT; preserved and corrected rerun documented separately. `experiments/pairwise_ranking_aux/`, `experiments/pairwise_ranking_aux_corrected/` |
| Transformer student | 2-layer, d=64, 4 heads, 12 epochs; 45 s | AP S/O/I .3316/.1110/.1346; Macro-3 AP/F1 .1924/.2484 | Underperformed GRU across classes; retain GRU. `experiments/transformer_student_screen/` |
| Hard-negative mining | GRU-64, lambda .1/top-128, 53 s | Macro-3 AP/F1 .2377/.2922; Macro-4 F1 .4357; FWR 6.72 | Negligible AP gain, F1/risk worse. `experiments/hard_negative_mining/` |

## Fusion and calibration experiments

- **EXP-TSB-01:** shared convex EXP-36/Gold-Only blend selected alpha=.8 on DEV_CAL Macro-3 AP; one-seed SEARCH Macro-3 AP/F1 .2962/.3386. `experiments/teacher_student_blend_v1/`.
- **EXP-TSB-02:** alpha=.8 fixed across student seeds 202/303; positive Macro-3 AP and F1 deltas versus matched Gold-Only. `experiments/teacher_student_blend_seed_replication/`.
- **Shared three-way EXP-36/EXP-43/student:** selected on CAL; SEARCH Macro-3 AP/F1 .2955/.3350; did not improve the shared two-way AP. `experiments/teacher_threeway_blend/`.
- **Head-wise blend:** target .8 EXP-36/.2 matched student; gap .8 EXP-43/.2 student. Fixed-weight replication across seeds 202/303 and 404/505; five-seed evidence in `experiments/teacher_headwise_sourcewise_blend/`, `..._replication/`, `..._seed404_505/`.
- **Four-source head-wise sweep:** EXP-24 received zero selected weight under its predeclared S/O target-AP objective. `experiments/teacher_fourway_headwise_blend/`.
- **Confidence gate:** DEV_CAL selected gate slope b=0; static alpha=.8 retained. `experiments/confidence_gated_blend/`.
- **Risk-budget curves:** five seeds, five budgets, both fixed blends; all points retained irrespective of SEARCH FWR. `experiments/teacher_student_blend_operating_points/`, `experiments/omission_fusion_operating_points/`.

## Reproducibility and next research direction

The new training checkpoints and evaluation artifacts are retained in the experiment directories above. Baseline provenance and seed-specific hashes/commands are in their run receipts/reports. The working tree contains many pre-existing untracked research artifacts; none were staged or removed.

**Current decision:** keep the fixed head-wise blend as a stable general-purpose ranking candidate; keep the EXP-24 omission-specialist composition as the Macro-3 AP-leading exploratory candidate, with its F1/risk trade-off explicit. Do not claim blind-test or population-level superiority.

**Next useful work:** develop stronger frame-level/SSL acoustic representations or train a distilled student for the fixed fusion using only authorized TRAIN-derived teacher targets. More weight/gate searches on the already exposed DEV_SEARCH are unlikely to establish generalization. A separately authorized independent evaluation would be needed for confirmatory claims; no protected holdout was accessed in this cycle.
