# Hypothesis registry — cycle 1

Status vocabulary: proposed / evidence-gathering / accepted for experiment / rejected / inconclusive. No hypothesis is experimentally validated by this document.

## H-001 — Null-path alignment can improve omission evidence

- Rationale: current forced aligner has an emitting state for every intended phone; 207/209 SEARCH omissions receive nonzero model-derived duration. Skip-state HMM literature establishes a topology that permits zero-duration phones, not MDD gains.
- References: Yuan et al., Interspeech 2015, DOI 10.21437/Interspeech.2015-401 (author metadata to verify); Yang et al. (2022), “Phoneme Mispronunciation Detection by Jointly Learning to Align,” ICASSP 2022, IEEE record 9746727 (metadata incomplete).
- LANGFENS evidence: PHASE3_SO_FAILURE_DIAGNOSIS.md; EXP-24 O ranking evidence; forced Viterbi source behavior.
- Change: reference-conditioned monotonic DP with optional intended-phone null transition; separate acoustic emission from transition prior and output alignment uncertainty/competing path margin.
- Mechanism: omitted phone can receive null path; neighboring coarticulation may remain available without fabricating phone occupancy.
- Risks: legitimate short phones may be skipped; optional path may encode lexical expectations; timing/label tuning can leak; deletion may not mean zero acoustic evidence.
- Minimal test: offline paired alignments on same frozen frame posteriors; train no network. Compare forced vs null-path-derived features on existing TRAIN/DEV_CAL and exploratory DEV_SEARCH, with short-phone matched negatives. Only fit any mapping on CAL.
- Controls: unmodified forced alignment; null path with acoustic score only; lexical-prior-only; fixed regularization; identical downstream scorer if possible.
- Metrics: AP O, precision at fixed warning budgets, S/I AP, exact-phone duration-zero rate, false skips among short correct phones, per-speaker bootstrap.
- Acceptance: predeclare practical margin after examining data availability; require held-out speaker consistent O-tail improvement without material short-phone FP/S/I harm. If SEARCH alone improves, inconclusive due exposure.
- Status: evidence-gathering; offline feasibility and source inventory next.

## H-002 — Frozen intermediate SSL frame context improves S/O ranking

- Rationale: current 51D target features collapse a phone into summary statistics and may omit frame trajectories/competitor identity; EXP-24 local attention suggests context can matter but does not prove causal transfer.
- References: Xu et al. (2021), DOI 10.21437/Interspeech.2021-777; Wu et al. (2021), Interspeech 2021; Ye et al. (2022), ICASSP 2022.
- LANGFENS evidence: low-FP oracle ranking ceiling remains modest; EXP-24 exploratory O performance; current feature inventory.
- Change: frozen intermediate layer embeddings, local monotonic reference-conditioned pooling plus explicit null/neighbor comparison.
- Mechanism: retain transitions and alternative-phone evidence rather than only aligned aggregates.
- Risks: pretrained encoder may encode speaker/L1; layer selection overfits 4 SEARCH speakers; gold boundaries are forbidden at inference; compute/cache cost.
- Minimal test: first compare cached layer representations if already available; otherwise proposal only. Train identical lightweight heads with fixed seed/budget, speaker-group OOF and DEV_CAL checkpoint selection.
- Controls: current EXP-24 and current student specialist; acoustic-only and reference-prior-only branches.
- Metrics: AP per class, low-FP precision/recall, per-speaker, S/O/I, risk-coverage.
- Acceptance: paired speaker-held-out AP/low-FP benefit beyond uncertainty and no major I regression; otherwise reject.
- Status: proposed.

## H-003 — Boundary-conditioned gap trajectories improve exact insertion ranking

- Rationale: EXP-28 has promising energy information; scalar gap features may obscure onset/offset evidence. Exact task population is N+1 gaps.
- References: Xu et al. 2021 for prompt-conditioned scoring only; no reviewed paper directly validates LANGFENS exact N+1 semantics.
- LANGFENS evidence: gap/prosodic contribution reported; exact-gap topology is mandatory in EVAL-V3.
- Change: small scorer pools acoustic energy, voicing, spectral change and CTC blank/transition evidence in windows centered at each hypothesized gap.
- Risks: boundary comes from biased aligner; phrase pauses become lexical proxy; leading/trailing gaps omitted by implementation; speaker energy variation.
- Minimal test: use all N+1 indexed gaps; compare with EXP-28 summary baseline under identical labels/scorer; never map to utterance-level detection.
- Controls: energy-only, blank-only, lexical prior-only, current EXP-28, no-reference event-rate baseline.
- Metrics: exact-gap AP, precision/recall/F1, leading/trailing strata, coverage, per-speaker low-FP.
- Acceptance: exact-gap AP and precision at fixed budget improve with no topology omissions; gains must reproduce across speakers.
- Status: proposed.

## H-004 — OOF disagreement-aware multi-teacher learning beats scalar KD only if complementary evidence survives

- Rationale: Phase 3 scalar KD-All does not show stable superiority; specialists have useful ranking. Multi-head/uncertainty-aware transfer may preserve task-specific evidence.
- References: Phase3 OOF provenance and paired report; relevant exact multi-teacher paper not yet selected (do not retrofit a generic citation).
- Change: teacher-specific heads or relation/feature targets with teacher disagreement and confidence, generated via speaker-grouped OOF.
- Mechanism: retain complementary S/O/I structure instead of averaging potentially conflicting scalar probabilities.
- Risks: OOF provenance complexity, overfitting tiny speakers, lexical teacher dominating audio, added model capacity.
- Minimal test: fixed Gold vs scalar KD vs multi-head/OOF target comparison, matched seeds/budgets/checkpoint rule; only after H-001/H-002 feasibility.
- Controls: acoustic-only Gold; prior-only; shuffled-teacher target; teacher disagreement ablation.
- Metrics: AP S/O/I, low-FP PR, FWR policy, speaker-stratified paired differences, calibration separately.
- Acceptance: consistent paired S/O-tail improvement and retained I with no leakage; otherwise scalar KD remains unproven.
- Status: proposed.

## H-005 — Speaker-grouped calibration improves confidence reliability, not ranking

- Rationale: speaker shift and calibration are distinct from discrimination; current calibration uses only two DEV_CAL speakers.
- References: Guo et al. (2017), “On Calibration of Modern Neural Networks,” ICML 2017, https://proceedings.mlr.press/v70/guo17a.html (general calibration reference; not MDD-specific); LANGFENS EVAL-V3 calibration contract.
- Change: compare temperature, isotonic and class-conditional mapping within speaker-grouped OOF/calibration; selective abstention by estimated uncertainty.
- Mechanism: align confidence/risk policies for unseen speakers, without changing ranking for monotone per-class mappings.
- Risks: too few calibration speakers, exchangeability failure, calibrator overfit, threshold policy tuned on SEARCH.
- Minimal test: OOF calibration on TRAIN speakers only, then DEV_CAL diagnostics; require additional unseen-speaker validation before claims.
- Controls: raw score, temperature, isotonic; class prevalence baseline; per-speaker reliability.
- Metrics: Brier vs prevalence baseline, reliability plots, risk-coverage, selective risk, AP invariant check, per-speaker degradation.
- Acceptance: reliable improvement over baseline across held-out speaker groups; ECE alone cannot pass.
- Status: proposed.

## H-006 — Articulatory auxiliary tasks help rare S/O only if phone diagnosis is preserved

- Rationale: low resource and sparse error classes may share manner/place/voicing structure; phonological-level MDD literature provides a plausible task family, not LANGFENS evidence.
- References: Shahin, Epps & Ahmed (2023 preprint), arXiv:2311.07037; Shahin & Ahmed (2024), Interspeech 2024.
- Change: auxiliary manner/place/voicing heads on frozen/local acoustic representations, with gradient-conflict monitoring in a separately authorized training experiment.
- Risks: attributes correlate with reference/lexical identity, blur fine phone distinctions, loss competition, phoneme inventory mapping errors.
- Minimal test: matched multi-task vs no auxiliary head with identical seed/budget; speaker-grouped evaluation and per-phone-family breakdown.
- Controls: single-task, each auxiliary independently, shuffled attribute labels as negative control.
- Metrics: AP S/O/I, Macro-3 Error F1, rare-phone and family AP, fixed-FP precision, attribute accuracy.
- Acceptance: target AP/low-FP gains without loss of exact diagnosis; else reject.
- Status: proposed.
